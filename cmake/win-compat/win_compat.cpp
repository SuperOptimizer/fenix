// win_compat.cpp — Win32 implementations of the POSIX <sys/mman.h> shim. This is the ONE TU
// where <windows.h> is included, keeping its macros out of the header-only fenix tree.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winioctl.h>   // FSCTL_SET_SPARSE

#include <io.h>
#include <fcntl.h>
#include <cstdlib>
#include <cstddef>
#include <mutex>
#include <unordered_map>

// Process-wide CRT setup for the native-Windows build (runs before main via static init):
//  - binary file I/O (no CRLF translation) — the archive/io layer is binary, and text mode would
//    corrupt the superblock pread / zarr chunk reads (MSVC defaults to text mode).
//  - route asserts and abort() to stderr instead of a GUI dialog / Windows Error Reporting popup,
//    so a failing test or CHECK prints its message to the console rather than blocking on a dialog.
namespace {
struct FxCrtInit {
    FxCrtInit() {
        _set_fmode(_O_BINARY);
        _set_error_mode(_OUT_TO_STDERR);                            // asserts -> stderr, not a msgbox
        _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT); // abort() -> silent, no dialog/WER
        SetErrorMode(SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS); // no "stopped working" fault box
    }
} g_fx_crt_init;
}  // namespace

// Keep in sync with cmake/win-compat/include/sys/mman.h (same constants; not included here to
// avoid dragging the extern-"C" decls' off_t typedef vs windows.h ordering concerns).
#define PROT_READ  0x1
#define PROT_WRITE 0x2
#define MAP_NORESERVE 0x4000
#define MAP_FAILED (reinterpret_cast<void*>(-1))

extern "C" {

void* mmap(void* /*addr*/, size_t length, int prot, int flags, int fd, long long offset) {
    // 32 TiB MAP_NORESERVE reservations (codec/archive.hpp) have no faithful Win32 analogue.
    if (flags & MAP_NORESERVE) return MAP_FAILED;

    DWORD flProtect, access;
    if (prot & PROT_WRITE)     { flProtect = PAGE_READWRITE; access = FILE_MAP_WRITE; }
    else if (prot & PROT_READ) { flProtect = PAGE_READONLY;  access = FILE_MAP_READ;  }
    else return MAP_FAILED;

    HANDLE hFile = (fd >= 0) ? reinterpret_cast<HANDLE>(_get_osfhandle(fd)) : INVALID_HANDLE_VALUE;
    const unsigned long long end = static_cast<unsigned long long>(offset) + length;
    HANDLE hMap = CreateFileMappingA(hFile, nullptr, flProtect,
                                     static_cast<DWORD>(end >> 32),
                                     static_cast<DWORD>(end & 0xffffffffull), nullptr);
    if (!hMap) return MAP_FAILED;
    void* p = MapViewOfFile(hMap, access,
                            static_cast<DWORD>(static_cast<unsigned long long>(offset) >> 32),
                            static_cast<DWORD>(static_cast<unsigned long long>(offset) & 0xffffffffull),
                            length);
    CloseHandle(hMap);  // the view keeps the section alive
    return p ? p : MAP_FAILED;
}

// flock() advisory whole-file lock via LockFileEx (see cmake/win-compat/include/sys/file.h).
#define LOCK_SH 1
#define LOCK_EX 2
#define LOCK_NB 4
#define LOCK_UN 8
int flock(int fd, int operation) {
    HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (h == INVALID_HANDLE_VALUE) return -1;
    OVERLAPPED ov = {};
    if (operation & LOCK_UN) return UnlockFileEx(h, 0, MAXDWORD, MAXDWORD, &ov) ? 0 : -1;
    DWORD flags = 0;
    if (operation & LOCK_EX) flags |= LOCKFILE_EXCLUSIVE_LOCK;
    if (operation & LOCK_NB) flags |= LOCKFILE_FAIL_IMMEDIATELY;
    return LockFileEx(h, flags, 0, MAXDWORD, MAXDWORD, &ov) ? 0 : -1;
}

int munmap(void* addr, size_t /*length*/) { return UnmapViewOfFile(addr) ? 0 : -1; }
int madvise(void* /*addr*/, size_t /*length*/, int /*advice*/) { return 0; }
int msync(void* addr, size_t length, int /*flags*/) { return FlushViewOfFile(addr, length) ? 0 : -1; }
int mprotect(void* addr, size_t length, int prot) {
    DWORD newProt = (prot & PROT_WRITE) ? PAGE_READWRITE : PAGE_READONLY, old = 0;
    return VirtualProtect(addr, length, newProt, &old) ? 0 : -1;
}

}  // extern "C"

// ---- codec/archive.hpp backing store -------------------------------------------------------
// The Linux archive maps a huge MAP_NORESERVE region at a stable base over a file grown via
// fallocate. Faithful Win32 equivalent: mark the file sparse, size it to `reserve` (sparse — no
// real allocation), create ONE section of that size and map it whole. Writes fault in sparse
// pages on demand, exactly like MAP_NORESERVE; on close the file is compacted to the committed
// size so the persisted archive stays small.
namespace {
std::mutex g_fx_map_mu;
std::unordered_map<void*, HANDLE> g_fx_sections;  // base -> section handle (closed on unmap)
}  // namespace

extern "C" void* fenix_win_archive_map(int fd, unsigned long long reserve) {
    HANDLE hFile = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (hFile == INVALID_HANDLE_VALUE) return nullptr;
    // Best-effort sparse: sizing the file to `reserve` then costs no real disk. Ignore failure
    // (some filesystems lack sparse support — the mapping still works, just not thin).
    DWORD junk = 0;
    DeviceIoControl(hFile, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &junk, nullptr);

    ULARGE_INTEGER sz;
    sz.QuadPart = reserve;
    HANDLE hMap = CreateFileMappingW(hFile, nullptr, PAGE_READWRITE, sz.HighPart, sz.LowPart, nullptr);
    if (!hMap) return nullptr;
    void* base = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, 0);  // 0 bytes => whole section
    if (!base) { CloseHandle(hMap); return nullptr; }

    std::lock_guard<std::mutex> lk(g_fx_map_mu);
    g_fx_sections[base] = hMap;
    return base;
}

extern "C" int fenix_win_open(const char* path, int oflag, int pmode) {
    (void)pmode;
    DWORD access = (oflag & _O_RDWR) ? (GENERIC_READ | GENERIC_WRITE)
                 : (oflag & _O_WRONLY) ? GENERIC_WRITE
                                       : GENERIC_READ;
    const bool creat = (oflag & _O_CREAT) != 0, trunc = (oflag & _O_TRUNC) != 0;
    DWORD disp = creat ? (trunc ? CREATE_ALWAYS : OPEN_ALWAYS)
                       : (trunc ? TRUNCATE_EXISTING : OPEN_EXISTING);
    HANDLE h = CreateFileA(path, access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, disp, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return -1;
    int fd = _open_osfhandle(reinterpret_cast<intptr_t>(h), oflag);
    if (fd < 0) { CloseHandle(h); return -1; }
    return fd;
}

extern "C" int fenix_win_archive_unmap(void* base, int fd, long long compact_to) {
    HANDLE hMap = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_fx_map_mu);
        auto it = g_fx_sections.find(base);
        if (it != g_fx_sections.end()) { hMap = it->second; g_fx_sections.erase(it); }
    }
    if (base) UnmapViewOfFile(base);
    if (hMap) CloseHandle(hMap);
    // Compact the sparse backing file back to the committed size now that no section pins it.
    if (compact_to >= 0) {
        HANDLE hFile = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
        if (hFile != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER li;
            li.QuadPart = compact_to;
            if (SetFilePointerEx(hFile, li, nullptr, FILE_BEGIN)) SetEndOfFile(hFile);
        }
    }
    return 0;
}
