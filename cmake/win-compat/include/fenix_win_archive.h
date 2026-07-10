#pragma once
// Win32 backing store for codec/archive.hpp's out-of-core mapping (implemented in
// cmake/win-compat/win_compat.cpp, which isolates <windows.h>).
//
//   fenix_win_archive_map(fd, reserve)
//       Mark fd's file sparse, size it to `reserve` (sparse — no real allocation), create one
//       file-mapping section of that size and map it whole. Returns a stable base pointer where
//       writes fault in sparse pages on demand — the faithful equivalent of the Linux
//       mmap(MAP_SHARED|MAP_NORESERVE) reservation. Returns nullptr on failure.
//
//   fenix_win_archive_unmap(base, fd, compact_to)
//       Unmap the view and close the section. If compact_to >= 0, shrink the (sparse) file back
//       to that many bytes (SetEndOfFile) now that nothing pins it — so the persisted archive is
//       compact, not the full `reserve`. Pass compact_to < 0 for read-only opens (leave the file
//       untouched).
extern "C" {
void* fenix_win_archive_map(int fd, unsigned long long reserve);
int   fenix_win_archive_unmap(void* base, int fd, long long compact_to);

// open() equivalent that grants FILE_SHARE_DELETE, so the archive file can be unlinked while
// still open/mapped — matching POSIX semantics (the CRT's open() denies delete-sharing). oflag
// uses the usual _O_RDWR/_O_CREAT/_O_TRUNC bits; pmode is ignored (NTFS perms differ). -1 on error.
int   fenix_win_open(const char* path, int oflag, int pmode);
}
