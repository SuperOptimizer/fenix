#pragma once
// POSIX <unistd.h> shim for NATIVE WINDOWS (clang + MSVC CRT). Self-contained: maps the POSIX
// file API onto the MSVC CRT (<io.h>/<direct.h>) — deliberately NO <windows.h> here, so this
// stays cheap to include across the whole header-only tree. Companion to the <sys/mman.h> shim.
//
// Only the names MSVC does NOT already provide are defined here. MSVC's <io.h> already supplies
// (deprecated-aliased) open/close/read/write/lseek/access/unlink AND the off_t typedef, so we
// leave those alone and just add the missing bits: ssize_t, the F_OK/… modes, and
// pread/pwrite/ftruncate/posix_fallocate/mkdir. Offsets are taken as `long long` (64-bit) even
// though MSVC's off_t is 32-bit, so these shims themselves don't cap file sizes.
#include <io.h>
#include <direct.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

#ifndef _SSIZE_T_DEFINED
typedef long long ssize_t;
#define _SSIZE_T_DEFINED
#endif

// access() modes — values chosen to match MSVC's _access(): F_OK=0, W_OK=2, R_OK=4.
#ifndef F_OK
#define F_OK 0
#define X_OK 1
#define W_OK 2
#define R_OK 4
#endif
#ifndef O_BINARY
#define O_BINARY _O_BINARY
#endif

// mkdir: MSVC only has single-arg _mkdir — provide the 2-arg POSIX form (mode ignored on NTFS).
static inline int mkdir(const char* path, int /*mode*/) { return _mkdir(path); }

// pread/pwrite: MSVC has no positional read/write. Seek-then-read (not atomic, but the archive
// header reads that use these are single-threaded). 64-bit offset via _lseeki64.
static inline ssize_t pread(int fd, void* buf, size_t n, long long off) {
    if (_lseeki64(fd, off, SEEK_SET) < 0) return -1;
    return _read(fd, buf, static_cast<unsigned>(n));
}
static inline ssize_t pwrite(int fd, const void* buf, size_t n, long long off) {
    if (_lseeki64(fd, off, SEEK_SET) < 0) return -1;
    return _write(fd, buf, static_cast<unsigned>(n));
}
// ftruncate/posix_fallocate -> _chsize_s (grows sparsely on NTFS). _chsize_s returns 0 or an
// errno, which is exactly posix_fallocate's contract; ftruncate maps that to 0/-1.
static inline int ftruncate(int fd, long long len) { return _chsize_s(fd, len) == 0 ? 0 : -1; }
static inline int posix_fallocate(int fd, long long off, long long len) {
    return _chsize_s(fd, off + len);
}

// Environment: POSIX setenv/unsetenv -> MSVC _putenv_s (MSVC provides neither POSIX name).
static inline int setenv(const char* name, const char* value, int /*overwrite*/) {
    return _putenv_s(name, value);
}
static inline int unsetenv(const char* name) { return _putenv_s(name, ""); }

// Subprocess pipes: POSIX popen/pclose -> MSVC _popen/_pclose.
static inline FILE* popen(const char* command, const char* mode) { return _popen(command, mode); }
static inline int   pclose(FILE* stream) { return _pclose(stream); }

// SIGPIPE has no Windows equivalent (a broken-pipe write returns an error instead of raising a
// signal). Define it so `std::signal(SIGPIPE, …)` in io/slice.hpp compiles; the call is a benign
// no-op at runtime (MSVC's signal() ignores unknown numbers). 13 is the conventional value.
#ifndef SIGPIPE
#define SIGPIPE 13
#endif
