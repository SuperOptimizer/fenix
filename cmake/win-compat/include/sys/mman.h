#pragma once
// POSIX <sys/mman.h> shim for NATIVE WINDOWS (clang + MSVC). Declares mmap/munmap/... (defined
// in cmake/win-compat/win_compat.cpp, which isolates <windows.h>). Also pulls the <unistd.h>
// shim so codec/io headers that assume the POSIX file API "just work".
//
// IMPORTANT LIMITATION: codec/archive.hpp reserves a 32 TiB MAP_NORESERVE mapping over a sparse,
// ftruncate-grown file at a stable base address. Windows has no faithful equivalent (a file
// mapping can't exceed the file's size, and MapViewOfFile picks the address). This shim therefore
// returns MAP_FAILED for MAP_NORESERVE, so archive creation fails at runtime with io_error. The
// binary builds and `fenix help` (and any non-archive path) runs; full out-of-core scroll I/O on
// Windows needs a real section-object/VirtualAlloc rewrite (tracked as a follow-up).
#include <cstddef>
#include <unistd.h>   // off_t + POSIX file API

#define PROT_NONE  0x0
#define PROT_READ  0x1
#define PROT_WRITE 0x2
#define PROT_EXEC  0x4

#define MAP_FILE      0x0000
#define MAP_SHARED    0x0001
#define MAP_PRIVATE   0x0002
#define MAP_FIXED     0x0010
#define MAP_ANONYMOUS 0x0020
#define MAP_ANON      MAP_ANONYMOUS
#define MAP_NORESERVE 0x4000
#define MAP_FAILED    (reinterpret_cast<void*>(-1))

#define MADV_NORMAL     0
#define MADV_RANDOM     1
#define MADV_SEQUENTIAL 2
#define MADV_WILLNEED   3
#define MADV_DONTNEED   4

#define MS_ASYNC      0x1
#define MS_SYNC       0x4
#define MS_INVALIDATE 0x2

extern "C" {
void* mmap(void* addr, size_t length, int prot, int flags, int fd, long long offset);
int   munmap(void* addr, size_t length);
int   madvise(void* addr, size_t length, int advice);
int   msync(void* addr, size_t length, int flags);
int   mprotect(void* addr, size_t length, int prot);
}
