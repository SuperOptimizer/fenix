#pragma once
// POSIX <sys/file.h> shim for NATIVE WINDOWS — flock() advisory whole-file locks, implemented
// via LockFileEx/UnlockFileEx in cmake/win-compat/win_compat.cpp. Standard BSD flock constants.
#define LOCK_SH 1   // shared lock
#define LOCK_EX 2   // exclusive lock
#define LOCK_NB 4   // non-blocking (OR into LOCK_SH/LOCK_EX)
#define LOCK_UN 8   // unlock

extern "C" int flock(int fd, int operation);
