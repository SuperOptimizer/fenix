"""feed_reader.py — zero-copy consumer for the `fenix train-feed` shm ring.

Ring protocol (ml/feed.hpp): 4096-B header, then nslots fixed slots.
  header: magic 'FXRING1\\0' | u32 version | u32 nslots | u64 patch | u64 slot_bytes
          | u32 channels | u32 reserved
  slot:   u32 state (0 FREE / 1 READY / 2 WRITING) | u32 mesh | u64 draw | s64 origin[3]
          | pad to 64 B | channels x u8[patch^3]
Consumer contract: scan for READY, read, store FREE. The producer's release-store on
state=READY orders the data before the flag on x86/arm64; our plain u32 store back is
atomic at this width.
"""
import mmap
import os
import struct
import time

import numpy as np

_HDR = struct.Struct("<8sIIQQII")
_SLOT_HDR_BYTES = 64
FREE, READY, WRITING = 0, 1, 2


class FeedRing:
    # stripe_rank/stripe_world: multi-consumer partition for DDP — rank r consumes only
    # slots with index % world == r (one feeder, one ring, no lock contention; each rank
    # walks its own deterministic stripe). Default = single consumer, whole ring.
    def __init__(self, path: str, stripe_rank: int = 0, stripe_world: int = 1):
        self.stripe_rank, self.stripe_world = stripe_rank, stripe_world
        self.fd = os.open(path, os.O_RDWR)
        size = os.fstat(self.fd).st_size
        self.mm = mmap.mmap(self.fd, size)
        magic, ver, nslots, patch, slot_bytes, channels, _ = _HDR.unpack_from(self.mm, 0)
        assert magic == b"FXRING1\x00" and ver == 1, f"bad ring header: {magic} v{ver}"
        self.nslots, self.patch, self.slot_bytes, self.channels = nslots, patch, slot_bytes, channels
        self._states = [
            np.frombuffer(self.mm, dtype=np.uint32, count=1, offset=4096 + s * slot_bytes)
            for s in range(nslots)
        ]
        self._cursor = 0

    def ready_count(self) -> int:
        return sum(1 for st in self._states if st[0] == READY)

    def next_batch(self, n: int, timeout_s: float = 60.0):
        """Collect n READY slots -> dict of stacked arrays (COPIES — slots are freed after)."""
        P, C = self.patch, self.channels
        tensor = P * P * P
        ct, gt, te, meta = [], [], [], []
        deadline = time.monotonic() + timeout_s
        while len(ct) < n:
            s = self._cursor
            self._cursor = (self._cursor + 1) % self.nslots
            if self.stripe_world > 1 and s % self.stripe_world != self.stripe_rank:
                continue
            if self._states[s][0] != READY:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"feed ring starved ({len(ct)}/{n} after {timeout_s}s)")
                if s == 0:
                    time.sleep(0.001)
                continue
            base = 4096 + s * self.slot_bytes
            mesh, draw = struct.unpack_from("<IQ", self.mm, base + 4)
            oz, oy, ox = struct.unpack_from("<qqq", self.mm, base + 16)
            data = np.frombuffer(self.mm, dtype=np.uint8, count=tensor * C,
                                 offset=base + _SLOT_HDR_BYTES)
            cube = data.reshape(C, P, P, P).copy()  # copy out, then release the slot
            self._states[s][0] = FREE
            ct.append(cube[0])
            gt.append(cube[1])
            if C == 3:
                te.append(cube[2])
            meta.append((mesh, draw, oz, oy, ox))
        out = {"ct": np.stack(ct), "gt": np.stack(gt), "meta": meta}
        if te:
            out["teacher"] = np.stack(te)
        return out

    def close(self):
        self._states = []  # drop the frombuffer views: an exported pointer makes mm.close() throw
        self.mm.close()
        os.close(self.fd)
