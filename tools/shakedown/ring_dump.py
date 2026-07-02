import sys
sys.path.insert(0, "/workspace/fenix/tools/train")
import numpy as np
from feed_reader import FeedRing, READY
import struct
r = FeedRing(sys.argv[1])
mode = sys.argv[2] if len(sys.argv) > 2 else "meta"
recs = []
for s in range(r.nslots):
    if r._states[s][0] != READY: continue
    base = 4096 + s * r.slot_bytes
    mesh, draw = struct.unpack_from("<IQ", r.mm, base + 4)
    oz, oy, ox = struct.unpack_from("<qqq", r.mm, base + 16)
    if mode == "meta":
        recs.append((draw, mesh, oz, oy, ox))
    else:  # content stats
        P, C = r.patch, r.channels
        t = P * P * P
        d = np.frombuffer(r.mm, dtype=np.uint8, count=t * C, offset=base + 64).reshape(C, P, P, P)
        ct, gt = d[0], d[1]
        recs.append((draw, float(ct.std()), int((gt == 255).sum()), int((gt == 128).sum()),
                     int((gt == 0).sum())))
for x in sorted(recs): print(x)
