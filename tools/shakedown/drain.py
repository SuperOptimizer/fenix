import sys, time
sys.path.insert(0, "/home/forrest/fenix/tools/train")
from feed_reader import FeedRing, READY, FREE
import os
while not os.path.exists(sys.argv[1]):
    time.sleep(0.2)
time.sleep(0.5)
r = FeedRing(sys.argv[1])
n = 0
t0 = time.time()
deadline = t0 + float(sys.argv[2])
while time.time() < deadline:
    got = False
    for s in range(r.nslots):
        if r._states[s][0] == READY:
            r._states[s][0] = FREE
            n += 1
            got = True
    if not got:
        time.sleep(0.0005)
print(f"drained {n} slots in {time.time()-t0:.1f}s = {n/(time.time()-t0):.1f} draws/s")
