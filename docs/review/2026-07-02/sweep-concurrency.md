# sweep-concurrency — cross-cutting concurrency review of src/

Scope: every std::thread, OpenMP region (via `core/parallel.hpp`), atomic, mutex, and condition
variable in `src/`, with a focus on cross-module interactions (IO thread fan-out ↔ libcurl,
producer/consumer pipelines ↔ archive/codec locks, ML ring ↔ libtorch, FUSE ↔ block cache).

**Overall assessment:** the concurrency architecture is genuinely disciplined. The single-writer
archive rule is respected everywhere (export-scroll keeps all `write_chunk`/`commit` on the consumer
thread), red-black Gauss-Seidel sweeps (`winding/relax.hpp`, `winding/patch_field.hpp`,
`segment/grow.hpp` ARAP) have correct parity separation, the slab-local union-find in
`geom/connected_components.hpp` provably never races on `parent`, the anti-nesting
`SerialRegion`/`g_parallel_serial` design is sound, error propagation out of parallel bodies uses the
CAS-winner-writes pattern correctly, and the export-scroll bounded queue's shutdown paths (producer
fetch failure, consumer write failure, normal drain) all terminate without deadlock. The real
problems are at module seams: a per-thread libcurl handle design that assumes long-lived threads
combined with a `parallel_for_io` that spawns fresh threads per call (unbounded resource leak over a
whole-scroll export), an unsynchronized lazy-init of the archive's block cache on an API documented
as thread-safe, and a producer thread in the ML pipelined path with no shutdown path when libtorch
throws.

## [high/resource-safety] Per-call thread spawning in parallel_for_io leaks one CURL easy handle (with its 256 KiB buffer) per spawned thread — unbounded over a whole-scroll export

**Verdict:** CONFIRMED — Every load-bearing fact checks out. (1) src/io/s3.hpp:121-144 — thread_handle() stores a raw `thread_local CURL* h` initialized by curl_easy_init(); a raw pointer has no destructor, so nothing runs at thread exit, and `grep -rn curl_easy_cleanup src tools apps` returns zero hits — the easy handle is never freed anywhere in the tree. (2) src/core/parallel.hpp:192-196 — parallel_for_io constructs fresh std::threads per call (`pool.emplace_back(worker)`) and joins them; there is no persistent pool (core/CLAUDE.md lists "first-party async thread pool" under TODO). (3) src/io/zarr.hpp:165 — read_zarr_region fetches remote chunks through parallel_for_io with fetch_threads=24 (clamped to chunk count at parallel.hpp:183, so ~7 spawned threads for an 8-chunk region); each spawned thread's first http_get (s3.hpp:154) creates a handle that leaks at join. (4) src/io/io.hpp:277-314 — export-scroll producers call read_zarr_region once per region in a loop (plus a retry at line 293), so leaked handles grow linearly and unboundedly with region count over a whole-scroll export. The handle carries CURLOPT_BUFFERSIZE=262144 (s3.hpp:132) plus internal state. No guard, no documented-stub disclaimer (io/CLAUDE.md calls s3.hpp implemented), and the failure path is the module's flagship out-of-core resumable export. Two claim details are overstated but don't change the verdict: (a) recent libcurl releases the 256 KiB download buffer at transfer end, so the steady-state per-handle leak may be tens of KiB rather than 256 KiB+ depending on the curl version — still unbounded and multi-GB at whole-scroll region counts; (b) "defeats the warm-connection purpose" is largely mitigated by the CURLSH share (s3.hpp:77-79 shares DNS + TLS sessions + the connection cache across all handles), so warm connections survive worker-thread death even though the handles leak.

**Fix notes:** The RAII fix is correct and safe here: wrap the handle in a thread_local struct whose destructor calls curl_easy_cleanup(h). libcurl requires the CURLSH to outlive attached easy handles; that holds because SharePool (s3.hpp:91) is a function-local static that never calls curl_share_cleanup ("leaked at exit intentionally"), so thread_local destruction order is not a hazard. Note the destructor-based fix alone restores nothing performance-wise for spawned workers (they die anyway) — connection reuse across calls is already provided by the CURL_LOCK_DATA_CONNECT share, so the persistent-IO-pool half of the proposal is a perf/cleanliness improvement, not required for correctness of the leak fix. Also add curl_global_cleanup consideration is NOT needed (process-lifetime is fine); only the per-thread easy handle must be freed.

**Location:** src/io/s3.hpp:121-144 (`thread_handle`) × src/core/parallel.hpp:176-197 (`parallel_for_io`)

**Evidence:**
```cpp
// s3.hpp — Thread-local reused handle (connection + TLS reuse). Lives for the thread's lifetime.
inline CURL* thread_handle() {
    thread_local CURL* h = [] {
        CURL* c = curl_easy_init();
        ...
        curl_easy_setopt(c, CURLOPT_BUFFERSIZE, 262144L);  // 256 KB recv buffer
```
```cpp
// parallel.hpp — parallel_for_io
    std::vector<std::thread> pool;
    pool.reserve(static_cast<usize>(nt - 1));
    for (int t = 0; t < nt - 1; ++t) pool.emplace_back(worker);
```
`thread_handle()` is a `thread_local` **raw pointer** — there is no destructor, and
`curl_easy_cleanup` appears nowhere in the tree. The design comment assumes the thread lives for the
process ("Lives for the thread's lifetime... keeps the connection pool + TLS session warm across
requests on a thread"). But `io::read_zarr_region` fetches remote chunks through
`parallel_for_io(0, nchunks, fetch_threads, ...)` (zarr.hpp:165), which constructs **fresh
std::threads on every call** and joins them at the end. Each spawned worker calls `http_get` →
`thread_handle()` → `curl_easy_init()` and then exits, leaking the easy handle (and its 256 KiB
receive buffer, TLS state, etc.).

**Failure scenario:** `fenix export-scroll` on a real scroll: each of its 8 producer threads calls
`read_zarr_region` once per region; for a 256³ region over 128³ zarr chunks `parallel_for_io` spawns
up to 7 fresh threads → ~7 leaked easy handles (~2 MB counting the recv buffers) per region. A
multi-hundred-thousand-region export (PHerc-scale volumes) leaks tens-to-hundreds of GB of heap and
eventually OOMs the multi-hour unattended job it was specifically designed to survive
(io.hpp:284-294 goes to great lengths to ride out network blips — then dies of its own leak). It also
defeats the stated purpose of the thread-local handle: no connection/TLS warmth survives across
regions on the spawned workers (only the CURLSH share mitigates this partially).

**Suggested fix:** either (a) make the thread-local an RAII wrapper (`struct H { CURL* c;
~H(){ if(c) curl_easy_cleanup(c); } }; thread_local H h{...};`) so thread exit reclaims the handle,
or (b) give `parallel_for_io` a persistent first-party IO thread pool (already on core's TODO list)
so the "handle per long-lived thread" assumption becomes true. (a) is the two-line correctness fix;
(b) additionally restores connection reuse.

**Outcome:** fixed — implemented option (a): `thread_handle()` in src/io/s3.hpp now wraps
the handle in a `ThreadHandle` RAII struct (`~ThreadHandle() { if (h) curl_easy_cleanup(h); }`,
copy-disabled), stored as `thread_local ThreadHandle t;`. The CURLSH share pool
(`shared_pool()`) remains a process-lifetime leaked static as before, so cleanup order is
not a hazard per the fix notes. Did not implement (b) (persistent IO pool) — the RAII fix
alone is the correctness-required minimal change; a persistent pool is a separate perf
improvement, not attempted here. Verified via full build + test_zarr/test_trace_stream
(both exercise `parallel_for_io` fan-out through `read_zarr_region`, 5/5 and 7/7 pass).

## [high/concurrency] VolumeArchive::block16 lazily initializes the block cache without synchronization — data race / use-after-free on an API documented as thread-safe

**Verdict:** CONFIRMED — The code is exactly as described: src/codec/archive.hpp:402 does an unsynchronized test-and-assign on the mutable unique_ptr block_cache_ (declared line 806) inside const block16(); there is no mutex/once_flag in VolumeArchive. The thread-safety contract is genuinely advertised — archive.hpp:466 documents gather_box_f32 (which calls block16 at line 486) as "Thread-safe for disjoint boxes", and src/fs/CLAUDE.md:35-36 states "FUSE is multithreaded (block16 is const + the cache is sharded/locked …)". BlockCache (src/codec/block_cache.hpp) locks per-shard for get/put but cannot protect the pointer that holds it. The only current callers avoid the race by convention, not enforcement: fxfs.cpp:135 calls reserve_cache in single-threaded main before fuse_main, and test_fxvol.cpp:256 is single-threaded — precisely the "safe today only by happenstance" situation the reviewer described. Any conforming caller exercising the documented concurrent contract without a prior reserve_cache hits a data race (racing non-atomic pointer store/load) escalating to use-after-free when the losing thread's assignment destroys the BlockCache the winner is using. Reachable per the API's own contract → not refutable.

**Fix notes:** Eager construction in open()/create() (dropping the lazy init from block16) is the right fix and the simplest — reserve_cache() then becomes a pre-threading resize, which should be documented as "call before concurrent readers" since it too reassigns the unique_ptr unsynchronized and would destroy an in-use cache (same UAF) if called mid-flight; consider a FENIX_ASSERT or a comment there. If lazy init is kept instead, std::once_flag works, but note it only fixes first-touch — the reserve_cache-after-threads hazard remains either way. std::call_once with a non-throwing callable is fine under -fno-exceptions. cache_hits()/cache_misses()/cache_bytes() (lines 391-393) read the same pointer and become benign once init is eager. The move-assignment (line 773) touching block_cache_ is fine (moves are inherently exclusive).

**Location:** src/codec/archive.hpp:402 (and member decl at :806)

**Evidence:**
```cpp
[[nodiscard]] Expected<BlockCache::Ref> block16(s64 lod, ChunkCoord bc) const {
    if (!block_cache_) block_cache_ = std::make_unique<BlockCache>(detail::kFxDefaultCache);
```
with `mutable std::unique_ptr<BlockCache> block_cache_;`. `block16`, `sample_f32`, and
`gather_box_f32` are all `const` and explicitly advertised as concurrent-safe
(`gather_box_f32`: "Thread-safe for disjoint boxes"; src/fs/CLAUDE.md: "FUSE is multithreaded
(`block16` is const + the cache is sharded/locked ...)"). The sharded BlockCache itself is
correctly locked — but the lazy check-then-assign of the `unique_ptr` is not. Two threads that both
observe `!block_cache_` each construct a cache; the second `operator=` **destroys** the BlockCache
the first thread may currently be executing `get()`/`put()` on → use-after-free. Even without
destruction, the pointer store/load is a plain data race (UB).

**Failure scenario:** any consumer that starts parallel reads without first calling
`reserve_cache()` — e.g. a streaming inference filler calling `gather_box_f32` from
`parallel_for`, or a future fxfs change that drops the `reserve_cache` call in `main()` (nothing in
the type enforces it) — two first-touch threads race, one thread's shard-mutex `lock()` executes on
a freed BlockCache → crash or silent corruption. Today fxfs happens to be safe only because
`main()` calls `reserve_cache` before mounting; the archive API itself is the trap.

**Suggested fix:** construct the default cache eagerly in `open()`/`create()` (it's lazy only to
save one allocation), or guard the init with a `std::once_flag` member. Remove the mutation from the
const read path entirely.

**Outcome:** fixed — eager construction in both `create()` and `open()` (src/codec/archive.hpp), as this
note and the codec.md duplicate both recommend; `block16()` no longer mutates `block_cache_` on its const
path at all (a null cache there now returns an `Errc::internal` error rather than lazily constructing).
`reserve_cache()` gained a doc comment calling out that it still unconditionally replaces the pointer and
must only be called before concurrent `block16()`/`gather_box_f32()` use, matching this note's remark
about the "reserve_cache-after-threads hazard" remaining otherwise. See the fuller outcome note under the
duplicate entry in `docs/review/2026-07-02/codec.md`. Tested via
`tests/test_archive.cpp::archive_block16_concurrent_use_is_safe` (8 threads × 200 `block16()` calls with
no prior `reserve_cache()`) and the pre-existing `tests/test_fxvol.cpp::fxvol_block_cache`, both green
under ASan. TSan was not run for this pass.

## [medium/concurrency] ML pipelined inference: producer thread has no shutdown path — a libtorch exception in the consumer loop calls std::terminate, and no stop flag exists to ever join the producer early

**Verdict:** unverified (medium/low)

**Location:** src/ml/infer.hpp:330-381 (producer thread + ring waits)

**Evidence:**
```cpp
std::thread producer([&] {
    for (long i0 = resume_from; i0 < total; i0 += B) {
        ...
        cv_free.wait(lk, [&] { return filled < kSlots; });   // no stop/cancel in the predicate
        ...
    }
});
for (;;) {
    ...
    auto logits = net->forward(xin);       // libtorch: can throw (CUDA OOM etc.)
    ...
}
producer.join();
```
The ML TU is compiled **with** exceptions (libtorch ABI, per ml/CLAUDE.md), and
`net->forward`/`.to(dev)` throw `c10::Error` on CUDA OOM — an observed real condition here (the
batch-size comment documents b=4 as a "VRAM-pressure regression"; b=3 ≈ 28 GB on a 32 GB card, i.e.
the margin is ~4 GB). If the consumer loop throws: stack unwinding destroys the joinable
`std::thread producer` → immediate `std::terminate` (no Expected error, no final checkpoint flush).
Worse, the code is unfixable by a caller-side try/catch: there is **no stop flag**, and
`cv_free.wait(lk, filled < kSlots)` can block forever once the consumer stops consuming — a
catch-then-join would deadlock. There is also no way for the consumer to abort early for any other
reason (the `prod_done`/`filled` protocol only handles normal completion).

**Failure scenario:** predict-surface on a card near its VRAM limit; a fragmentation-induced OOM on
tile ~2000 of 36000 → `std::terminate` mid-run instead of `Expected` error; the run's `.ckpt` is
whatever the last cadence flush was, and any wrapper that wanted to catch/downgrade can't.

**Suggested fix:** add `bool stop = false;` under the same mutex; include `|| stop` in the
`cv_free.wait` predicate and check it at the top of the producer loop; wrap the consumer body in
try/catch (ML TU has exceptions) or a scope guard that sets `stop`, notifies `cv_free`, and joins
before rethrowing/returning an `Errc` error.

## [medium/performance] Pipelined inference intentionally overlaps two full-width OpenMP teams: producer prep() and consumer scatter() each run cpu_budget()-wide parallel_for concurrently — 2× CPU oversubscription

**Verdict:** unverified (medium/low)

**Location:** src/ml/infer.hpp:263 (prep's parallel_for), :298 (scatter's parallel_for), :330 (producer thread)

**Evidence:** the producer thread runs `prep()` (which calls `parallel_for(0, P, ...)` for the fill
and the z-score reduction — and the fillers in inference.cpp:298 and infer.hpp:445 are themselves
`parallel_for` over P) at the same time as the main thread runs `scatter()`'s
`parallel_for(0, P, ...)`. `parallel_for` sizes every team to `cpu_budget()` regardless of the
caller thread, and libomp keeps a **separate persistent team per external calling thread** — so the
process holds 2×`cpu_budget()` OMP workers, both actively compute-bound during the
prep-overlaps-scatter window. This is exactly the oversubscription pattern the project measured as
harmful elsewhere ("nesting oversubscribes the cores (measured ~10× slowdown)", parallel.hpp:96-99);
here it is sibling-region rather than nested, so milder, but the scatter and prep both being
memory-bandwidth-bound makes their concurrent full-width teams contend rather than compose. The
comment claims prep is overlapped with the *GPU forward*, but it equally overlaps the CPU scatter.

**Failure scenario:** on a CPU-quota'd container (the cgroup case parallel.hpp exists for), a
16-CPU budget runs 32 compute threads during the overlap window; barrier churn + bandwidth
contention erodes the ~16% prep-overlap win the pipeline was built to capture, and the profile
(`t_scat`) silently inflates.

**Suggested fix:** cap the two sides to complementary budgets (e.g. pass
`max_threads = cpu_budget()/2` to prep's and scatter's `parallel_for` when B>1), or run prep under
a half-width cap only while a batch is in flight.

## [medium/correctness] stream_tile_u8 silently converts a hard fetch failure into an all-zero tile — the streamed tracer's production entry points trace transient network failures as air

**Verdict:** unverified (medium/low)

**Location:** src/segment/trace_stream.hpp:27-38 (used by trace_volume_streamed:60 and trace_volume_streamed_to_disk:105)

**Evidence:**
```cpp
inline Volume<u8> stream_tile_u8(const std::string& root, Index3 porg, Extent3 pe, f32 scale) {
    Volume<u8> out(pe);
    auto r = io::read_zarr_region(root, porg, pe);
    if (!r) return out;   // hard fetch error -> zero-filled tile
```
`read_zarr_region` carefully distinguishes 404-absent from fetch-failed (`Errc::fetch_failed` after
retries, s3.hpp/zarr.hpp) precisely so this can't happen — and this call site erases the
distinction. The comment acknowledges it ("production callers ... should thread the Expected
through"), but both shipping entry points — `trace_volume_streamed` and the "FULLY out-of-core"
`trace_volume_streamed_to_disk` (which already returns `Expected`) — call it as-is. This is a direct
violation of the root hard rule: "A transient fetch error must **never** silently become air."

**Failure scenario:** a multi-hour streamed whole-volume trace over S3 hits one sustained blip after
`http_get`'s retries are exhausted; that tile's prediction block reads as all-zero, the tracer
finds no seeds/sheet there, and the output manifest is simply missing every fragment of that tile —
indistinguishable from real air, unflagged, and persisted to disk as authoritative.

**Suggested fix:** change `stream_tile_u8` to return `Expected<Volume<u8>>` and propagate through
both entry points (`trace_volume_streamed_to_disk` already has the right return type;
`trace_volume_streamed` needs one). Retry policy can stay in io.

## [low/performance] fxfs fx_read takes a sharded-cache lookup (mutex lock) per BYTE — the "current block cached across the loop" claimed in its comment is not implemented

**Verdict:** unverified (medium/low)

**Location:** src/fs/fxfs.cpp:96-113

**Evidence:**
```cpp
// The "current block" is cached across the loop so a contiguous x-run (16 voxels) costs one
// block16 call ...
for (u64 i = 0; i < n; ++i) {
    ...
    auto v = g->ar.sample_f32(0, z, y, x);   // block16() -> BlockCache::get() -> shard mutex, EVERY byte
```
`sample_f32` calls `block16` → `BlockCache::get` (shard `std::mutex` lock + hash lookup) for every
voxel; nothing holds the previous `Ref` across the x-run. Under FUSE's default multithreaded loop,
several reader threads hammer the 16 shard mutexes at byte granularity — 4096 lock/unlock+hash
cycles per 4 KiB page fault instead of ~16 block fetches + memcpys, so a parallel consumer (e.g.
mmap-driven inference over the mount) serializes on cache locks. This is exactly the lock-bound
regime `gather_box_f32`'s block-major redesign (archive.hpp:462-466) was built to escape.

**Failure scenario:** multi-threaded sequential read of `volume.raw` scales far below the decode
throughput; profile shows the time in `BlockCache::get`'s mutex, not in the DCT.

**Suggested fix:** hoist a `BlockCache::Ref cur` + its block coord out of the loop and refetch only
on block change (the comment's design), or better, route whole x-runs through
`gather_box_f32`/a u8 gather variant.

---

**Verified clean (checked, no finding):** export-scroll producer/consumer shutdown protocol (all
three exit paths terminate; `cancel` is in every wait predicate that needs it); archive single-writer
discipline (encode-parallel/commit-serial two-phase in `write_level_`; consumer-only writes in
export-scroll); `read_volume_as`/zarr error propagation (CAS-winner writes the message, read after
the join barrier); red-black parity in relax/patch_field/grow ARAP (all 6-neighbourhoods flip
parity); slab-local union-find in connected_components (unions provably confined to slab index
ranges; serial boundary merge; read-only find in phase 3); BlockCache SIEVE under its shard locks
(shared_ptr pinning makes eviction safe against in-flight readers); CURLSH SharePool lock callbacks
(leaf mutexes, no ordering hazard against any fenix lock); `curl_global_once` via `call_once`;
per-tile `SerialRegion` anti-nesting in the tiled tracer; entropy.hpp relaxed stat atomics.
