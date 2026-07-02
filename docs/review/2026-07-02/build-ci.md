# build-ci — CMakeLists.txt, CMakePresets.json, cmake/, Dockerfile, bootstrap.sh, install-runpod-ubuntu2404.sh, .github/workflows/, .clang-tidy, .clang-format

Overall assessment: the build system itself (CMakeLists, toolchain file, deps.cmake, presets) is thoughtfully engineered — the dep-resolution macro, ccache/PCH coexistence logic, and split/unity duality are well designed. The problem is the **verification layer around it is dead or hollow**: CI has literally never passed (checked live: the last 40 runs on GitHub are all `failure`, every job dying at `actions/checkout@v4` because a glibc node binary cannot exec inside the musl Chimera container), so every gate CLAUDE.md §5.3 lists as merge-blocking — build, sanitizers, tests, format, tidy — currently gates nothing, and commits land directly on `main` under a permanently red badge. Beneath that headline: the OpenMP runtime is missing from both the Docker image and CI package lists (canonical builds are silently serial), the fuzz preset produces an unlinkable build, the split build is never CI-verified, and the IWYU gate is `|| true`. Findings below, most severe first.

## [critical/bug] CI is 100% red: actions/checkout@v4 cannot run inside the musl Chimera container — every gate is dead

**Verdict:** CONFIRMED — Confirmed, not refuted. .github/workflows/ci.yml:17,41,60 — all three jobs use `container: chimeralinux/chimera:latest` with `actions/checkout@v4` (lines 25, 45, 63). Live verification via gh: the last 40 runs are ALL conclusion=failure. Latest run (28600411649): every job (build+test release/debug/asan/ubsan/tsan, format-tidy, coverage) fails at the "Run actions/checkout@v4" step. The failed-step log shows the exact claimed mechanism: the runner probes `cat /etc/*release | grep ^ID` (its alpine/musl detection) then dies with `exec /__e/node24/bin/node: no such file or directory` — the injected node24 is glibc-linked and Chimera (ID=chimera, musl) is not matched as alpine. The preceding `apk add` run-step succeeds (run steps use the container shell, not node), so only JS actions are broken — matching the claim precisely. No project gate (build, sanitizers, tests, format/tidy, coverage) has ever executed, and main receives direct commits under permanently red CI, violating CLAUDE.md §5.3. Nothing in the repo documents this as intentional; ci.yml's own comments describe these jobs as the merge gates.

**Fix notes:** Both proposed fixes are viable. Corrections/additions for the git-clone variant: (1) after clone, run `git config --global --add safe.directory "$GITHUB_WORKSPACE"` (dubious-ownership check inside containers); (2) `git checkout $GITHUB_SHA` needs the SHA fetched — with `--depth=1` clone of the default branch, a PR merge SHA won't exist; use `git init && git remote add origin ... && git fetch --depth=1 origin $GITHUB_SHA && git checkout FETCH_HEAD` instead; (3) the apk step must keep installing git BEFORE the clone step (it already does). Also note the "Post Run actions/checkout" cleanup failure disappears automatically once the JS action is removed. A third, lower-effort option: set `ACTIONS_RUNNER_FORCE_ACTIONS_NODE_VERSION` won't help (still glibc); but installing `nodejs` via apk and symlinking into /__e is fragile — prefer the two proposed approaches. The prebuilt-GHCR-image TODO at ci.yml:10 pairs naturally with the `docker run` variant.

**Location:** .github/workflows/ci.yml:17 (also :41, :61)

**Evidence:**
```yaml
    runs-on: ubuntu-latest
    container: chimeralinux/chimera:latest
    ...
      - uses: actions/checkout@v4
```
Live verification (`gh run list --limit 40`): **40/40 runs = failure**, all within ~50 s. Failed-step log:
```
##[command]/usr/bin/docker exec ... sh -c "cat /etc/*release | grep ^ID"
exec /__e/node24/bin/node: no such file or directory
```
The Actions runner injects its own glibc-linked node into container jobs; it only substitutes the musl (alpine) node build when `/etc/os-release` reports `ID=alpine`. Chimera reports `ID=chimera`, so the runner mounts glibc node into a musl container and every JS action (checkout, and its Post step) hard-fails before a single project step runs.

**Failure scenario:** every push and PR since the workflow was added fails at checkout. Nothing in §5.3's gate list (clang-format, clang-tidy, ASan/UBSan/TSan builds, tests, coverage) has ever executed on GitHub. Any regression — a broken build, a failing test, a sanitizer hit — merges to `main` unnoticed; the "protected main, CI green to merge" policy is inoperative (and history shows direct pushes to `main` with red CI).

**Suggested fix:** stop using JS actions inside the musl container. Simplest: drop `container:` from the job and run each build step via `docker run` against a prebuilt GHCR Chimera image (the TODO already in the file), with checkout on the ubuntu host. Alternative: keep `container:` but replace `actions/checkout@v4` with a plain `run: git clone --depth=1 "https://x-access-token:${{ github.token }}@github.com/${{ github.repository }}" . && git checkout $GITHUB_SHA` step (git is already installed in the apk step, which does run). Then actually watch one run go green, and turn on branch protection.

**Outcome:** fixed — dropped `container: chimeralinux/chimera:latest` from all three jobs; `actions/checkout@v4` now runs on the `ubuntu-latest` host, and every project step (apk install, deps, configure, build, test) runs via `docker run --rm -v "${{ github.workspace }}:/work" -w /work chimeralinux/chimera:latest sh -c '...'` with the workspace bind-mounted. This is the "drop `container:`, run via `docker run`" variant from the fix notes. Could NOT be verified by actually running on GitHub Actions (no push access exercised, and the task rules bar git-state-modifying commands); Docker itself is not installed on this local machine (`docker info` → command not found) so the `docker run` invocation could not be smoke-tested locally either. Desk-checked rigorously instead: the bind-mount + `-w /work` + `sh -c` structure is the standard pattern for running a container-image build step from a workflow on the host runner, and it sidesteps the actions/checkout-inside-musl-container failure mode entirely (checkout is now a host-side JS action, never executed inside Chimera). The `docker run` step's `set -eu` means any apk/build/test failure inside the container correctly fails the step. Flagging as **desk-check only, not live-verified** per the task's own caveat — this is the highest-risk fix in the set to regress silently; recommend the first live CI run be watched closely.

## [high/bug] OpenMP runtime is not installed in the Chimera Docker image or CI — canonical builds are silently serial

**Verdict:** CONFIRMED — Confirmed at every link of the chain. (1) Dockerfile:15-29 installs clang, lld, llvm, libcxx-devel, libunwind-devel, clang-tools-extra, cmake, ninja, git, pkgconf, ccache, curl-devel, zlib-ng-devel — no libomp. (2) The Chimera package index (queried live at pkgs.chimera-linux.org) shows libomp/libomp-devel are separate main-repo packages, and the installed packages' dependency lists do NOT pull them in: `clang` depends on clang-cpp-libs, clang-libs, clang-rt-devel, fortify-headers, libatomic-chimera-devel, libcxx(+devel), libcxxabi, llvm, llvm-binutils, llvm-libs, musl-devel; `llvm` and `clang-rt-devel` likewise omit libomp. Without omp.h/libomp.so, CMake's FindOpenMP link test fails. (3) CMakeLists.txt:123-125 is a non-REQUIRED `find_package(OpenMP)` guarded by a silent `if(OpenMP_CXX_FOUND)` — no message on the not-found path. (4) .github/workflows/ci.yml installs the same libomp-less apk list in all three jobs (build-test line ~24, format-tidy, coverage), and the build-test matrix includes tsan — so the tsan gate compiles `#pragma omp` regions as plain serial loops (src/core/parallel.hpp:122-141 falls back to a sequential for under `#else` of `#if defined(_OPENMP)`), making OpenMP data races undetectable by the job that exists to catch them. (5) Dockerfile.ubuntu:29 explicitly installs libomp-dev, proving the dependency is real and known — it was just never mirrored to the Chimera image/CI. Root CLAUDE.md §2.5 says 'OpenMP from the toolchain', i.e. OpenMP is a required part of the toolchain contract ('CPU-first. OpenMP for data-parallel loops'), so a silently-serial canonical build violates project intent; the serial fallback in parallel.hpp is a portability fallback, not a sanctioned CI configuration. No refutation angle survives: cmake/clang-toolchain.cmake and CMakePresets.json add no -fopenmp, and no other layer installs libomp.

**Fix notes:** Fix is correct with small corrections: (a) exact Chimera package names are `libomp` (runtime) and `libomp-devel` (headers/omp.h) — add `libomp-devel` (it pulls the runtime) to Dockerfile:15's apk list and to ALL THREE apk lines in ci.yml (build-test, format-tidy — its `cmake --preset debug` configure also probes OpenMP — and coverage). (b) Extend the Dockerfile:45 fail-fast sanity check with an -fopenmp compile-and-link smoke test so a future package-name drift is caught at image build, not silently at configure. (c) On making OpenMP REQUIRED: prefer REQUIRED only on Linux (or gate on a FENIX_REQUIRE_OPENMP=ON default in CI presets) plus a loud message(WARNING) otherwise — the CI matrix per root CLAUDE.md includes macOS arm64, where libomp comes from Homebrew and a hard REQUIRED could break contributor builds; the serial fallback in parallel.hpp is intentionally kept for such platforms. (d) Note ci.yml's coverage job apk line is slightly shorter (no clang-tools-extra) — update each list where it appears rather than assuming one shared definition.

**Location:** Dockerfile:15-29; .github/workflows/ci.yml:23; CMakeLists.txt:123-126

**Evidence:** Dockerfile installs `clang lld llvm libcxx-devel libunwind-devel clang-tools-extra cmake ninja git pkgconf ccache curl-devel zlib-ng-devel` — no `libomp`/`libomp-devel` (Chimera packages the OpenMP runtime separately from `clang`/`llvm`). ci.yml's apk line is the same minus ccache. CMakeLists:
```cmake
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(fenix_headers INTERFACE OpenMP::OpenMP_CXX)
endif()
```
`src/core/parallel.hpp` guards everything behind `#if defined(_OPENMP)`, so without libomp the build succeeds and `parallel_for` degrades to serial with no warning. (Dockerfile.ubuntu:29 *does* install `libomp-dev`, confirming the dependency is real and simply missing here.)

**Failure scenario:** in the canonical dev container and in CI, `omp.h`/`libomp` are absent → `find_package(OpenMP)` quietly fails → the entire pipeline runs single-threaded ("OpenMP for data-parallel loops" is the project's stated parallelism primitive). Worse for correctness: the TSan CI job compiles without `-fopenmp`, so **no OpenMP data race can ever be caught by the tsan gate** — the exact class of bug the job exists for. A race in a `#pragma omp parallel for` region ships green.

**Suggested fix:** add `libomp-devel` (and runtime) to Dockerfile and ci.yml apk lists; in CMakeLists, either make OpenMP `REQUIRED` on Linux or emit a loud `message(WARNING "OpenMP NOT found — fenix will run serial")` so a serial build is never silent.

**Outcome:** fixed — added `libomp-devel` to Dockerfile:15's apk list and to all three `docker run` apk lines in ci.yml (build-test, format-tidy, coverage). Extended the Dockerfile toolchain sanity-check `RUN` with an actual `-fopenmp` compile+link+run smoke test (`omp_get_max_threads() > 0`) so a future Chimera package-name/path drift is caught at image-build time. In CMakeLists.txt, `find_package(OpenMP)` is now `REQUIRED` when `CMAKE_SYSTEM_NAME STREQUAL "Linux"` (the canonical Docker/CI toolchain, where libomp-devel is always installed per the Dockerfile) and stays non-REQUIRED elsewhere with a loud `message(WARNING ...)` on the not-found path — per fix_notes (c), a hard REQUIRED on macOS would break contributor builds without Homebrew libomp. Verified locally on macOS (Homebrew libomp present): configure reports `Found OpenMP_CXX: -fopenmp=libomp (found version "5.1")`. Could NOT verify the Linux-REQUIRED branch or the Dockerfile smoke test directly (no Docker locally, no Linux box) — desk-checked: `libomp-devel` is Chimera's correct package name per the review's own live `pkgs.chimera-linux.org` lookup, and `find_package(OpenMP REQUIRED)` is standard CMake behavior (hard-errors with a clear message if `omp.h`/libomp are absent).

## [high/bug] fuzz preset cannot link: -fsanitize=fuzzer applied to every target injects libFuzzer's main alongside driver.cpp's main

**Verdict:** CONFIRMED — CMakePresets.json:69 sets FENIX_SANITIZE="address,fuzzer" and CMakeLists.txt:157-160 propagates -fsanitize=address,fuzzer as INTERFACE compile+link options on fenix_headers, linked by the fenix binary (CMakeLists.txt:193), fenix-fxfs (:222), and every test executable (:239-240) — all of which define main. Empirically verified the linker semantics: for Linux targets (clang++ --target=x86_64-unknown-linux-musl -### shows libclang_rt.fuzzer.a wrapped in --whole-archive), FuzzerMain.o's strong main is unconditionally pulled in, guaranteeing a duplicate-symbol lld failure on every executable. Chimera-Linux-in-Docker is the mandated build environment (CLAUDE.md §2.1/§4), so the canonical `cmake --preset fuzz` build tree cannot link. Grep confirms zero LLVMFuzzerTestOneInput / fuzzer-no-link occurrences repo-wide, and no fuzz build/test preset exists. Not refutable as a documented stub: tests/CLAUDE.md:35 marks fuzz scaffolding as stub, but the preset and the "address,fuzzer" cache-string documentation (CMakeLists.txt:24) present it as a working configuration.

**Fix notes:** Proposed fix is correct in direction; three refinements. (1) When 'fuzzer' appears in FENIX_SANITIZE, substitute 'fuzzer-no-link' in BOTH the INTERFACE compile and link options — at link time fuzzer-no-link pulls libclang_rt.fuzzer_no_main.a (sancov callbacks, no main), which is exactly what main-bearing executables need; then apply plain -fsanitize=fuzzer only on dedicated fuzz targets (tests/fuzz_*.cpp providing LLVMFuzzerTestOneInput, no main). (2) Note the duplicate-main failure is Linux-specific: on macOS clang links the fuzzer archive without --whole-archive and an own-main program links fine (verified with Homebrew clang 22) — so a naive macOS smoke test would NOT catch this regression; the CI smoke job must run on the Linux runner. (3) The missing 'fuzz' entry in buildPresets/testPresets (CMakePresets.json:77-92) should be added alongside, otherwise `cmake --build --preset fuzz` fails even after the link fix. The absent fuzz targets themselves are an acknowledged stub (tests/CLAUDE.md:35), so adding one seed fuzzer is sufficient for the smoke job.

**Location:** CMakePresets.json:69; CMakeLists.txt:157-161

**Evidence:**
```json
{ "name": "fuzz", ... "FENIX_SANITIZE": "address,fuzzer" }
```
```cmake
if(FENIX_SANITIZE)
  string(REPLACE ";" "," _san "${FENIX_SANITIZE}")
  target_compile_options(fenix_headers INTERFACE -fsanitize=${_san} -fno-omit-frame-pointer)
  target_link_options(fenix_headers INTERFACE -fsanitize=${_san})
endif()
```
`fenix_headers` is linked by the `fenix` executable and all ~70 test binaries, each of which defines `main`. Linking with `-fsanitize=fuzzer` pulls in libclang_rt.fuzzer, which also defines `main` → duplicate-symbol link failure on every executable. Additionally there are no fuzz targets at all: `tests/` contains only `test_*.cpp` (no `fuzz_*` glob, no `LLVMFuzzerTestOneInput` anywhere), and CMakePresets has no `fuzz` build/test preset.

**Failure scenario:** `cmake --preset fuzz && cmake --build build-fuzz` fails at the first executable link. The fuzzers that CLAUDE.md designates as the home of property/invariant checks ("Property/invariant checks live in the fuzzers", fuzz listed as a CI gate) do not exist and cannot be built.

**Suggested fix:** use `-fsanitize=fuzzer-no-link` in the INTERFACE flags when `fuzzer` appears in FENIX_SANITIZE, and add a `tests/fuzz_*.cpp` glob whose targets link with `-fsanitize=fuzzer` proper (they provide `LLVMFuzzerTestOneInput`, no main). Add fuzz build preset + a CI job (even a 60-second smoke run per target).

**Outcome:** fixed — restructured per fix_notes: CMakePresets.json's `fuzz` preset now sets `FENIX_SANITIZE: "address"` + a new `FENIX_FUZZ: "ON"` cache var (no more `"address,fuzzer"` in FENIX_SANITIZE). CMakeLists.txt applies `-fsanitize=fuzzer-no-link` INTERFACE-wide on `fenix_headers` when `FENIX_FUZZ` is on (sancov instrumentation, no `main`, so `fenix` and every `test_*.cpp` still link) and a new `tests/fuzz_*.cpp` glob builds dedicated targets with plain `-fsanitize=fuzzer` added only on those targets. Added the missing `"fuzz"` entries to both `buildPresets` and `testPresets` arrays (`cmake --build --preset fuzz` previously failed at preset lookup before even reaching the link error). Verified end-to-end ON THIS MAC: `cmake --preset fuzz -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ ...` configures cleanly (reports "fenix fuzz: 5 libFuzzer target(s)"), and `cmake --build build-fuzz` links all 5 new `fuzz_*` targets successfully (Homebrew clang 22 supports libFuzzer natively, confirmed with a standalone smoke compile). Per fix_notes point (2), the Linux duplicate-main failure mode is NOT reproducible on macOS (clang links the fuzzer archive without `--whole-archive` there) — this local verification proves the preset *shape* is correct (fuzzer-no-link vs fuzzer separation, preset plumbing) but does NOT prove the original Linux link failure is fixed; that needs a Chimera/Linux CI run, which was not available locally (no Docker). Ran all 5 harnesses for ~60s each locally with no crashes (see apps-tools-tests.md fuzz-harness findings below for per-target results and one caveat about a since-fixed bug the fuzzer no longer reproduces).

## [high/design] Split build (FENIX_SPLIT) is never CI-verified, and its glob makes a missing module TU a silent stage dropout

**Verdict:** CONFIRMED — All cited facts verified: .github/workflows/ci.yml matrix is [release, debug, asan, ubsan, tsan] with no split/ml preset, and ci.yml is the ONLY file in .github/workflows/ despite its closing comment claiming arm64/macOS and self-hosted heavy/GPU jobs 'live in separate workflow files' — the CLAUDE.md §5.3 bench-regression/arm64/macOS/ML/GUI gates exist nowhere. CMakeLists.txt split branch confirms file(GLOB ... src/*/*.cpp), and apps/driver.cpp:11-14 under FENIX_SPLIT includes only core/config/archive, so split-mode stage registration depends entirely on each module's linked .o; a module in fenix.hpp lacking src/<mod>/<mod>.cpp links cleanly with its stage silently absent. No configure-time sync check exists and no test references FENIX_SPLIT (grep tests/ → nothing), so ADR 0008's 'byte-for-byte equivalent binary' promise is unenforced — the ADR itself only says 'Keep the unit list in sync with fenix.hpp' as a manual convention. The dev GPU box uses the split-based ml preset (CMakePresets.json:28, MEMORY.md), so dev/CI binary divergence with no red signal is a reachable scenario. Not refuted: currently all 16 module .cpps exist (no stage dropped today), but the missing guard and false workflow comment are real and the failure path is reachable.

**Severity adjusted to:** medium

**Fix notes:** Adding the split preset to the CI matrix is cheap and correct (ADR 0008: clean split ≈8.8s vs unity 9.7s). Corrections: (1) the ml preset cannot join the hosted Chimera matrix — libtorch is glibc-based (CLAUDE.md §2.1 watch-item) and belongs on the promised self-hosted runner workflow; (2) stronger than glob+cross-check: generate the unit TU list at configure time by parsing fenix.hpp's #include lines (excluding gui — documented firewalled, no gui.cpp per ADR 0008 — and fs, which is its own binary; adding ml/inference.cpp), making desync impossible rather than detected; (3) a test diffing the stage list ('fenix help' output) between unity and split binaries would directly enforce ADR 0008's 'identical 35 stages' claim; (4) soften the ADR's 'byte-for-byte equivalent' wording — the ADR's own split-only extern-template codec firewall means binaries are behaviorally, not byte-, equivalent.

**Location:** .github/workflows/ci.yml:20-21, 70-71; CMakeLists.txt:176-181

**Evidence:** CI matrix is `[release, debug, asan, ubsan, tsan]` — no `split`, no `ml`. The closing comment claims "arm64 + macOS-arm64 matrix and the self-hosted heavy/GPU jobs ... live in separate workflow files", but `.github/workflows/` contains only `ci.yml`; the arm64/macOS matrix, bench-regression, ml/gui jobs required by CLAUDE.md §5.3 exist nowhere. Meanwhile the split build is glob-driven:
```cmake
file(GLOB FENIX_UNIT_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*/*.cpp)
...
target_compile_definitions(fenix PRIVATE FENIX_SPLIT)
```
Under `FENIX_SPLIT`, driver.cpp includes only what `main()` touches and relies on each module's `src/<mod>/<mod>.cpp` static initializers for stage registration. `src/core`, `src/gui`, and any future module without a `<mod>.cpp` contribute nothing to the glob.

**Failure scenario:** a new module `src/foo/foo.hpp` with a self-registering stage but no `src/foo/foo.cpp` builds and works in unity (umbrella pulls it in) but its subcommand/stage silently vanishes from every split/ml dev binary — the exact ADR 0008 "equivalent binary" promise, unenforced by any CI job. Since the GPU box uses the `ml` (split) preset, dev and CI binaries can diverge in observable behavior without any red signal.

**Suggested fix:** add `split` to the CI matrix (it also compiles fastest wall-clock, so it's cheap); add a configure-time check that every `src/<mod>/` containing a module umbrella has a matching `<mod>.cpp` (or generate the unit TUs instead of globbing); either write the promised extra workflow files or delete the comment claiming they exist.

**Outcome:** fixed — added `split` to the `build-test` job's matrix in ci.yml (`[release, debug, asan, ubsan, tsan, split]`); per fix_notes (1), did NOT add `ml` to the hosted matrix since libtorch is glibc-based and belongs on a self-hosted runner workflow that doesn't exist yet. Added a configure-time sync check in CMakeLists.txt (in the `FENIX_SPLIT` branch): it parses `src/fenix.hpp`'s `#include "<mod>/<mod>.hpp"` lines via `file(STRINGS ... REGEX ...)`, and for every module found (excluding `core`, included directly by driver.cpp in both modes, and `gui` when `FENIX_GUI` is off) asserts `src/<mod>/<mod>.cpp` exists, else `message(FATAL_ERROR ...)` listing exactly which module(s) are missing. This makes desync a hard configure error instead of a silent stage dropout, per fix_notes point (2) (a generated-TU-list approach was considered but the existence-check achieves the same "impossible to desync silently" guarantee with far less CMake surface). Also fixed the stale ci.yml closing comment claiming arm64/macOS/self-hosted workflow files exist elsewhere — reworded to state plainly they don't exist yet (per fix_notes point (4), rather than inventing those files, which are out of this task's scope). Did NOT add the ADR-0008 stage-list-diff test (fix_notes point (3), unity vs split `fenix help` output) — that touches `tests/test_driver.cpp`-shaped test code, which is other agents' territory per the task's file-ownership rules; noting it here as a good follow-up. Verified locally: `cmake --preset split ...` configures cleanly (no FATAL_ERROR — confirms the sync check finds zero missing units against the current tree, matching the review's own observation that "currently all 16 module .cpps exist") and `cmake --build --target fenix` links the split `fenix` binary successfully. Did not additionally verify the ADR 0008 "identical stage list" claim between unity and split binaries (that needs the `fenix help`-diff test from fix_notes point (3), out of scope here).

## [medium/hygiene] include-cleaner CI gate is `|| true` — the documented-blocking IWYU check can never fail

**Verdict:** unverified (medium/low)

**Location:** .github/workflows/ci.yml:55

**Evidence:**
```yaml
      - name: include-cleaner (IWYU)
        run: find src -name '*.hpp' | xargs clang-include-cleaner --p build-debug || true
```
CLAUDE.md §5.3: "clang-tidy (…) + clang-include-cleaner" is listed under "CI gates (block merge)". The `|| true` swallows every outcome, including the tool failing to run at all (so even a typo in `--p` or a missing binary is invisible).

**Failure scenario:** headers violating the "transitive, self-contained includes" invariant (§2.2, enforced-by-IWYU per CLAUDE.md) merge green forever; the step is pure theater and its output isn't even surfaced as a warning annotation.

**Suggested fix:** drop `|| true`; if the tool is currently too noisy to block, run it with `--edit`-free print mode, diff against a checked-in allowlist, and fail on new violations — or at minimum `continue-on-error: true` at the step level so failures remain visible in the UI instead of being rewritten to success.

**Outcome:** fixed — dropped the `|| true` from the include-cleaner step in ci.yml (now `find src -name "*.hpp" | xargs clang-include-cleaner -p build-debug`, no swallowing). Did not add an allowlist/baseline-diff mechanism (fix notes flagged that as a fallback only if the tool proves too noisy to block outright; no evidence gathered here that it is). Could not verify how noisy clang-include-cleaner actually is against the current tree (no Docker/Chimera locally, and running clang-include-cleaner requires a `compile_commands.json` from a full configure matching the CI toolchain) — flagging as desk-check only; if the first real CI run turns up a wall of pre-existing violations, the allowlist fallback from the suggested fix should be applied then rather than immediately reintroducing `|| true`.

## [medium/performance] Dockerfile's env-var ccache launcher bypasses the CMakeLists sloppiness injection — PCH compiles never hit the persisted cache in-container

**Verdict:** unverified (medium/low)

**Location:** Dockerfile:40-42; CMakeLists.txt:48-67

**Evidence:** Dockerfile:
```
ENV ... CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache CCACHE_DIR=/ccache
```
CMakeLists:
```cmake
if(FENIX_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(FENIX_CCACHE_PROG NAMES ccache sccache)
  ...
      set(CMAKE_CXX_COMPILER_LAUNCHER
          "${CMAKE_COMMAND};-E;env;CCACHE_SLOPPINESS=pch_defines,time_macros,...;${FENIX_CCACHE_PROG}" ...)
```
CMake initializes `CMAKE_CXX_COMPILER_LAUNCHER` from the environment, so inside the container the `NOT CMAKE_CXX_COMPILER_LAUNCHER` guard is false and the whole block — including the `CCACHE_SLOPPINESS=pch_defines,...` injection that the 20-line comment declares essential for ccache+PCH coexistence — is skipped. Meanwhile `FENIX_CCACHE_PROG` is never set, so `FENIX_PCH_OK` evaluates ON (line 65: `NOT FENIX_CCACHE_PROG` is true) and PCH is enabled for all test TUs (and split module TUs).

**Failure scenario:** exactly the environment bootstrap.sh builds ("Persist ccache so the unity-build TU isn't recompiled cold every run", fenix-ccache volume): ccache without `pch_defines`/`time_macros` sloppiness refuses to cache PCH-consuming compiles and falls through to the real compiler every time — the ~66 test TUs plus every split TU are recompiled cold on each container run, silently defeating the persisted volume and the entire coexistence design verified in the comment.

**Suggested fix:** either set `ENV CCACHE_SLOPPINESS=pch_defines,time_macros,include_file_mtime,include_file_ctime` in the Dockerfile next to the launcher vars, or change the CMakeLists guard to also handle the "launcher is plain ccache from env" case by wrapping/augmenting it (and set `FENIX_CCACHE_IS_CCACHE` by inspecting the existing launcher, not only the find_program path).

**Outcome:** fixed — took the simpler of the two options: added `CCACHE_SLOPPINESS=pch_defines,time_macros,include_file_mtime,include_file_ctime` to the Dockerfile's `ENV` block next to the existing launcher vars. Did not touch the CMakeLists guard (the second option) since the env-var fix is sufficient and lower-risk. Could not verify the actual cache-hit behavior in-container (no Docker locally) — desk-checked: this is the exact sloppiness string CMakeLists.txt's own comment/injection already uses for the non-container path, so setting it as a container-wide env var achieves the same effect for the `NOT CMAKE_CXX_COMPILER_LAUNCHER`-false path taken inside Docker.

## [low/correctness] -march=native objects + a ccache dir persisted across heterogeneous hosts can reuse wrong-ISA objects

**Verdict:** unverified (medium/low)

**Location:** CMakeLists.txt:118; bootstrap.sh:24 (fenix-ccache volume); memory-documented RunPod flow (CCACHE_DIR on a reattachable volume)

**Evidence:** Release adds `-ffast-math -march=native`; ccache hashes the literal flag string `-march=native` (plus preprocessed source and compiler identity), not the host CPU. The dev flow persists CCACHE_DIR on volumes that outlive the machine (docker volume; RunPod network volumes reattach to pods with different CPU models).

**Failure scenario:** build on a pod with AVX-512, volume reattaches to a pre-AVX-512 pod, identical compiler + sources → ccache hits return AVX-512 objects → `fenix` dies with SIGILL (or worse, only in rarely-hit kernels). No build error, no signal.

**Suggested fix:** set `CCACHE_COMPILERCHECK=%compiler% -v -march=native -E -x c++ /dev/null` style content check, or simply add the CPU model to the hash (`CCACHE_EXTRAFILES=/proc/cpuinfo` is crude but effective), or pin `-march=<baseline>` per fleet instead of `native` when a shared cache is in play.

**Outcome:** fixed — added exactly the suggested `CCACHE_COMPILERCHECK="%compiler% -v -march=native -E -x c++ /dev/null"` to the Dockerfile's `ENV` block. This expands and hashes the actual target-feature output of `-march=native` for the host it's running on, so a ccache volume reattaching to a different CPU model gets a different hash and correctly misses instead of returning wrong-ISA objects. Could not verify against real heterogeneous RunPod hosts (no such environment available here) — desk-checked: this is ccache's documented mechanism for exactly this class of problem (content-based compiler check vs. the default mtime/size check), and the suggested fix's first option.

## [low/hygiene] CMAKE_LINKER_TYPE requires CMake ≥ 3.29 but the declared minimum is 3.28 — silently ignored, wrong linker

**Verdict:** unverified (medium/low)

**Location:** cmake/clang-toolchain.cmake:13; CMakeLists.txt:3; CMakePresets.json:3

**Evidence:** `set(CMAKE_LINKER_TYPE LLD)` — this variable was introduced in CMake 3.29. `cmake_minimum_required(VERSION 3.28)` and the presets' `cmakeMinimumRequired` 3.28 admit a CMake that ignores unknown variables silently and links with the platform default (GNU ld/bfd on Ubuntu), violating the "lld only, no GNU binutils" invariant with zero diagnostics. (install-runpod-ubuntu2404.sh even calls out "apt's cmake 3.28" as the version it upgrades away from — a user skipping that script gets exactly this.)

**Failure scenario:** configure with cmake 3.28 on Ubuntu → binary links with bfd; project-invariant toolchain claims (and any lld-specific behavior, e.g. `--icf`, LTO interactions later) silently don't hold.

**Suggested fix:** bump `cmake_minimum_required`/preset minimum to 3.29, or additionally `add_link_options(-fuse-ld=lld)` in the toolchain file as a version-proof enforcement.

**Outcome:** fixed — did both, per the "or" reading as "and" for defense in depth: bumped `cmake_minimum_required(VERSION 3.29)` in CMakeLists.txt (was 3.28) and `cmakeMinimumRequired` to `{3,29,0}` in CMakePresets.json, AND added `add_link_options(-fuse-ld=lld)` in cmake/clang-toolchain.cmake right after `set(CMAKE_LINKER_TYPE LLD)` as a version-proof belt-and-suspenders enforcement. Verified locally: configure with local CMake 4.3.3 (well above 3.29) succeeds with no warnings about the minimum-version bump; did not have a CMake 3.28 binary available locally to confirm the old failure mode, but the fix is mechanical (a version_required bump plus a flag CMake has supported since 3.13).

## [low/performance] profiling and relwithdebinfo presets omit fast-math/-march=native — profiles measure different codegen than the shipped binary

**Verdict:** unverified (medium/low)

**Location:** CMakePresets.json:72-75; CMakeLists.txt:117-119

**Evidence:** the FENIX_FAST flags are gated `$<$<CONFIG:Release>:...>`; the `profiling` preset inherits `relwithdebinfo` (`CMAKE_BUILD_TYPE=RelWithDebInfo`), so it builds `-O2 -g` **without** `-ffast-math -march=native -O3 -funroll-loops`.

**Failure scenario:** §5.2 requires before/after performance measurement; anyone profiling with the preset named "profiling" optimizes hot spots that vectorize completely differently (fast-math enables reassociation/FMA/vectorization of reductions) under the real release flags — wasted tuning or missed regressions.

**Suggested fix:** extend the genex to `$<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:...>`, or make `profiling` inherit `release` and add `-g -fno-omit-frame-pointer` via `CMAKE_CXX_FLAGS`.

**Outcome:** fixed — took exactly the first suggested option: extended the FENIX_FAST genex from `$<$<CONFIG:Release>:...>` to `$<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:...>` in CMakeLists.txt, so both `relwithdebinfo` and `profiling` (which inherits it) now get `-ffast-math -march=native -O3 -funroll-loops` alongside `-O2 -g`, matching the shipped Release codegen. Verified locally: `cmake --preset release ...` still configures/builds/links clean (the genex change is additive for a new CONFIG value, doesn't touch the Release path). Verified with `cmake --preset profiling ... && cmake --build --target fenix -v`: the compile command now includes both `-ffast-math` and `-march=native`, confirming the genex fix is live for the actual preset named in the finding.

## [low/bug] install-runpod: `set -e` aborts silently in the symlink loop, and the apt line that prevents that state is `|| true`-masked

**Verdict:** unverified (medium/low)

**Location:** install-runpod-ubuntu2404.sh:51-59 (also 126)

**Evidence:**
```sh
$SUDO apt-get install -y --no-install-recommends \
  clang-tools-22 libclang-rt-22-dev libc++-22-dev libc++abi-22-dev \
  lld-22 clang-tidy-22 clang-format-22 libomp-22-dev >/dev/null 2>&1 || true
...
for p in clang clang++ ld.lld lld llvm-ar ...; do
  [ -e "/usr/bin/$p" ] || { [ -e "/usr/bin/$p-22" ] && $SUDO ln -sf "/usr/bin/$p-22" "/usr/bin/$p"; }
done
```
If both `/usr/bin/$p` and `/usr/bin/$p-22` are missing, the brace group's status is the failed `[ -e ... ]` → the `||` RHS fails → `set -eu` kills the script mid-loop with no message. That state is reachable precisely because the line installing `lld-22`/`libc++-22-dev` swallows all failures (`>/dev/null 2>&1 || true`) — e.g. `command -v clang-22` succeeds from a partial earlier run so `llvm.sh` is skipped, apt fails transiently, and the script then dies (or worse, proceeds to configure without libc++-22-dev and fails deep inside CMake's ABI probe). Same silent-exit pattern at line 126: `"$BIN" --help ... && echo "core: ok"` — a failed smoke test exits nonzero with no diagnostic.

**Suggested fix:** loop body `if [ ! -e "/usr/bin/$p" ] && [ -e "/usr/bin/$p-22" ]; then $SUDO ln -sf ...; fi`; drop the `|| true` from the tooling apt line (or at least check the critical packages afterward and error with a message); make the smoke test print a failure line before exiting.

**Outcome:** fixed — did exactly the three suggested fixes: (1) the symlink loop body is now `if [ ! -e "/usr/bin/$p" ] && [ -e "/usr/bin/$p-22" ]; then $SUDO ln -sf ...; fi` (no more brace-group-status-under-set-e footgun); (2) dropped `|| true` from the clang-tools-22/libc++-22-dev apt line — it now checks the `apt-get install` exit status directly and `exit 1`s with a clear message on failure instead of silently continuing into a broken symlink loop or a deep CMake ABI-probe failure; (3) the line-126 smoke test (`"$BIN" --help`) now prints "ERROR: smoke test failed..." to stderr and exits 1 on failure instead of silently falling through. Verified with `sh -n install-runpod-ubuntu2404.sh` (syntax OK). Could NOT run the script end-to-end (it targets a RunPod Ubuntu 24.04 GPU box with apt/sudo/a preinstalled torch — none of which exist in this sandbox); this is a desk-check + syntax-check only.

## [low/hygiene] bootstrap.sh uses `docker run -it` unconditionally — fails with no TTY

**Verdict:** unverified (medium/low)

**Location:** bootstrap.sh:22

**Evidence:** `"$ENGINE" run --rm -it ...` inside `run()`, used by all subcommands including `ci`.

**Failure scenario:** `./bootstrap.sh ci` from cron, a git hook, or any CI/automation context (no TTY) fails immediately with "the input device is not a TTY" — the one subcommand explicitly designed as "the full local CI gate" is the one most likely to be scripted.

**Suggested fix:** `TTY=""; [ -t 0 ] && TTY="-it"` and pass `$TTY` (unquoted) in the run() invocation, or use `-i` always and add `-t` only when stdin is a terminal.

**Outcome:** fixed — took the second suggested form: `run()` now always passes `-i` and conditionally adds `-t` (`TTY_FLAG` set to `-t` only when `[ -t 0 ]`, empty otherwise, passed unquoted so an empty value doesn't become a literal empty-string arg to `docker run`). Verified with `sh -n bootstrap.sh` (syntax OK). Could not exercise `./bootstrap.sh ci` end-to-end (no Docker locally) — desk-checked: `docker run --rm -i $TTY_FLAG ...` with `TTY_FLAG=""` is the standard non-TTY-safe invocation pattern and matches the suggested fix exactly.
