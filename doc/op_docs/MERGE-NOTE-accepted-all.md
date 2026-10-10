# MERGE NOTE — `op/accepted-all`

**Purpose.** One branch the lead can merge into `main` instead of resolving the same
`api/torch_ext.cpp` registration table seven times.

**Server date:** 2026-10-09 (UTC). **Base:** `main` = `31621a19bded927f9c4d6d7cf986c4bfd738bc94`.
**No accelerator card was used.** Everything below is CPU/git/host-side evidence only.

---

## 1. What this branch consolidates

Five merges onto `main`, in this order (each `--no-ff`, so every contributing branch becomes an
ancestor of `op/accepted-all` and a later re-merge of any of them is a no-op):

| # | Merged branch | Source sha | What it brings |
| --- | --- | --- | --- |
| 1 | `op/flash-attention-d256-d512-tile` | `3d6793a` | Gemma D256/BM64 + D512/BM32 tiling, Step B rescale SIMD, Step E SV-add SIMD, explicit `softmax_scale`; carries upstream `#30` as prerequisite `e029b35` |
| 2 | `op/ms_deform_attn` | `d1a884b` | Deformable-DETR MSDA forward + first-order backward |
| 3 | `op/rms-norm-epilogue-simd-dma` | `af57110` | RMSNorm second-pass SIMD epilogue + plain-DMA output overlap |
| 4 | `op/reshape-and-cache-abi` | `d54a06f` | Non-contiguous reshape-and-cache ABI (see §3 — already in `main`) |
| 5 | `op/flash-attention-d256-d512-tile` (again) | `795bfb8` | D512/BM32 `cur_BM`-scoped post-processing loops (+9 lines, `doc/op_docs/curbm-loops-note.md`) |

Plus one follow-up commit that drops a duplicated `#include <limits>` left behind by the
automatic merge of branch 2.

### ⚠ Merge 5 exists because the source branch moved during this work

`op/flash-attention-d256-d512-tile` was `3d6793a` when this task started (10:16 UTC,
confirmed by `git ls-remote --heads`). At **10:24:49 UTC** another workstream pushed `795bfb8`
onto it. It is **forward progress, not a force-push**: `git merge-base --is-ancestor 3d6793a
795bfb8` is true. A consolidation has to reflect the source branch's *current* head, so merge 5
folds it in. `teco/ual/kernel/flash_attention/flash_attention.scpp`
in `op/accepted-all` is now blob `d542b9a33c031320c3202c5ba1a96f04fdc24bc1`, **byte-identical**
to that branch's scpp (`git diff` between them is empty). Re-check the branch head before
merging this into `main` — it may have moved again.

### Acceptance inventory (asserted on the final tree)

The tree index used for this assertion **excludes `doc/op_docs/MERGE-NOTE-accepted-all.md`**
(this file), because §3b below quotes the rejected hunks verbatim and would otherwise credit
lines that exist in no source file.

| Branch | distinct non-blank lines added vs `main` | present in final tree | dropped |
| --- | ---: | ---: | ---: |
| `op/flash-attention-d256-d512-tile` | 420 | **420** | 0 |
| `op/ms_deform_attn` | 1869 | **1869** | 0 |
| `op/rms-norm-epilogue-simd-dma` | 71 | **71** | 0 |
| `op/reshape-and-cache-abi` | 34 | 15 | 19 (all superseded/regressive — §3) |

`api/torch_ext.cpp` registers the union, with the real lines:

```
374    m.def("ms_deform_attn_backward", &ms_deform_attn_backward_torch,
376    m.def("flatten_rays", &flatten_rays_torch, "flatten_rays (SDAA)");
377    m.def("morton3D_invert", &morton3D_invert_torch, "morton3D_invert (SDAA)");
378    m.def("reshape_and_cache", &reshape_and_cache_torch, "reshape_and_cache (SDAA)");
379    m.def("rms_norm", &rms_norm_torch, "rms_norm (SDAA)");
380    m.def("flash_attn_varlen_func", &flash_attn_varlen_func_torch, "flash_attn_varlen_func (SDAA)");
381    m.def("causal_conv1d_fn_torch", &causal_conv1d_fn_torch, "causal_conv1d_fn_torch (SDAA)");
382    m.def("ms_deform_attn_forward", &ms_deform_attn_forward_torch,
```

`tecoopsSetStream` call sites in the same file: L103 (`rms_norm_torch`), L151
(`flash_attn_varlen_func_torch`), L282 and L343 (the two MSDeformAttn entries). None dropped.

`teco/ual/kernel/flash_attention/flash_attention.scpp` is the flash-attention branch's version
(upstream PR41 head). Upstream `#30` is still in it: `compute_exp_sum` absent, the
`for (i = 0; i <= size - 16; i += 16)` bound and the `for (; i < size; i++)` tail loop present,
the old `i < size % 16` tail gone.

---

## 2. Compile (CPU only)

```
source /opt/tecoai/setvars.sh
rm -rf build api/tecoops/*.so          # clean, from scratch
WITH_TORCH=ON WITH_INFERENCE_PLUGIN=OFF /home/py312/bin/python setup.py build_ext --inplace
```

**Result: `rc=0`** on a build with `build/` and both `.so` files removed first.
`libteco_ops.so built successfully!` and the torch extension compiled and linked.

Artifacts (sha256):

```
cbb19129ee6df076687a17f32708fc35091816b8fc39e37eb05f266b983f51d9  api/tecoops/libteco_ops.so
2961b5cc2df59a5ab6731564097f4420c12182e91a1652b845d05ff756471973  api/tecoops/_torch_ext.cpython-312-loongarch64-linux-gnu.so
```

`git status --short` is empty afterwards — build outputs are ignored.

**Every** warning/error line in the whole log (3 lines, all pre-existing/environmental,
verbatim):

```
Error in cpuinfo: processor architecture is not supported in cpuinfo
/home/py312/lib/python3.12/site-packages/torch_sdaa/utils/cpp_extension.py:198: UserWarning: torch_sdaa and extension build with different sdaa_runtime version
  warnings.warn('torch_sdaa and extension build with different sdaa_runtime version')
```

The `-Wall` compile of `api/torch_ext.cpp` (the merged file, 385 lines) produced **no**
compiler diagnostics. The built library carries the new device stubs, e.g.
`__device_stub__teco_slave_ms_deform_attn_forward_fp16/fp32`,
`…_backward_fp16/fp32`, `…_backward_list_reduce_*`, `…_backward_list_producer_*`. This proves
the sources compile and link; it says nothing about kernel correctness (see §5).

---

## 3. What was deliberately NOT integrated, and why

### 3a. All four `codex/*` branches — nothing to integrate (they are ancestors of `main`)

This differs from the pre-work table, so it is worth stating plainly. `main`'s graph already
contains `Merge pull request #1..#4` for these branches, and
`git rev-list --count main..codex/<branch>` is **0** for every one of them:

| Branch | sha | commits ahead of `main` | `git diff main <branch>` |
| --- | --- | ---: | --- |
| `codex/internvl-flash-no-output-clear` | `bda6b7e` | 0 | empty |
| `codex/internvl-flash-device-lengths` | `e0f0127` | 0 | empty |
| `codex/internvl-rms-norm-current-stream` | `64a7f58` | 0 | empty |
| `codex/minicpm5-1b-reshape-abi` | `1a06e06` | 0 | empty |

`git merge-base --is-ancestor <sha> main` is true for all four. Re-applying any of them would
be a no-op at best; the "lines they add vs `main`" that the assertion reports as absent are in
fact `main`'s **newer** content that these older branches predate (for example `out =
torch::zeros_like(q)` / `out.zero_()`, which PR #4 replaced with `empty_like` + a sentinel-seeded
test, and the CPU `q_seq_lens` round trip, which PR #3 replaced with the device-side subtraction).
So they are excluded by evidence, not by omission.

### 3b. `op/reshape-and-cache-abi` — an unrelated history whose tree is *stale*

**`op/reshape-and-cache-abi` shares no common ancestor with `main`** (`git merge-base main
origin/op/reshape-and-cache-abi` is empty; its root is upstream's `e2d7130 Init this repo`, while
`main`'s root is `0f970e2 Initial commit`). It was therefore merged with
`--allow-unrelated-histories`, which treats an empty tree as the base and raises an add/add
conflict on every differing path: **20 files, 58 conflict blocks**.

Its substantive contribution is a single commit, `d54a06f`, touching only three files. All three
are already in `main`:

| `d54a06f` touches | status vs `main` |
| --- | --- |
| `api/torch_ext.cpp` (reshape ABI materialization) | already present — `key.contiguous()` / `value.contiguous()` at the reshape ABI boundary; only a comment's double-space differs |
| `python_api_test/test_reshape_and_cache.py` | **byte-identical** to `main` (sha256 `a1e8b80c…`) |
| `doc/op_docs/reshape_and_cache.md` | main lacked it — **kept here** (see below) |
| `teco/interface/ops/reshape_and_cache.cpp` | **byte-identical** to `main` (sha256 `0f8050ac…`) |
| `teco/ual/kernel/flash_attention/flash_attention.scpp` | **byte-identical** to upstream `#30` (`sha256 389c44b5…`), i.e. exactly what the flash-attention branch already carries as `e029b35` |

The reshape ABI work reached `main` first through the merged `codex/minicpm5-1b-reshape-abi`
(PR #1).

**The one genuinely new thing** the branch has is the “PyTorch ABI” note in
`doc/op_docs/reshape_and_cache.md`. That note **is kept** in this branch — it is the only
keep-both in the whole merge. Everything else on its side of the 20 conflicts is the pre-PR#2/#3/#4
state, and re-introducing it would silently revert accepted work:

```
    // parse q_seq_lens
    auto cu_seqlens_q_cpu = cu_seqlens_q.cpu();
    auto cu_ptr = cu_seqlens_q_cpu.data_ptr<int>();
    auto q_lens_cpu = torch::empty({batch_size}, torch::kInt32);
    auto ql_cpu_ptr = q_lens_cpu.data_ptr<int>();
    for (int b = 0; b < batch_size; b++) {
        ql_cpu_ptr[b] = cu_ptr[b + 1] - cu_ptr[b];
    }
    auto q_lens = q_lens_cpu.to("sdaa");
    ...
    if (!out.defined()) {
        out = torch::zeros_like(q);
    } else {
        out.zero_();
    }
```

reverts PR #3 (device-side `q_seq_lens`) and PR #4 (no redundant output clear), and

```
- ### 输出初始化契约
- ### PyTorch 绑定的变长元数据
- ### PyTorch 当前流绑定
| q_seq_lens     | 输入      | 主机端        | ... |
| 测试环境    | 测例                 | 硬件时间 (us) |
    out_dev = torch.empty_like(q_dev)
```

would delete `main`'s PR #2/#3/#4 documentation, revert the sentinel-seeded output test, and fix
the `q_seq_lens` row back to “主机端”. These alternatives are **mutually exclusive** with the
accepted code (`out` cannot be both `empty_like` and `zeros_like`), so they are not applied. The
19 lines are accounted for above rather than silently dropped.

### 3c. Upstream `#30` was not double-applied

Both `op/reshape-and-cache-abi` (`de27305`) and `op/flash-attention-d256-d512-tile` (`e029b35`)
carry `#30`, and the two commits have the **same stable patch-id** `c6c5f965c350e2366fa810a45be149418292e645`.
`flash_attention.scpp` resolves to the flash-attention branch's version, which is upstream PR41
head and a strict superset of the `#30`-only text (proved by byte-identical sha256
`389c44b5…` for the `#30` tree, and by the assertion that all 9 scpp lines the reshape branch
adds are present verbatim).

---

## 4. Conflicts and resolutions

| File | blocks | resolution |
| --- | ---: | --- |
| `api/torch_ext.cpp` | 7 | consolidated side (includes + scale + MSDA fns + stream bindings) |
| `doc/op_docs/flash_attention.md` | 10 | consolidated side |
| `teco/ual/kernel/flash_attention/flash_attention.scpp` | 6 | consolidated side |
| `teco/interface/ops/flash_attention.cpp` | 5 | consolidated side (keeps both `tecoopsFlashAttention` and `…WithScale`) |
| `python_api_test/test_flash_attention.py` | 4 | consolidated side (sentinel seed kept) |
| `teco/ual/kernel/rms_norm/rms_norm_fp16.scpp` | 4 | consolidated side |
| `teco/interface/include/tecoops.h` | 3 | consolidated side |
| `README.md` | 2 | consolidated side |
| `doc/op_docs/rms_norm.md` | 2 | consolidated side |
| `teco/ual/ops/flash_attention/flash_attention.hpp` | 2 | consolidated side |
| `test/test_proto/tecokernel.proto` | 2 | consolidated side |
| `test/zoo/teco/flash_attention/flash_attention.cpp` | 2 | consolidated side |
| `test/zoo/teco/flash_attention/flash_attention.py` | 2 | consolidated side |
| `doc/op_docs/reshape_and_cache.md` | 1 | **keep both** — reshape branch's ABI note added |
| `api/tecoops/__init__.py` | 1 | consolidated side |
| `python_api_test/test_rms_norm.py` | 1 | consolidated side |
| `teco/ual/kernel/flash_attention/flash_attention.h` | 1 | consolidated side |
| `teco/ual/ops/flash_attention/find_flash_attention.cpp` | 1 | consolidated side |
| `test/test_proto/tecokernel/flash_attention.proto` | 1 | consolidated side |
| `test/zoo/teco/flash_attention/flash_attention.h` | 1 | consolidated side |
| **total** | **58** | |

Merges 1–3 and 5 were auto-merged by the `ort` strategy with **zero** conflicts; only the
unrelated-history merge 4 produced conflicts (20 files, all add/add against an empty base,
58 conflict blocks).

One cosmetic artifact: the auto-merge of `op/ms_deform_attn` kept **both** `#include <limits>`
lines (one from the flash-attention branch, one from MSDA). The duplicate line is removed in a
follow-up commit; `<limits>` is still included.

---

## 5. Device-level validation NOT done

**Do not read anything in this branch as device-validated.** Explicitly not performed here:

- No SDAA / accelerator-card run of any kind. The card was not touched (`teco-smi` not invoked
  from this workstream). **CARD STATE: free — not used.**
- No numerical or bitwise comparison against a reference for any operator (flash attention
  D256/D512, MSDA forward/backward, RMSNorm epilogue, reshape-and-cache).
- No performance measurement; no 3-run median; no steady-state run.
- No model integration, no greedy-32-token smoke for Gemma / Deformable-DETR / MiniCPM /
  InternVL / Hy-MT2.
- No `python_api_test/*` execution — those tests require `sdaa` tensors and were not run.
- No C++ `test/build.sh --arch teco` gtest run and no `test/zoo` case execution.
- Only the host-side compile in §2 was run. It proves the sources compile and link on this
  machine; it proves nothing about kernel correctness.

Each contributing branch carries its own device evidence and its own stated limits in
`doc/op_docs/*.md`; that evidence is **not** re-verified or extended by this consolidation, and
the noted upstream-CI/wheel/accuracy gaps in those documents still stand.

### 6a. Pre-existing observation (not introduced by this work)

`python_api_test/test_flash_attention.py:41` defines
`DATA_DIR = os.path.join(os.path.dirname(__file__), "fa_pt")`, and `python_api_test/fa_pt/`
does not exist — not on `main`, not on this branch, and not on any contributing branch
(`git ls-tree -r --name-only` finds 0 entries in all of them). The name is never referenced
anywhere else in that file, so it opens nothing and raises nothing; it is dead code that
predates this consolidation and was deliberately left untouched rather than "fixed" here.
No other `__file__` / `sys.path` expression added by the merged branches points at a missing
target (`Path(__file__).resolve().parents[1]` → `test/zoo/teco`, `parents[3]` → repo root).
## 6. Hygiene gates run on this branch

- `git diff --check main` — clean.
- `bash -n` on every shell script the branch touches — see the worker report; none is touched.
- `python -m json.tool` on every JSON file the branch touches — see the worker report; none is touched.
- Path-existence gate — no file was relocated by this work (`git diff --name-status main` contains
  no renames), so the gate is satisfied vacuously; the check is stated in the report rather than
  silently skipped.
- `git ls-remote --heads` after pushing confirms `main` and all eight pre-existing branches are at
  their original shas.
