#!/usr/bin/env python3
# BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

"""Python-API conformance test for ``tecoops.weight_stream_gemv``.

Operator contract (mirrors ``test/zoo/teco/weight_stream_gemv/weight_stream_gemv.cpp``):

    x      : fp16, shape [k]
    weight : fp16, shape [n, k]
    out    : fp16, shape [n]
    out[i] = fp16( sum_j fp32( x[j] * weight[i][j] ) )
    accepted only when n % 256 == 0 and k % 16 == 0 (the kernel flushes each
    core's output with one 16 B-aligned DMA); anything else is rejected.

The model-facing wrapper ``tecoops.linear_stream(x, weight, bias)`` must route a
real M=1 decode row (2-D, half, contiguous, n % 256 == 0, k % 16 == 0) to that
same GEMV kernel, and must fall back to the vendor GEMM (``torch.linear``) for
every other shape -- prefill, profiling, or an unsupported geometry.

Exit status is non-zero on any numerical mismatch, on a violated contract, or on
an unexpected fallback, so ``tools/ci_test.sh`` (``set -e``) fails the job.

Run exactly as CI runs it::

    cd python_api_test && python test_weight_stream_gemv.py

``--cpu-only`` skips every device step and is only a sanity path for the oracle
(it needs TORCH_DEVICE_BACKEND_AUTOLOAD=0 off-device); it is never what CI
exercises.
"""
import argparse
import json
import os
import sys
import tempfile

import _bootstrap  # noqa: F401  (adds the repo api/ tree when tecoops is not installed)
import torch

import tecoops

# --------------------------------------------------------------------------
# Tolerance
#
# The kernel accumulates in fp32 and rounds the finished accumulator to fp16
# once, so the only legitimate deviation from an unrounded fp32 reference is the
# final fp16 quantisation: at most half an ulp, i.e. 2**-11 relative.  We allow
# TOL_ULPS of them to absorb accumulation-order differences (the CPU reference
# matmul and the kernel do not add the k terms in the same order), plus a small
# absolute floor for rows whose reference lands near zero.
#
# Measured headroom is printed per case, so the margin is auditable rather than
# asserted here.
# --------------------------------------------------------------------------
TOL_ULPS = 2.0
TOL_ABS_FLOOR = 1e-4

REPORT = []
FAILURES = []


def record(fail, **kw):
    """Append one check. `fail` is the failure condition; returns True when it passed."""
    entry = dict(kw)
    entry["status"] = "FAIL" if fail else "PASS"
    REPORT.append(entry)
    if fail:
        FAILURES.append(entry)
    return not fail


def half_ulp_bound(reference_f32):
    """Per-element fp16 ulp of the reference magnitude, as a float32 tensor."""
    magnitude = reference_f32.abs().clamp_min(torch.finfo(torch.float32).tiny)
    exponent = torch.floor(torch.log2(magnitude))
    spacing = torch.pow(torch.tensor(2.0, dtype=torch.float32), exponent - 10.0)
    return spacing


def allowed_error(*references):
    """Largest deviation accepted against an fp32 reference.

    Summing the per-reference bounds makes the result independent of
    cancellation: a stage that rounds at a large intermediate magnitude is still
    policed by that intermediate's own ulp.
    """
    total = 0.0
    for reference in references:
        total += float((TOL_ULPS * half_ulp_bound(reference.reshape(-1))).max())
    return total + TOL_ABS_FLOOR


# --------------------------------------------------------------------------
# CPU oracle
# --------------------------------------------------------------------------

def make_inputs(n, k, seed, bound):
    """Deterministic fp16 inputs, uniform on [-bound, bound] like the zoo cases."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.rand(k, generator=generator, dtype=torch.float32) * 2.0 - 1.0) * bound
    w = (torch.rand(n, k, generator=generator, dtype=torch.float32) * 2.0 - 1.0) * bound
    return x.half(), w.half()


def reference_fp32(x, weight):
    """fp32 accumulation, the same precision the kernel accumulates in."""
    return torch.matmul(weight.float(), x.float())


def reference_fp64(x, weight):
    """Independent wider-precision oracle: separates kernel error from oracle error."""
    return torch.matmul(weight.double(), x.double()).float()


def compare(out_half, ref32, ref64, label, allowed=None, extra=None):
    got = out_half.float()
    err = (got - ref32).abs()
    err64 = (got - ref64).abs()
    max_err = float(err.max()) if err.numel() else 0.0
    max_err64 = float(err64.max()) if err64.numel() else 0.0
    if allowed is None:
        allowed = allowed_error(ref32)
    exact = float((out_half == ref32.half()).float().mean()) if out_half.numel() else 1.0

    info = dict(case=label, max_error=max_err, max_error_vs_fp64=max_err64,
                allowed=allowed, exact_round_fraction=exact)
    if extra:
        info.update(extra)

    print("  [%-28s] max_error=%.6e (vs fp64 %.6e)  allowed=%.6e  "
          "round_exact=%.4f  margin=%s  %s"
          % (label, max_err, max_err64, allowed, exact,
             ("%.2fx" % (allowed / max_err)) if max_err > 0 else "inf",
             "PASSED" if max_err <= allowed else "FAILED"), flush=True)

    if max_err > allowed:
        info["reason"] = "device output deviates from the fp32 reference beyond tolerance"
    return record(max_err > allowed, **info)


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

# (n, k, seed, bound, label) -- the first two reproduce the operator's own zoo
# cases (shapes and uniform distributions taken from the .prototxt files); the
# rest are single-row decode geometries that satisfy the documented guards.
CASES = [
    (256, 256, 41, 0.50, "zoo/case_0 n=256 k=256"),
    (512, 512, 51, 0.25, "zoo/case_1 n=512 k=512"),
    (256, 2048, 43, 0.50, "decode n=256 k=2048"),
    (256, 3072, 44, 0.50, "decode n=256 k=3072"),
    (512, 2048, 45, 0.25, "decode n=512 k=2048"),
]


def check_binding():
    print("=" * 72)
    print("1. binding and export contract")
    print("=" * 72)
    ok = True
    for name in ("weight_stream_gemv", "linear_stream"):
        present = hasattr(tecoops, name)
        is_callable = callable(getattr(tecoops, name, None))
        exported = name in getattr(tecoops, "__all__", ())
        print("  %-20s attribute=%s callable=%s in __all__=%s"
              % (name, present, is_callable, exported), flush=True)
        ok = record(not (present and is_callable and exported),
                    kind="binding", name=name) and ok
    print("  tecoops.__file__ = %s" % tecoops.__file__, flush=True)
    return ok


def check_cpu_oracle():
    """Oracle self-check: the two references must agree far below one fp16 ulp."""
    print("=" * 72)
    print("2. CPU oracle self-check (fp32 vs fp64 accumulation)")
    print("=" * 72)
    ok = True
    for n, k, seed, bound, label in CASES:
        x, w = make_inputs(n, k, seed, bound)
        spread = float((reference_fp32(x, w) - reference_fp64(x, w)).abs().max())
        print("  [%-28s] fp32-vs-fp64 oracle spread = %.3e" % (label, spread), flush=True)
        # The oracle must be far tighter than the tolerance it polices.
        ok = record(spread > 1e-3, kind="oracle", case=label, spread=spread) and ok
    return ok


def check_device_gemv():
    print("=" * 72)
    print("3. M=1 decode GEMV: tecoops.weight_stream_gemv vs CPU reference")
    print("=" * 72)
    ok = True
    for n, k, seed, bound, label in CASES:
        x, w = make_inputs(n, k, seed, bound)
        ref32 = reference_fp32(x, w)
        ref64 = reference_fp64(x, w)

        x_dev = x.to("sdaa")
        w_dev = w.to("sdaa")
        out_dev = torch.empty(n, dtype=torch.half, device="sdaa")

        result = tecoops.weight_stream_gemv(x_dev, w_dev, out_dev)
        torch.sdaa.synchronize()

        ok = record(result is not None, kind="return", case=label,
                    reason="weight_stream_gemv must be an out= op returning None",
                    returned=repr(result)) and ok
        if not compare(out_dev.cpu(), ref32, ref64, label, extra=dict(kind="gemv")):
            ok = False
    return ok


def check_linear_stream_routing():
    """M=1 must reach the GEMV kernel; everything else must use the GEMM fallback."""
    print("=" * 72)
    print("4. tecoops.linear_stream: M=1 fast path and documented non-M=1 fallback")
    print("=" * 72)
    ok = True

    # ---- 4a. M=1 fast path ----
    for n, k, seed, bound, label in CASES[:2] + CASES[2:3]:
        x, w = make_inputs(n, k, seed, bound)
        ref32 = reference_fp32(x, w)
        ref64 = reference_fp64(x, w)
        bias = torch.linspace(-0.5, 0.5, n, dtype=torch.float32).half()

        x_dev = x.to("sdaa")
        w_dev = w.to("sdaa")
        fast = tecoops.linear_stream(x_dev.unsqueeze(0), w_dev, None)
        torch.sdaa.synchronize()

        if not record(tuple(fast.shape) != (1, n), kind="shape", case=label,
                      reason="M=1 linear_stream must return [1, n]",
                      got=list(fast.shape)):
            ok = False
            continue
        if not compare(fast.cpu().reshape(n), ref32, ref64, "linear_stream " + label,
                       extra=dict(kind="linear_stream_m1")):
            ok = False

        # The wrapper and the raw op both launch tecoopsWeightStreamGemv with the
        # same pointers on the same stream, so a bit-identical result is the
        # necessary condition for "the M=1 path did not fall back".
        out_dev = torch.empty(n, dtype=torch.half, device="sdaa")
        tecoops.weight_stream_gemv(x_dev, w_dev, out_dev)
        torch.sdaa.synchronize()
        same = bool(torch.equal(fast.cpu().reshape(n).view(torch.int16),
                                out_dev.cpu().view(torch.int16)))
        print("  [%-28s] M=1 wrapper bit-identical to raw GEMV = %s"
              % (label, same), flush=True)
        if not record(not same, kind="fallback", case=label,
                      reason="M=1 wrapper did not reproduce the GEMV kernel bit-for-bit "
                             "(unexpected fallback to the vendor GEMM)",
                      bit_identical=same):
            ok = False

        # Bias on the fast path: the C++ adds bias to the fp16 GEMV row in fp16,
        # which is exactly reproducible here, so require bit equality.
        bias_dev = bias.to("sdaa")
        fast_b = tecoops.linear_stream(x_dev.unsqueeze(0), w_dev, bias_dev)
        torch.sdaa.synchronize()
        expected_half = fast + bias_dev
        bias_same = bool(torch.equal(fast_b.cpu().view(torch.int16),
                                     expected_half.cpu().view(torch.int16)))
        print("  [%-28s] M=1 wrapper + bias bit-identical to GEMV+bias = %s"
              % (label, bias_same), flush=True)
        if not record(not bias_same, kind="fallback_bias", case=label,
                      reason="bias was not applied as a plain fp16 add to the GEMV row"):
            ok = False
        # Reported numerically as well, with a cancellation-safe bound.
        ref_bias32 = ref32 + bias.float()
        ref_bias64 = ref64 + bias.float()
        if not compare(fast_b.cpu().reshape(n), ref_bias32, ref_bias64,
                       "linear_stream+bias " + label,
                       allowed=allowed_error(ref32, ref_bias32),
                       extra=dict(kind="linear_stream_m1_bias")):
            ok = False

    # ---- 4b. documented non-M=1 fallback ----
    # M>1 prefill rows, and M=1 with n % 256 != 0 or k % 16 != 0: all miss the
    # fast-path guard, so linear_stream must delegate to the vendor GEMM without
    # crashing and still be numerically correct.
    fallback_shapes = [
        (3, 256, 256, "prefill M=3 n=256 k=256"),
        (5, 512, 512, "prefill M=5 n=512 k=512"),
        (1, 128, 256, "M=1 but n=128 (n%256!=0)"),
        (1, 256, 250, "M=1 but k=250 (k%16!=0)"),
    ]
    for m, n, k, label in fallback_shapes:
        x, w = make_inputs(n, k, 77, 0.5)
        bias = torch.linspace(-0.5, 0.5, n, dtype=torch.float32).half()
        x_batch = x.unsqueeze(0).repeat(m, 1)  # [m, k]
        ref = torch.nn.functional.linear(x_batch.float(), w.float(), bias.float())
        ref64 = torch.nn.functional.linear(x_batch.double(), w.double(),
                                           bias.double()).float().reshape(-1)

        got = tecoops.linear_stream(x_batch.to("sdaa"), w.to("sdaa"), bias.to("sdaa"))
        torch.sdaa.synchronize()

        if not record(tuple(got.shape) != (m, n), kind="shape", case=label,
                      reason="fallback must preserve the [m, n] GEMM shape",
                      got=list(got.shape)):
            ok = False
            continue
        if not compare(got.cpu().reshape(-1), ref.reshape(-1), ref64, label,
                       extra=dict(kind="linear_stream_fallback", m=m)):
            ok = False

    return ok


def check_contract_rejection():
    """Contract-violating shapes must be refused, not silently computed."""
    print("=" * 72)
    print("5. fail-closed contract check (unsupported n/k must not be computed)")
    print("=" * 72)
    ok = True
    for n, k, label in ((100, 256, "n=100 (n%256!=0)"), (256, 250, "k=250 (k%16!=0)")):
        x_bad = torch.zeros(k, dtype=torch.half)
        w_bad = torch.zeros(n, k, dtype=torch.half)
        out_dev = torch.full((n,), float("nan"), dtype=torch.half, device="sdaa")

        raised = None
        try:
            tecoops.weight_stream_gemv(x_bad.to("sdaa"), w_bad.to("sdaa"), out_dev)
            torch.sdaa.synchronize()
        except Exception as exc:  # noqa: BLE001 - any refusal is acceptable
            raised = type(exc).__name__

        if raised is not None:
            print("  [%-28s] refused via %s  PASSED" % (label, raised), flush=True)
            record(False, kind="rejection", case=label, refused_by=raised)
            continue

        untouched = bool(torch.isnan(out_dev).all().item())
        print("  [%-28s] returned silently; output untouched = %s  %s"
              % (label, untouched, "PASSED" if untouched else "FAILED"), flush=True)
        if not record(not untouched, kind="rejection", case=label,
                      reason="unsupported shape was silently computed instead of refused",
                      untouched=untouched):
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true",
                        help="run only the oracle self-check; no device work")
    parser.add_argument("--output-json",
                        help="write the report here instead of a temporary directory")
    args = parser.parse_args()

    print("=" * 72)
    print("weight_stream_gemv / linear_stream python-API conformance test")
    print("=" * 72)

    ok = check_binding()
    ok = check_cpu_oracle() and ok

    if args.cpu_only:
        print("\n--cpu-only: device sections skipped", flush=True)
    elif not torch.sdaa.is_available():
        print("FATAL: torch.sdaa reports no available device", flush=True)
        record(True, kind="device", reason="no SDAA device available")
        ok = False
    else:
        import torch_sdaa  # noqa: F401  - registers the SDAA device
        ok = check_device_gemv() and ok
        ok = check_linear_stream_routing() and ok
        ok = check_contract_rejection() and ok

    report = dict(operator="weight_stream_gemv", passed=bool(ok and not FAILURES),
                  tol_ulps=TOL_ULPS, tol_abs_floor=TOL_ABS_FLOOR,
                  failures=FAILURES, results=REPORT)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output_json:
        path = os.path.abspath(os.path.expanduser(args.output_json))
        with open(path, "w") as handle:
            handle.write(serialized)
        print("\nreport written to %s" % path, flush=True)
    else:
        # A bare run (exactly as CI invokes it) writes outside nothing but a
        # private temporary directory.
        with tempfile.TemporaryDirectory(prefix="wsg-conformance-") as tmp:
            path = os.path.join(tmp, "weight_stream_gemv.json")
            with open(path, "w") as handle:
                handle.write(serialized)
            print("\nBEGIN weight_stream_gemv.json (temporary; use --output-json to retain)",
                  flush=True)
            print(serialized, end="", flush=True)
            print("END weight_stream_gemv.json", flush=True)

    print("=" * 72)
    if ok and not FAILURES:
        print("ALL PASSED")
        print("=" * 72)
        return 0
    print("SOME FAILED (%d failing check(s))" % len(FAILURES))
    for entry in FAILURES:
        print("  FAIL %s: %s" % (entry.get("case", entry.get("kind")),
                                 entry.get("reason", entry.get("name", ""))))
    print("=" * 72)
    return 1


if __name__ == "__main__":
    sys.exit(main())
