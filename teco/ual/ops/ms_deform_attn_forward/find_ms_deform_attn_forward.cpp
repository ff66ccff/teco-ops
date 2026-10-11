// BSD 3-Clause License
//
// Copyright (c) 2024, Tecorigin Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "ual/ops/ms_deform_attn_forward/find_ms_deform_attn_forward.h"

namespace tecoops {
namespace ual {
namespace ops {

// Binds the ms_deform_attn_forward implementation once per call, outside the
// kernel (AGENTS.md rule 6: no hot-path selector, no getenv).
//
// The span-merge kernel loads every corner block of one (level, point) with ONE
// blocked transfer into T value_record[128].  The widest such span is 2*width
// pixels and width <= value_len for every level, so
//
//     2 * value_len * num_heads * head_dim <= 128
//
// is a conservative, data independent guarantee that the record never overflows
// (it implies the tighter per-level form 2*width*num_heads*head_dim <= 128 that
// the CPU bit-exact model uses).  Shapes that fail it keep the delivered kernel
// verbatim, so the specialization is fail-closed on the shape it was validated
// for.
int findMsDeformAttnForwardBranch(
    const args::MsDeformAttnForwardPatchArgs *arg) {
    const args::MsDeformAttnForwardArgs *a = arg->atargs;
    const bool span_fits =
        a != nullptr && a->value_len > 0 && a->num_heads > 0 && a->head_dim > 0 &&
        2LL * a->value_len * a->num_heads * a->head_dim <= 128;
    if (arg->data_type == common::UALDataType::UAL_DTYPE_FLOAT) {
        return span_fits ? 2 : 0;
    }
    if (arg->data_type == common::UALDataType::UAL_DTYPE_HALF) {
        return span_fits ? 3 : 1;
    }
    return -1;
}

}  // namespace ops
}  // namespace ual
}  // namespace tecoops
