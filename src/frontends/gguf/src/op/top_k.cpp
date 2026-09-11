// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

<<<<<<< HEAD
#include <memory>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
=======
#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/squeeze.hpp"
>>>>>>> 891ebb895f6f89baa30a675bce32edf45c800f06
#include "openvino/op/topk.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

<<<<<<< HEAD
using namespace ov::op;

// ggml_top_k(a, k): the top-k indices along the last axis (descending). Used by the native
// .gguf builder for MoE expert selection. Wrap the index output in a Convert so downstream
// consumers see a single-output source node.
OutputVector translate_top_k(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto a = context.get_input(0);
    int64_t k = context.get_output_shape().to_shape().back();
    auto k_node = v0::Constant::create(element::i64, Shape{}, {k});
    auto topk = std::make_shared<v11::TopK>(a,
                                            k_node,
                                            -1,
                                            v11::TopK::Mode::MAX,
                                            v11::TopK::SortType::SORT_VALUES,
                                            element::i32);
    auto idx = std::make_shared<v0::Convert>(topk->output(1), element::i32);
    return rename_outputs_with_suffix({idx}, context.get_name());
=======
// ggml_top_k(a, k): the indices of the k largest values along ne[0] (the OV last axis),
// ordered by descending value, as i32. k is the extent of that axis on the output.
OutputVector translate_top_k(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);

    // k is the output's last-axis extent. Prefer the static value, but fall back to reading it off
    // the output shape at runtime so a dynamic extent converts instead of throwing (ARGSORT derives
    // its k dynamically for the same reason).
    const auto& out_ps = context.get_output_shape();
    const auto rank = out_ps.rank();
    const int64_t axis = rank.is_static() ? rank.get_length() - 1 : -1;
    ov::Output<ov::Node> k_node;
    if (rank.is_static() && out_ps[rank.get_length() - 1].is_static()) {
        k_node =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {out_ps[rank.get_length() - 1].get_length()});
    } else {
        k_node = std::make_shared<ov::op::v0::Squeeze>(
            get_dimensions(input, {static_cast<int>(rank.is_static() ? rank.get_length() - 1 : 3)}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    }

    auto indices = make_topk_indices(input,
                                     k_node,
                                     axis,
                                     ov::op::v11::TopK::Mode::MAX,
                                     context.get_attribute<ov::element::Type>("output_type"));

    return rename_outputs_with_suffix({std::move(indices)}, context.get_name());
>>>>>>> 891ebb895f6f89baa30a675bce32edf45c800f06
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
