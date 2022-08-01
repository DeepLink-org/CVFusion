#include <iostream>

#include "api.h"

using namespace ir;

namespace Resize {

ir::TensorVarPtr Nearest(const std::vector<ir::ExprPtr> &shape,
                         ir::Array<ir::IterVar> iter_vars,
                         ir::TensorVarPtr input,
                         const std::string &name = "ResizeNearest") {
  ELENA_ASSERT(shape.size() == input->shape->size(), "ResizeNearest");

  std::vector<ir::ExprPtr> scale_shape{api::constant<uint64_t>(2)};
  auto iter_scale = api::construct_indices(scale_shape);
  auto if_zero = api::logical::eq(iter_scale[0], api::constant<int>(0));

  auto scale =
      api::compute(scale_shape, iter_scale,
                   api::if_then_else(
                       if_zero,
                       cast(input->shape->element[0], ir::ScalarType::Float32) /
                           cast(shape[0], ir::ScalarType::Float32),
                       cast(input->shape->element[1], ir::ScalarType::Float32) /
                           cast(shape[1], ir::ScalarType::Float32)),
                   "scale");
  auto bound_h = input->shape->element[0] - api::constant<int>(1);
  auto bound_w = input->shape->element[1] - api::constant<int>(1);

  return api::compute(
      shape, iter_vars,
      (*input)(
          min(floor(iter_vars[0] * (*scale)(api::constant<int>(0))), bound_h),
          min(floor(iter_vars[1] * (*scale)(api::constant<int>(1))), bound_w),
          iter_vars[2]),
      name);
}

ir::TensorVarPtr Bilinear(const std::vector<ir::ExprPtr> &shape,
                          ir::Array<ir::IterVar> iter_vars,
                          ir::TensorVarPtr input, ir::TensorVarPtr cubfh,
                          ir::TensorVarPtr cubfw, ir::TensorVarPtr inth,
                          ir::TensorVarPtr intw,
                          const std::string &name = "ResizeBilinear") {
  ELENA_ASSERT(shape.size() == input->shape->size(), "ResizeBilinear");

  auto zero = api::constant<uint64_t>(0);
  auto one = api::constant<uint64_t>(1);
  return api::compute(
      shape, iter_vars,
      ((*cubfh)(zero, iter_vars[0]) * (*cubfw)(zero, iter_vars[1]) *
           (*input)((*inth)(zero, iter_vars[0]), (*intw)(zero, iter_vars[1]),
                    iter_vars[2]) +
       (*cubfh)(one, iter_vars[0]) * (*cubfw)(zero, iter_vars[1]) *
           (*input)((*inth)(one, iter_vars[0]), (*intw)(zero, iter_vars[1]),
                    iter_vars[2]) +
       (*cubfh)(zero, iter_vars[0]) * (*cubfw)(one, iter_vars[1]) *
           (*input)((*inth)(zero, iter_vars[0]), (*intw)(one, iter_vars[1]),
                    iter_vars[2]) +
       (*cubfh)(one, iter_vars[0]) * (*cubfw)(one, iter_vars[1]) *
           (*input)((*inth)(one, iter_vars[0]), (*intw)(one, iter_vars[1]),
                    iter_vars[2]) +
       api::constant<int>(2097152)) /
          api::constant<int>(4194304),  //  (... + 1 << (22 - 1)) >> 22
      name);
}

ir::TensorVarPtr BilinearFloat(const std::vector<ir::ExprPtr> &shape,
                               ir::Array<ir::IterVar> iter_vars,
                               ir::TensorVarPtr input, ir::TensorVarPtr cubfh,
                               ir::TensorVarPtr cubfw, ir::TensorVarPtr inth,
                               ir::TensorVarPtr intw,
                               const std::string &name = "ResizeBilinear") {
  ELENA_ASSERT(shape.size() == input->shape->size(), "ResizeBilinear");

  auto zero = api::constant<uint64_t>(0);
  auto one = api::constant<uint64_t>(1);
  return api::compute(
      shape, iter_vars,
      ((*cubfh)(zero, iter_vars[0]) * (*cubfw)(zero, iter_vars[1]) *
           (*input)((*inth)(zero, iter_vars[0]), (*intw)(zero, iter_vars[1]),
                    iter_vars[2]) +
       (*cubfh)(one, iter_vars[0]) * (*cubfw)(zero, iter_vars[1]) *
           (*input)((*inth)(one, iter_vars[0]), (*intw)(zero, iter_vars[1]),
                    iter_vars[2]) +
       (*cubfh)(zero, iter_vars[0]) * (*cubfw)(one, iter_vars[1]) *
           (*input)((*inth)(zero, iter_vars[0]), (*intw)(one, iter_vars[1]),
                    iter_vars[2]) +
       (*cubfh)(one, iter_vars[0]) * (*cubfw)(one, iter_vars[1]) *
           (*input)((*inth)(one, iter_vars[0]), (*intw)(one, iter_vars[1]),
                    iter_vars[2])),
      name);
}

ir::TensorVarPtr BilinearCUDA(const std::vector<ir::ExprPtr> &shape,
                          ir::Array<ir::IterVar> iter_vars,
                          ir::TensorVarPtr input, 
                          ir::ExprPtr src_h, ir::ExprPtr src_w,
                          const std::string &name = "ResizeBilinear") {
  ELENA_ASSERT(shape.size() == input->shape->size(), "ResizeBilinear");

  auto zero = api::constant<uint64_t>(0);
  auto one = api::constant<uint64_t>(1);
  auto two = api::constant<uint64_t>(2);

  auto inth = api::compute({two}, api::construct_indices({two}), api::constant<int32_t>(0), "inth");
  auto intw = api::compute({two}, api::construct_indices({two}), api::constant<int32_t>(0), "intw");
  auto cubh = api::compute({two}, api::construct_indices({two}), api::constant<int16_t>(0), "cubh");
  auto cubw = api::compute({two}, api::construct_indices({two}), api::constant<int16_t>(0), "cubw");

  auto resize_h = shape[0];
  auto resize_w = shape[1];

  // hack evaluate
  std::vector<ir::ExprPtr> preprocess_args{src_h, resize_h, src_w, resize_w,
                                          (*cubh)(zero), (*inth)(zero), (*cubw)(zero), (*intw)(zero)};
  auto preprocess_call = std::make_shared<ir::Call>(
            ir::CallFunction::bilinear_resize_preprocess,
            std::make_shared<ir::Array<ir::Expr>>(preprocess_args),
            ir::ScalarType::Int32);
  inth  = api::compute({two}, api::construct_indices({one}), {}, preprocess_call, "inth");

  return api::compute(
      shape, iter_vars,
      ((*cubh)(zero) * (*cubw)(zero) *
           (*input)((*inth)(zero), (*intw)(zero),
                    iter_vars[2]) +
       (*cubh)(one) * (*cubw)(zero) *
           (*input)((*inth)(one), (*intw)(zero),
                    iter_vars[2]) +
       (*cubh)(zero) * (*cubw)(one) *
           (*input)((*inth)(zero), (*intw)(one),
                    iter_vars[2]) +
       (*cubh)(one) * (*cubw)(one) *
           (*input)((*inth)(one), (*intw)(one),
                    iter_vars[2]) +
       api::constant<int>(2097152)) /
          api::constant<int>(4194304),  //  (... + 1 << (22 - 1)) >> 22
      name);
}

ir::TensorVarPtr BilinearFloatCUDA(const std::vector<ir::ExprPtr> &shape,
                          ir::Array<ir::IterVar> iter_vars,
                          ir::TensorVarPtr input, 
                          ir::ExprPtr src_h, ir::ExprPtr src_w,
                          const std::string &name = "ResizeBilinear") {
  ELENA_ASSERT(shape.size() == input->shape->size(), "ResizeBilinear");

  auto zero = api::constant<uint64_t>(0);
  auto one = api::constant<uint64_t>(1);
  auto two = api::constant<uint64_t>(2);

  auto inth = api::compute({two}, api::construct_indices({two}), api::constant<int32_t>(0), "inth");
  auto intw = api::compute({two}, api::construct_indices({two}), api::constant<int32_t>(0), "intw");
  auto cubfh = api::compute({two}, api::construct_indices({two}), api::constant<float>(0), "cubh");
  auto cubfw = api::compute({two}, api::construct_indices({two}), api::constant<float>(0), "cubw");

  auto resize_h = shape[0];
  auto resize_w = shape[1];

  // hack evaluate
  std::vector<ir::ExprPtr> preprocess_args{src_h, resize_h, src_w, resize_w,
                                          (*cubfh)(zero), (*inth)(zero), (*cubfw)(zero), (*intw)(zero)};
  auto preprocess_call = std::make_shared<ir::Call>(
            ir::CallFunction::bilinear_float_resize_preprocess,
            std::make_shared<ir::Array<ir::Expr>>(preprocess_args),
            ir::ScalarType::Int32);
  inth  = api::compute({two}, api::construct_indices({one}), {}, preprocess_call, "inth");

  return api::compute(
      shape, iter_vars,
      ((*cubfh)(zero) * (*cubfw)(zero) *
           (*input)((*inth)(zero), (*intw)(zero),
                    iter_vars[2]) +
       (*cubfh)(one) * (*cubfw)(zero) *
           (*input)((*inth)(one), (*intw)(zero),
                    iter_vars[2]) +
       (*cubfh)(zero) * (*cubfw)(one) *
           (*input)((*inth)(zero), (*intw)(one),
                    iter_vars[2]) +
       (*cubfh)(one) * (*cubfw)(one) *
           (*input)((*inth)(one), (*intw)(one),
                    iter_vars[2])), 
      name);
}

}  // namespace Resize
