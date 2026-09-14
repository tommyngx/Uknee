#include <torch/extension.h>

#include <vector>


std::vector<torch::Tensor> wkv6a_cuda_forward(
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& k,
    const torch::Tensor& v);

std::vector<torch::Tensor> wkv6a_cuda_backward(
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& states,
    const torch::Tensor& grad_y);


namespace {

void check_inputs(
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& k,
    const torch::Tensor& v) {
  TORCH_CHECK(r.is_cuda(), "r must be a CUDA tensor");
  TORCH_CHECK(r.dim() == 4, "expected r with shape [B_seq, T, H, D]");
  TORCH_CHECK(r.is_contiguous(), "r must be contiguous");
  TORCH_CHECK(r.scalar_type() == torch::kFloat32 ||
                  r.scalar_type() == torch::kFloat16 ||
                  r.scalar_type() == torch::kBFloat16,
              "supported dtypes are float32, float16, and bfloat16");

  for (const auto& item : {decay, k, v}) {
    TORCH_CHECK(item.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(item.is_contiguous(), "all inputs must be contiguous");
    TORCH_CHECK(item.sizes() == r.sizes(), "all inputs must have identical shapes");
    TORCH_CHECK(item.scalar_type() == r.scalar_type(), "all inputs must have one dtype");
    TORCH_CHECK(item.device() == r.device(), "all inputs must be on one CUDA device");
  }
  TORCH_CHECK(r.size(1) > 0, "sequence length T must be positive");
  TORCH_CHECK(r.size(2) > 0, "head count H must be positive");
  TORCH_CHECK(r.size(3) > 0, "head dimension D must be positive");
}

}  // namespace


std::vector<torch::Tensor> forward(
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& k,
    const torch::Tensor& v) {
  check_inputs(r, decay, k, v);
  return wkv6a_cuda_forward(r, decay, k, v);
}


std::vector<torch::Tensor> backward(
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& states,
    const torch::Tensor& grad_y) {
  check_inputs(r, decay, k, v);
  TORCH_CHECK(states.is_cuda() && states.scalar_type() == torch::kFloat32,
              "saved states must be CUDA float32");
  TORCH_CHECK(states.is_contiguous(), "saved states must be contiguous");
  TORCH_CHECK(states.dim() == 5, "saved states must have shape [B_seq, T, H, D, D]");
  TORCH_CHECK(states.size(0) == r.size(0) && states.size(1) == r.size(1) &&
                  states.size(2) == r.size(2) && states.size(3) == r.size(3) &&
                  states.size(4) == r.size(3),
              "saved-state shape does not match the inputs");
  TORCH_CHECK(grad_y.is_cuda() && grad_y.is_contiguous(),
              "grad_y must be a contiguous CUDA tensor");
  TORCH_CHECK(grad_y.sizes() == r.sizes() && grad_y.scalar_type() == r.scalar_type(),
              "grad_y shape and dtype must match r");
  return wkv6a_cuda_backward(r, decay, k, v, states, grad_y);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &forward, "RWKV V6a matrix-state forward (CUDA)");
  module.def("backward", &backward, "RWKV V6a matrix-state backward (CUDA)");
}
