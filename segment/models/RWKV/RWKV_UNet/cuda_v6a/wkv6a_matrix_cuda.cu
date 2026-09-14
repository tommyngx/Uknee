#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <vector>


namespace {

constexpr int kThreads = 256;


template <typename scalar_t>
__global__ void matrix_scan_forward_kernel(
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ decay,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ y,
    float* __restrict__ states,
    int64_t batch,
    int64_t length,
    int64_t heads,
    int64_t head_dim) {
  const int64_t sequence_head = blockIdx.x;
  const int64_t batch_index = sequence_head / heads;
  const int64_t head_index = sequence_head % heads;
  if (batch_index >= batch) {
    return;
  }

  const int64_t matrix_size = head_dim * head_dim;
  for (int64_t t = 0; t < length; ++t) {
    const int64_t vector_base = ((batch_index * length + t) * heads + head_index) * head_dim;
    const int64_t state_base = vector_base * head_dim;
    const int64_t previous_base = state_base - heads * head_dim * head_dim;

    for (int64_t flat = threadIdx.x; flat < matrix_size; flat += blockDim.x) {
      const int64_t d = flat / head_dim;
      const int64_t e = flat - d * head_dim;
      const float previous = t == 0 ? 0.0f : states[previous_base + flat];
      states[state_base + flat] =
          static_cast<float>(decay[vector_base + d]) * previous +
          static_cast<float>(k[vector_base + d]) * static_cast<float>(v[vector_base + e]);
    }
    __syncthreads();

    for (int64_t e = threadIdx.x; e < head_dim; e += blockDim.x) {
      float value = 0.0f;
      for (int64_t d = 0; d < head_dim; ++d) {
        value += static_cast<float>(r[vector_base + d]) *
                 states[state_base + d * head_dim + e];
      }
      y[vector_base + e] = static_cast<scalar_t>(value);
    }
    __syncthreads();
  }
}


template <typename scalar_t>
__global__ void matrix_scan_backward_kernel(
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ decay,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float* __restrict__ states,
    const scalar_t* __restrict__ grad_y,
    scalar_t* __restrict__ grad_r,
    scalar_t* __restrict__ grad_decay,
    scalar_t* __restrict__ grad_k,
    scalar_t* __restrict__ grad_v,
    float* __restrict__ state_grad,
    int64_t batch,
    int64_t length,
    int64_t heads,
    int64_t head_dim) {
  const int64_t sequence_head = blockIdx.x;
  const int64_t batch_index = sequence_head / heads;
  const int64_t head_index = sequence_head % heads;
  if (batch_index >= batch) {
    return;
  }

  const int64_t matrix_size = head_dim * head_dim;
  const int64_t state_grad_base = sequence_head * matrix_size;

  for (int64_t t = length - 1; t >= 0; --t) {
    const int64_t vector_base = ((batch_index * length + t) * heads + head_index) * head_dim;
    const int64_t state_base = vector_base * head_dim;
    const int64_t previous_base = state_base - heads * head_dim * head_dim;

    for (int64_t flat = threadIdx.x; flat < matrix_size; flat += blockDim.x) {
      const int64_t d = flat / head_dim;
      const int64_t e = flat - d * head_dim;
      state_grad[state_grad_base + flat] +=
          static_cast<float>(r[vector_base + d]) *
          static_cast<float>(grad_y[vector_base + e]);
    }
    __syncthreads();

    for (int64_t component = threadIdx.x; component < head_dim; component += blockDim.x) {
      float r_value = 0.0f;
      float decay_value = 0.0f;
      float k_value = 0.0f;
      float v_value = 0.0f;
      for (int64_t e = 0; e < head_dim; ++e) {
        const int64_t flat = component * head_dim + e;
        const float adjoint = state_grad[state_grad_base + flat];
        const float previous = t == 0 ? 0.0f : states[previous_base + flat];
        r_value += static_cast<float>(grad_y[vector_base + e]) * states[state_base + flat];
        decay_value += adjoint * previous;
        k_value += adjoint * static_cast<float>(v[vector_base + e]);
        v_value += state_grad[state_grad_base + e * head_dim + component] *
                   static_cast<float>(k[vector_base + e]);
      }
      grad_r[vector_base + component] = static_cast<scalar_t>(r_value);
      grad_decay[vector_base + component] = static_cast<scalar_t>(decay_value);
      grad_k[vector_base + component] = static_cast<scalar_t>(k_value);
      grad_v[vector_base + component] = static_cast<scalar_t>(v_value);
    }
    __syncthreads();

    for (int64_t flat = threadIdx.x; flat < matrix_size; flat += blockDim.x) {
      const int64_t d = flat / head_dim;
      state_grad[state_grad_base + flat] *= static_cast<float>(decay[vector_base + d]);
    }
    __syncthreads();
  }
}

}  // namespace


std::vector<torch::Tensor> wkv6a_cuda_forward(
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& k,
    const torch::Tensor& v) {
  const c10::cuda::CUDAGuard device_guard(r.device());
  const auto batch = r.size(0);
  const auto length = r.size(1);
  const auto heads = r.size(2);
  const auto head_dim = r.size(3);
  auto y = torch::empty_like(r);
  auto states = torch::empty(
      {batch, length, heads, head_dim, head_dim},
      r.options().dtype(torch::kFloat32));
  const dim3 blocks(batch * heads);
  const auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kFloat16, torch::kBFloat16, r.scalar_type(), "wkv6a_matrix_forward", [&] {
        matrix_scan_forward_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            r.data_ptr<scalar_t>(), decay.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), states.data_ptr<float>(),
            batch, length, heads, head_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, states};
}


std::vector<torch::Tensor> wkv6a_cuda_backward(
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& states,
    const torch::Tensor& grad_y) {
  const c10::cuda::CUDAGuard device_guard(r.device());
  const auto batch = r.size(0);
  const auto length = r.size(1);
  const auto heads = r.size(2);
  const auto head_dim = r.size(3);
  auto grad_r = torch::empty_like(r);
  auto grad_decay = torch::empty_like(decay);
  auto grad_k = torch::empty_like(k);
  auto grad_v = torch::empty_like(v);
  auto state_grad = torch::zeros(
      {batch, heads, head_dim, head_dim},
      r.options().dtype(torch::kFloat32));
  const dim3 blocks(batch * heads);
  const auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kFloat16, torch::kBFloat16, r.scalar_type(), "wkv6a_matrix_backward", [&] {
        matrix_scan_backward_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            r.data_ptr<scalar_t>(), decay.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(), states.data_ptr<float>(), grad_y.data_ptr<scalar_t>(),
            grad_r.data_ptr<scalar_t>(), grad_decay.data_ptr<scalar_t>(),
            grad_k.data_ptr<scalar_t>(), grad_v.data_ptr<scalar_t>(),
            state_grad.data_ptr<float>(), batch, length, heads, head_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_r, grad_decay, grad_k, grad_v};
}
