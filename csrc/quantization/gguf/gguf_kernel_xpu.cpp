// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// SYCL dequantization kernels for GGUF quantization types.
//
// Ported from llama.cpp ggml-sycl.cpp:
// https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-sycl/ggml-sycl.cpp
//
// Supported types:
//   Q8_0  – 8-bit symmetric (standard)
//   Q4_K  – 4-bit k-quant with super-block scales/mins (QK_K = 256)

#include <sycl/sycl.hpp>
#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>

// ---------------------------------------------------------------------------
// Data structures (mirrored from ggml-common.h)
// ---------------------------------------------------------------------------

#define QK_K 256
#define QK8_0 32
#define QR8_0 1
#define QK4_K 256
#define QR4_K 2
#define K_SCALE_SIZE 12
#define SYCL_DEQUANTIZE_BLOCK_SIZE 256

typedef struct {
  sycl::half d;
  int8_t qs[QK8_0];
} block_q8_0;

typedef struct {
  sycl::half2 dm;                 // super-block scale/min
  uint8_t scales[3 * QK_K / 64];  // 6-bit scales/mins (12 bytes)
  uint8_t qs[QK_K / 2];           // 4-bit quants
} block_q4_K;

// ---------------------------------------------------------------------------
// Helper: extract 6-bit scale and min from the packed scales array
// (identical logic to get_scale_min_k4 in dequantize.cuh)
// ---------------------------------------------------------------------------
static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d,
                                    uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// ---------------------------------------------------------------------------
// Q8_0 dequantization kernel
//
// Each work-item handles two consecutive output elements.
// Layout of a block_q8_0 (34 bytes):
//   [d: half][qs[0]…qs[31]: int8_t]
// Dequantized value: qs[i] * d
// ---------------------------------------------------------------------------
template <typename dst_t>
static void dequantize_block_q8_0_sycl(const void* __restrict__ vx,
                                       dst_t* __restrict__ vy, int k,
                                       sycl::nd_item<1> item) {
  // Two output elements per work-item (matches CUDA dequantize_block)
  const int i = 2 * static_cast<int>(item.get_global_id(0));
  if (i >= k) return;

  const block_q8_0* x = reinterpret_cast<const block_q8_0*>(vx);
  const int ib = i / QK8_0;   // block index
  const int iqs = i % QK8_0;  // offset within block

  const float d =
      sycl::vec<sycl::half, 1>{x[ib].d}
          .template convert<float, sycl::rounding_mode::automatic>()[0];

  vy[ib * QK8_0 + iqs + 0] = static_cast<dst_t>(x[ib].qs[iqs + 0] * d);
  vy[ib * QK8_0 + iqs + 1] = static_cast<dst_t>(x[ib].qs[iqs + 1] * d);
}

// ---------------------------------------------------------------------------
// Q4_K dequantization kernel
//
// One work-group of 32 threads per super-block (256 output values).
// Layout of a block_q4_K (144 bytes total for QK_K=256):
//   [dm: half2][scales[12]: uint8_t][qs[128]: uint8_t]
// Dequantized value:
//   d_all = dm.x,  d_min = dm.y
//   For sub-block pair (il, ir):
//     d1 = d_all * sc1, m1 = d_min * min1
//     d2 = d_all * sc2, m2 = d_min * min2
//     y[+  0] = d1 * (qs & 0xF) - m1
//     y[+ 32] = d2 * (qs >>  4) - m2
// ---------------------------------------------------------------------------
template <typename dst_t>
static void dequantize_block_q4_K_sycl(const void* __restrict__ vx,
                                       dst_t* __restrict__ yy,
                                       sycl::nd_item<1> item) {
  const block_q4_K* x = reinterpret_cast<const block_q4_K*>(vx);

  // Super-block index = work-group index
  const int i = static_cast<int>(item.get_group(0));
  // Local thread id (0…31)
  const int tid = static_cast<int>(item.get_local_id(0));
  const int il = tid / 8;  // 0…3  (4 sub-groups of 8)
  const int ir = tid % 8;  // 0…7
  const int is = 2 * il;   // scale pair index
  constexpr int n = 4;     // elements per thread per half-block

  dst_t* y = yy + i * QK_K + 64 * il + n * ir;

  // Decode super-block scale and min
  const float dall =
      sycl::vec<sycl::half, 1>{x[i].dm.x()}
          .template convert<float, sycl::rounding_mode::automatic>()[0];
  const float dmin =
      sycl::vec<sycl::half, 1>{x[i].dm.y()}
          .template convert<float, sycl::rounding_mode::automatic>()[0];

  const uint8_t* q = x[i].qs + 32 * il + n * ir;

  uint8_t sc, m;

  get_scale_min_k4(is + 0, x[i].scales, sc, m);
  const float d1 = dall * sc;
  const float m1 = dmin * m;

  get_scale_min_k4(is + 1, x[i].scales, sc, m);
  const float d2 = dall * sc;
  const float m2 = dmin * m;

  for (int l = 0; l < n; ++l) {
    y[l + 0] = static_cast<dst_t>(d1 * static_cast<float>(q[l] & 0x0F) - m1);
    y[l + 32] = static_cast<dst_t>(d2 * static_cast<float>(q[l] >> 4) - m2);
  }
}

// ---------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------

template <typename dst_t>
static void launch_dequantize_q8_0(const void* vx, dst_t* vy, int k,
                                   sycl::queue& queue) {
  const int num_groups =
      (k / 2 + SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / SYCL_DEQUANTIZE_BLOCK_SIZE;
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(num_groups * SYCL_DEQUANTIZE_BLOCK_SIZE),
                        sycl::range<1>(SYCL_DEQUANTIZE_BLOCK_SIZE)),
      [=](sycl::nd_item<1> item) {
        dequantize_block_q8_0_sycl<dst_t>(vx, vy, k, item);
      });
}

template <typename dst_t>
static void launch_dequantize_q4_K(const void* vx, dst_t* vy, int k,
                                   sycl::queue& queue) {
  const int nb = k / QK_K;  // number of super-blocks
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(nb * 32), sycl::range<1>(32)),
      [=](sycl::nd_item<1> item) {
        dequantize_block_q4_K_sycl<dst_t>(vx, vy, item);
      });
}

// ---------------------------------------------------------------------------
// Public PyTorch op: ggml_dequantize_xpu
//
// W         – quantized weight tensor (uint8, on XPU)
// quant_type – GGMLQuantizationType integer value (8 = Q8_0, 12 = Q4_K)
// m, n       – output shape [m, n]
// dtype      – output dtype (float16, bfloat16, or float32)
// ---------------------------------------------------------------------------
torch::Tensor ggml_dequantize_xpu(torch::Tensor W, int64_t quant_type,
                                  int64_t m, int64_t n,
                                  std::optional<at::ScalarType> dtype) {
  const at::xpu::XPUStream stream = at::xpu::getCurrentXPUStream();
  sycl::queue& queue = stream.queue();

  auto dtype_ = dtype.value_or(torch::kFloat16);
  auto options = torch::TensorOptions().dtype(dtype_).device(W.device());
  at::Tensor DW = torch::empty({m, n}, options);

  const int64_t k = m * n;  // total number of dequantized elements

  switch (quant_type) {
    case 8:  // Q8_0
      if (dtype_ == torch::kFloat16) {
        launch_dequantize_q8_0<sycl::half>(
            W.data_ptr(), reinterpret_cast<sycl::half*>(DW.data_ptr()), k,
            queue);
      } else if (dtype_ == torch::kBFloat16) {
        launch_dequantize_q8_0<sycl::ext::oneapi::bfloat16>(
            W.data_ptr(),
            reinterpret_cast<sycl::ext::oneapi::bfloat16*>(DW.data_ptr()), k,
            queue);
      } else {
        launch_dequantize_q8_0<float>(
            W.data_ptr(), reinterpret_cast<float*>(DW.data_ptr()), k, queue);
      }
      break;

    case 12:  // Q4_K
      if (dtype_ == torch::kFloat16) {
        launch_dequantize_q4_K<sycl::half>(
            W.data_ptr(), reinterpret_cast<sycl::half*>(DW.data_ptr()), k,
            queue);
      } else if (dtype_ == torch::kBFloat16) {
        launch_dequantize_q4_K<sycl::ext::oneapi::bfloat16>(
            W.data_ptr(),
            reinterpret_cast<sycl::ext::oneapi::bfloat16*>(DW.data_ptr()), k,
            queue);
      } else {
        launch_dequantize_q4_K<float>(
            W.data_ptr(), reinterpret_cast<float*>(DW.data_ptr()), k, queue);
      }
      break;

    default:
      TORCH_CHECK(false, "ggml_dequantize_xpu: unsupported quant_type ",
                  quant_type, " (supported: 8=Q8_0, 12=Q4_K)");
  }

  return DW;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_dequantize_xpu", &ggml_dequantize_xpu,
        "GGUF dequantization for Q8_0 and Q4_K on Intel XPU (SYCL)");
}
