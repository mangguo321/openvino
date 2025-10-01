#include "xattention.hpp"

#include <algorithm>
#include <chrono>
#include <common/z_magic.hpp>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <ios>
#include <limits>
#include <numeric>
#include <vector>

#include "nodes/kernels/x64/brgemm_kernel.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>

#    include "common.hpp"
#endif
#include "softmax_kernel.hpp"
#include "transpose_kernel.hpp"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

using namespace ov::intel_cpu;
namespace ov::Extensions::Cpu::XARCH {

namespace ref {
void softmax(float* a, int len) {
    float max = *std::max_element(a, a + len);
    float sum = 0.0F;
    for (int i = 0; i < len; i++) {
        a[i] = ::exp(a[i] - max);
        sum += a[i];
    }
    float scale = 1.0F / sum;
    for (int i = 0; i < len; i++) {
        a[i] *= scale;
    }
}

template <typename D>
float dot_product(const D* a, const D* b, int len, int stride_b = 1) {
    float result = 0;
    if (stride_b == 1) {
        for (int i = 0; i < len; i++) {
            result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        }
    } else {
        for (int i = 0; i < len; i++) {
            result += static_cast<float>(a[i]) * static_cast<float>(b[i * stride_b]);
        }
    }
    return result;
}

void sum_blocks_ref(const float* a,
                    size_t M,
                    size_t N,
                    size_t a_stride,
                    float* dst,
                    size_t out_stride,
                    size_t num_per_block) {
    size_t block_num = div_up(M, num_per_block);
    for (size_t row = 0; row < block_num; row++) {
        for (size_t col = 0; col < div_up(N, num_per_block); col++) {
            float value = 0.0f;
            for (size_t i = 0; i < num_per_block; i++) {
                for (size_t j = 0; j < num_per_block; j++) {
                    auto r_idx = row * num_per_block + i;
                    auto c_idx = col * num_per_block + j;
                    if (r_idx < M && c_idx < N) {
                        auto in = a[r_idx * a_stride + c_idx];
                        value += in;
                    }
                }
            }
            dst[row * out_stride + col] = value;
        }
    }
}

}  // namespace ref

#if defined(HAVE_AVX512F)
template <typename T>
inline void transpose_16x16_kernel(float* _dst, T** src, size_t dst_stride) {
    auto* dst = reinterpret_cast<uint32_t*>(_dst);
    __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
    r0 = _mm512_castps_si512(mm512_uni_loadu_ps(src[0]));
    r1 = _mm512_castps_si512(mm512_uni_loadu_ps(src[1]));
    r2 = _mm512_castps_si512(mm512_uni_loadu_ps(src[2]));
    r3 = _mm512_castps_si512(mm512_uni_loadu_ps(src[3]));
    r4 = _mm512_castps_si512(mm512_uni_loadu_ps(src[4]));
    r5 = _mm512_castps_si512(mm512_uni_loadu_ps(src[5]));
    r6 = _mm512_castps_si512(mm512_uni_loadu_ps(src[6]));
    r7 = _mm512_castps_si512(mm512_uni_loadu_ps(src[7]));
    r8 = _mm512_castps_si512(mm512_uni_loadu_ps(src[8]));
    r9 = _mm512_castps_si512(mm512_uni_loadu_ps(src[9]));
    ra = _mm512_castps_si512(mm512_uni_loadu_ps(src[10]));
    rb = _mm512_castps_si512(mm512_uni_loadu_ps(src[11]));
    rc = _mm512_castps_si512(mm512_uni_loadu_ps(src[12]));
    rd = _mm512_castps_si512(mm512_uni_loadu_ps(src[13]));
    re = _mm512_castps_si512(mm512_uni_loadu_ps(src[14]));
    rf = _mm512_castps_si512(mm512_uni_loadu_ps(src[15]));

    transpose_m512i_16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf);

    _mm512_storeu_si512(dst, r0);
    _mm512_storeu_si512(dst + dst_stride, r1);
    _mm512_storeu_si512(dst + 2 * dst_stride, r2);
    _mm512_storeu_si512(dst + 3 * dst_stride, r3);
    _mm512_storeu_si512(dst + 4 * dst_stride, r4);
    _mm512_storeu_si512(dst + 5 * dst_stride, r5);
    _mm512_storeu_si512(dst + 6 * dst_stride, r6);
    _mm512_storeu_si512(dst + 7 * dst_stride, r7);
    _mm512_storeu_si512(dst + 8 * dst_stride, r8);
    _mm512_storeu_si512(dst + 9 * dst_stride, r9);
    _mm512_storeu_si512(dst + 10 * dst_stride, ra);
    _mm512_storeu_si512(dst + 11 * dst_stride, rb);
    _mm512_storeu_si512(dst + 12 * dst_stride, rc);
    _mm512_storeu_si512(dst + 13 * dst_stride, rd);
    _mm512_storeu_si512(dst + 14 * dst_stride, re);
    _mm512_storeu_si512(dst + 15 * dst_stride, rf);
}

inline void transpose_16x16_kernel(uint32_t* dst, uint32_t** src, size_t dst_stride) {
    __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
    r0 = _mm512_loadu_si512(src[0]);
    r1 = _mm512_loadu_si512(src[1]);
    r2 = _mm512_loadu_si512(src[2]);
    r3 = _mm512_loadu_si512(src[3]);
    r4 = _mm512_loadu_si512(src[4]);
    r5 = _mm512_loadu_si512(src[5]);
    r6 = _mm512_loadu_si512(src[6]);
    r7 = _mm512_loadu_si512(src[7]);
    r8 = _mm512_loadu_si512(src[8]);
    r9 = _mm512_loadu_si512(src[9]);
    ra = _mm512_loadu_si512(src[10]);
    rb = _mm512_loadu_si512(src[11]);
    rc = _mm512_loadu_si512(src[12]);
    rd = _mm512_loadu_si512(src[13]);
    re = _mm512_loadu_si512(src[14]);
    rf = _mm512_loadu_si512(src[15]);

    transpose_m512i_16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf);

    _mm512_storeu_si512(dst, r0);
    _mm512_storeu_si512(dst + dst_stride, r1);
    _mm512_storeu_si512(dst + 2 * dst_stride, r2);
    _mm512_storeu_si512(dst + 3 * dst_stride, r3);
    _mm512_storeu_si512(dst + 4 * dst_stride, r4);
    _mm512_storeu_si512(dst + 5 * dst_stride, r5);
    _mm512_storeu_si512(dst + 6 * dst_stride, r6);
    _mm512_storeu_si512(dst + 7 * dst_stride, r7);
    _mm512_storeu_si512(dst + 8 * dst_stride, r8);
    _mm512_storeu_si512(dst + 9 * dst_stride, r9);
    _mm512_storeu_si512(dst + 10 * dst_stride, ra);
    _mm512_storeu_si512(dst + 11 * dst_stride, rb);
    _mm512_storeu_si512(dst + 12 * dst_stride, rc);
    _mm512_storeu_si512(dst + 13 * dst_stride, rd);
    _mm512_storeu_si512(dst + 14 * dst_stride, re);
    _mm512_storeu_si512(dst + 15 * dst_stride, rf);
}
#elif defined(HAVE_AVX2)
template <typename T>
inline void transpose_16x16_kernel(float* dst, T** src, size_t dst_stride) {
    __m256 r0, r1, r2, r3, r4, r5, r6, r7;

    for (int i = 0; i < 16; i += 8) {
        for (int j = 0; j < 16; j += 8) {
            r0 = mm256_uni_loadu_ps(src[j]);
            r1 = mm256_uni_loadu_ps(src[1 + j]);
            r2 = mm256_uni_loadu_ps(src[2 + j]);
            r3 = mm256_uni_loadu_ps(src[3 + j]);
            r4 = mm256_uni_loadu_ps(src[4 + j]);
            r5 = mm256_uni_loadu_ps(src[5 + j]);
            r6 = mm256_uni_loadu_ps(src[6 + j]);
            r7 = mm256_uni_loadu_ps(src[7 + j]);

            transpose_8x8(r0, r1, r2, r3, r4, r5, r6, r7);

            _mm256_storeu_ps(dst + j, r0);
            _mm256_storeu_ps(dst + j + dst_stride, r1);
            _mm256_storeu_ps(dst + j + dst_stride * 2, r2);
            _mm256_storeu_ps(dst + j + dst_stride * 3, r3);
            _mm256_storeu_ps(dst + j + dst_stride * 4, r4);
            _mm256_storeu_ps(dst + j + dst_stride * 5, r5);
            _mm256_storeu_ps(dst + j + dst_stride * 6, r6);
            _mm256_storeu_ps(dst + j + dst_stride * 7, r7);
        }
        src += 8;
        dst += 8 * dst_stride;
    }
}
#else
template <typename TSRC, typename TDST>
inline void transpose_16x16_kernel(TDST* dst, TSRC** src, size_t dst_stride) {
    for (size_t i = 0; i < 16; i++) {
        for (size_t j = 0; j < 16; j++) {
            dst[i * dst_stride + j] = static_cast<TDST>(src[j][i]);
        }
    }
}
#endif

template <typename TSRC, typename TDST>
inline void transpose_16xK_kernel(TDST* dst, TSRC** src, size_t K, size_t dst_stride) {
    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < 16; j++) {
            dst[i * dst_stride + j] = static_cast<TDST>(src[j][i]);
        }
    }
}

template <typename TDST, typename TSRC>
inline void transpose_tailx16_kernel(TDST* dst, TSRC** src, size_t n_cnt, size_t k_cnt, size_t dst_stride) {
    for (size_t i = 0; i < n_cnt; i++) {
        for (size_t j = 0; j < k_cnt; j++) {
            dst[j * dst_stride + i] = static_cast<TDST>(src[i][j]);
        }
    }
}

template <typename TDST,
          ov::element::Type_t SRC_PREC,
          std::enable_if_t<(none_of(SRC_PREC, ov::element::i8, ov::element::u8, ov::element::u4)), bool> = true>
void transpose_16NxK(TDST* dst,
                     void** src,
                     const size_t N,
                     const size_t K,
                     const size_t block_size,
                     const size_t dst_stride) {
    size_t k = 0;
    auto** src_ptr = reinterpret_cast<typename ov::element_type_traits<SRC_PREC>::value_type**>(src);
    for (size_t k = 0; k < K; k++) {
        memset(dst + k * dst_stride + N, 0, (block_size - N) * sizeof(TDST));
    }

    for (; k + 16 <= K; k += 16) {
        size_t n = 0;
        for (; n + 16 <= N; n += 16) {
            transpose_16x16_kernel(dst + n, src_ptr + n, dst_stride);
        }

        if (n < block_size) {
            transpose_tailx16_kernel(dst + n, src_ptr + n, N - n, 16, dst_stride);
        }

        dst += 16 * dst_stride;

        for (size_t i = 0; i < N; i++) {
            src_ptr[i] += 16;
        }
    }
    if (k < K) {
        size_t n = 0;
        for (; n + 16 <= N; n += 16) {
            transpose_16xK_kernel(dst + n, src_ptr + n, K - k, dst_stride);
        }

        if (n < block_size) {
            transpose_tailx16_kernel(dst + n, src_ptr + n, N - n, K - k, dst_stride);
        }
    }
}

#if defined(HAVE_AVX512F)
template <typename T,
          ov::element::Type_t SRC_PREC,
          typename std::enable_if<(SRC_PREC == ov::element::bf16 || SRC_PREC == ov::element::f16) &&
                                      (SRC_PREC == precision_of<T>::value),
                                  bool>::type = true>
static void transpose_16NxK(T* dst,
                            T** src,
                            const size_t N,
                            const size_t K,
                            const size_t block_size,
                            const size_t dst_stride) {
    // will treat as uint32_t transpose
    auto s = reinterpret_cast<void**>(src);
    auto d = reinterpret_cast<uint32_t*>(dst);
    transpose_16NxK<uint32_t, ov::element::u32>(d, s, N, K >> 1, block_size, dst_stride);
}
#endif

#if defined(HAVE_AVX512F) || defined(HAVE_AVX2)
static inline float hsum256_ps(__m256 v) {
    __m256 t1 = _mm256_hadd_ps(v, v);
    __m256 t2 = _mm256_hadd_ps(t1, t1);
    __m128 lo = _mm256_castps256_ps128(t2);
    __m128 hi = _mm256_extractf128_ps(t2, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    return _mm_cvtss_f32(sum);
}

void sum_blocks8x8(const float* a, size_t M, size_t N, size_t a_stride, float* out, size_t out_stride) {
    size_t block_num = (M + 7) / 8;
    size_t col_num = rnd_up(N, 8); // The src must be packed.
    size_t i = 0;
    for (; i + 8 <= M; i += 8) {
        for (size_t j = 0; j + 8 <= col_num; j += 8) {
            __m256 r0 = _mm256_loadu_ps(a + (i + 0) * a_stride + j);
            __m256 r1 = _mm256_loadu_ps(a + (i + 1) * a_stride + j);
            __m256 r2 = _mm256_loadu_ps(a + (i + 2) * a_stride + j);
            __m256 r3 = _mm256_loadu_ps(a + (i + 3) * a_stride + j);
            __m256 r4 = _mm256_loadu_ps(a + (i + 4) * a_stride + j);
            __m256 r5 = _mm256_loadu_ps(a + (i + 5) * a_stride + j);
            __m256 r6 = _mm256_loadu_ps(a + (i + 6) * a_stride + j);
            __m256 r7 = _mm256_loadu_ps(a + (i + 7) * a_stride + j);

            __m256 sum = _mm256_add_ps(r0, r1);
            sum = _mm256_add_ps(sum, r2);
            sum = _mm256_add_ps(sum, r3);
            sum = _mm256_add_ps(sum, r4);
            sum = _mm256_add_ps(sum, r5);
            sum = _mm256_add_ps(sum, r6);
            sum = _mm256_add_ps(sum, r7);

            const int ib = i >> 3;
            const int jb = j >> 3;
            const float block_sum = hsum256_ps(sum);
            out[ib * out_stride + jb] = block_sum;
        }
    }

    auto tails = M - i;
    if (tails) {
        for (size_t j = 0; j + 8 <= col_num; j += 8) {
            __m256 sum = _mm256_setzero_ps();
            for (size_t row = i; row < M; row++) {
                __m256 r = _mm256_loadu_ps(a + row * a_stride + j);
                sum = _mm256_add_ps(sum, r);
            }
            const float block_sum = hsum256_ps(sum);
            const int jb = j >> 3;
            out[(block_num - 1) * out_stride + jb] = block_sum;
        }
    }
}
#endif

#if defined(OPENVINO_ARCH_X86_64)
PlainTensor xattn_estimate(PlainTensor& query,
                           PlainTensor& key,
                           KAttr& k_attr,
                           size_t block_size,
                           size_t stride,
                           int norm = 1,
                           float threshold = 0.9f,
                           bool causal = true) {
    auto B = query.size(0);
    auto H = query.size(1);
    auto L = query.size(2);
    auto S = query.size(3);

    auto K_H = key.size(1);
    auto groups = H / K_H;

    // Align k length to multiple of stride and multiple of 32, since block
    // size in brgemm computation is 32.
    auto k_padded = rnd_up((size_t)k_attr.k_len, stride * 32);
    auto k_num_to_pad = k_padded - k_attr.k_len;
    auto k_num_strided = div_up(k_padded, stride);
    auto q_num_strided = div_up(B, stride);
    auto q_num_blocks = div_up(B, block_size);
    auto k_num_blocks = div_up((size_t)k_attr.k_len, block_size);
    auto num_per_block = block_size / stride;

    if (q_num_blocks == 0 || k_num_blocks == 0 || num_per_block == 0) {
        return {};
    }

    PlainTensor attn_sum_temp;
    attn_sum_temp.resize({H, L, q_num_strided, k_num_strided}, sizeof(float), ov::element::Type_t::f32);
    parallel_for3d(q_num_strided, H, L, [&](size_t b, size_t h, size_t l) {
        memset(attn_sum_temp.ptr<float>(h, l, b, 0), 0, k_num_strided * sizeof(float));
    });

    size_t n_block_size = 32;
    auto m_block_size = BrgemmKernel::get_mblk_size();
    auto m_num_blocks = div_up(q_num_strided, m_block_size);
    auto in_type = query.m_dt;

    std::vector<std::shared_ptr<BrgemmKernel>> kernels(m_block_size);
    for (size_t i = 1; i < m_block_size + 1; i++) {
        kernels[i - 1] = std::make_shared<BrgemmKernel>(i,                   // M
                                                        n_block_size,        // N
                                                        S,                   // K
                                                        S * stride * H * L,  // lda
                                                        n_block_size,        // ldb
                                                        k_num_strided,       // ldc
                                                        false,
                                                        in_type,
                                                        true);
    }

    auto max_threads_num = parallel_get_max_threads();
    auto scratch_a_size = kernels.back()->get_scratch_a_size();
    auto scratch_b_size = kernels.back()->get_scratch_b_size();
    auto wsp_size = kernels.back()->get_wsp_size();

    std::vector<uint8_t> scratch_a(scratch_a_size * max_threads_num, 0.0f);
    std::vector<uint8_t> scratch_b(scratch_b_size * max_threads_num, 0.0f);
    std::vector<uint8_t> wsp(wsp_size * max_threads_num, 0.0f);
    auto n_num_blocks = k_num_strided / n_block_size;

    // Repack key buffer to [H, L, S * n_num_block, n_block_size]
    PlainTensor key_repack;
    key_repack.resize({K_H, L, S * n_num_blocks, n_block_size}, key.m_element_size, key.m_dt);

    for (size_t i = 0; i < stride; i++) {
        ov::parallel_for2d(K_H, n_num_blocks, [&](size_t h, size_t n_blk) {
            auto n_start = n_blk * n_block_size;
            size_t need_to_pad = (k_num_to_pad + i) / stride;
            size_t N = (n_blk == n_num_blocks - 1) ? n_block_size - need_to_pad : 32;

            if (N) {
                std::vector<void*> src(N);
                auto* src_ptr = src.data();

                // Fill the src addresses from KVCache
                {
                    for (size_t j = 0; j < N; j++) {
                        auto n_index = n_start * stride + i + j * stride;

                        auto cache_block_offset = n_index % k_attr.block_size;
                        auto cache_block_num =
                            k_attr.block_indices
                                .ptr<int32_t>()[k_attr.block_indice_begin + n_index / k_attr.block_size];

                        src[j] = key.ptr_v(cache_block_num, h, cache_block_offset, 0);
                    }
                }

                void* dst = key_repack.ptr_v(h, 0, S * n_blk, 0);
                if (in_type == ov::element::bf16) {
#    if defined(HAVE_AVX512F)
                    transpose_16NxK<bfloat16, ov::element::bf16>(reinterpret_cast<bfloat16*>(dst),          // dst
                                                                 reinterpret_cast<bfloat16**>(src.data()),  // src
                                                                 N,                                         // N
                                                                 S,                                         // K
                                                                 n_block_size,  // block size
                                                                 n_block_size   // dst stride
                    );
#    else
                    OPENVINO_THROW("xattention: bf16 needs avx512+ hardware.");
#    endif
                } else if (in_type == ov::element::f32) {
                    transpose_16NxK<float, ov::element::f32>(reinterpret_cast<float*>(dst),  // dst
                                                             src.data(),                     // src
                                                             N,                              // N
                                                             S,                              // K
                                                             n_block_size,                   // block size
                                                             n_block_size);
                } else {
                    OPENVINO_THROW("xattention: unsupported precision: ", in_type);
                }

            } else {
                for (size_t k = 0; k < S; k++) {
                    auto* dst = key_repack.ptr_v(h, 0, S * n_blk + k, 0);
                    memset(dst, 0, n_block_size * key_repack.m_element_size);
                }
            }
        });

        auto n_past_end_block = (k_num_strided - q_num_strided) / n_block_size;
        ov::parallel_for2d(H, m_num_blocks, [&](size_t h, size_t m_blk) {
            auto ithr = parallel_get_thread_num();
            auto m_start = m_blk * m_block_size;
            auto m_end = std::min(m_start + m_block_size, q_num_strided);
            auto m_cnt = m_end - m_start;

            auto q_index = m_start * stride + stride - i - 1;
            auto q_end = q_index + stride * (m_cnt - 1);

            // q_end may extend the seq length since it was aligned to multiple of stride
            if (q_end >= B) {
                m_cnt--;
            }

            if (m_cnt > 0) {
                auto* q_ptr = query.ptr_v(stride - 1 - i + m_start * stride, h, 0, 0);
                auto cur_k_block_num = n_past_end_block + m_blk + 1;
                for (size_t n_blk = 0; n_blk < cur_k_block_num; n_blk++) {
                    auto n_start = n_blk * S;
                    auto* k_ptr = key_repack.ptr_v(h / groups, 0, n_start, 0);
                    auto* c_ptr = &attn_sum_temp.at<float>({h, 0, m_start, n_blk * n_block_size});
                    kernels[m_cnt - 1]->executeGemm(m_cnt < m_block_size,
                                                    q_ptr,
                                                    k_ptr,
                                                    c_ptr,
                                                    nullptr,
                                                    nullptr,
                                                    wsp.data() + wsp_size * ithr,
                                                    scratch_a.data() + scratch_a_size * ithr);
                }
            }
        });
    }

    parallel_for3d(q_num_strided, H, L, [&](size_t b, size_t h, size_t l) {
        auto* data = attn_sum_temp.ptr<float>(h, l, b, 0);
        auto ncausal = (k_attr.k_len - B) / stride + b + 1;
        auto scale = 1.0 / sqrt(S) / stride / norm;
        attn_softmax_kernel<float>(data,
                                   reinterpret_cast<float*>(data),
                                   scale,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   false,
                                   ncausal,
                                   k_num_strided,
                                   ov::element::f32,
                                   ov::element::f32,
                                   0);
    });

    PlainTensor attn_sum;
    attn_sum.resize({H, L, q_num_blocks, k_num_blocks}, attn_sum_temp.m_element_size, attn_sum_temp.m_dt);

    size_t src_stride = attn_sum_temp.size(3);
    size_t dst_stride = attn_sum.size(3);
    size_t row_num = attn_sum_temp.size(2);
    size_t col_num = div_up(k_attr.k_len, stride);
    parallel_for2d(H, L, [&](size_t h, size_t l) {
        auto* src = attn_sum_temp.ptr<float>(h, l, 0, 0);
        auto* dst = attn_sum.ptr<float>(h, l, 0, 0);
#    if defined(HAVE_AVX512F) || defined(HAVE_AVX2)
        if (num_per_block == 8) {
            sum_blocks8x8(src, row_num, col_num, src_stride, dst, dst_stride);
        } else {
            ref::sum_blocks_ref(src, row_num, col_num, src_stride, dst, dst_stride, num_per_block);
        }
#    else
        ref::sum_blocks_ref(src, row_num, col_num, src_stride, dst, dst_stride, num_per_block);
#    endif
    });

    // Find blocks
    PlainTensor mask;
    mask.resize({H, q_num_blocks, k_num_blocks}, sizeof(bool), ov::element::Type_t::boolean);
    parallel_for3d(q_num_blocks, H, L, [&](size_t b, size_t h, size_t l) {
        auto* row = attn_sum.ptr<float>(h, l, b, 0);
        float required_sum = std::accumulate(row, row + k_num_blocks, 0.0f, std::plus<>()) * threshold;
        auto identity_index = (k_attr.k_len - B) / block_size + b;

        std::vector<std::pair<float, int>> values_with_index(k_num_blocks);
        for (size_t k = 0; k < k_num_blocks; k++) {
            values_with_index[k] = std::make_pair(row[k], k);
        }

        if (causal) {
            values_with_index[identity_index].first += 100000.0f;
            values_with_index[0].first += 100000.0f;
        }

        std::sort(values_with_index.begin(),
                  values_with_index.end(),
                  [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                      return a.first > b.first;
                  });

        if (causal) {
            values_with_index[1].first += values_with_index[0].first - 100000.0f * 2;
            values_with_index[0].first = 0.0f;
        }

        std::vector<float> cumsum_without_self(k_num_blocks, 0.0f);
        for (size_t i = 1; i < k_num_blocks; i++) {
            cumsum_without_self[i] = values_with_index[i - 1].first + cumsum_without_self[i - 1];
        }

        for (size_t i = 0; i < k_num_blocks; i++) {
            bool value = (cumsum_without_self[i] < required_sum);

            if (causal && i > identity_index) {
                value = false;
            }

            if (values_with_index[i].second == 0) {
                value = true;
            }

            *(mask.ptr<bool>(h, b, values_with_index[i].second)) = value;
        }
    });

    return mask;
}
#endif
}  // namespace ov::Extensions::Cpu::XARCH
