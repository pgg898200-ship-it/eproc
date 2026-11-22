#if !defined(__GNUC__) || defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

#include "eproc.h"
#include "eproc_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

EPROC_STATIC_ASSERT(sizeof(eproc_cmonoid_t) == 640, "eproc_cmonoid_t must be exactly 640 bytes");

/* ============================================================================
 * 2A. MULTI-RATE CROSSING-FREE AgACI + CONFORMAL PID CALIBRATOR
 * ============================================================================ */

#if defined(__GNUC__) || defined(__clang__)
#define EPROC_ALWAYS_INLINE static inline __attribute__((always_inline))
#else
#define EPROC_ALWAYS_INLINE static inline
#endif

EPROC_ALWAYS_INLINE double eproc_sorted_quantile(const double *sorted, uint32_t len, double alpha) {
    if (len == 0U) {
        return 0.0;
    }
    if (!isfinite(alpha) || alpha <= 0.0) {
        return sorted[len - 1U];
    }
    if (alpha >= 1.0) {
        return sorted[0];
    }
    const double pos = (1.0 - alpha) * (double)(len + 1U);
    if (!isfinite(pos) || pos >= (double)len) {
        return sorted[len - 1U];
    }
    if (pos <= 1.0) {
        return sorted[0];
    }
    /* Exact ceil(pos) - 1 without calling libm ceil() and safe against float-cast-overflow */
    const uint32_t ipos = (uint32_t)pos;
    uint32_t idx = (pos > (double)ipos) ? ipos : (ipos - 1U);
    if (idx >= len) {
        idx = len - 1U;
    }
    return sorted[idx];
}

/* 4-wide bit-exact 1/sqrt(x) for x >= 1.0 using 64-bit IEEE magic + 2 Newton-Raphson steps
 * (error < 1e-7, ZERO vdivpd, ZERO vsqrtpd, 0-ULP AVX2/scalar identical). */
#if defined(__AVX2__)
EPROC_ALWAYS_INLINE __m256d eproc_rsqrt4_avx2(__m256d vx) {
    const __m256d half_x = _mm256_mul_pd(_mm256_set1_pd(0.5), vx);
    const __m256d three_half = _mm256_set1_pd(1.5);
    const __m256i magic = _mm256_set1_epi64x(0x5fe6eb50c7b537a9LL);
    __m256i ix = _mm256_castpd_si256(vx);
    ix = _mm256_sub_epi64(magic, _mm256_srli_epi64(ix, 1));
    __m256d y = _mm256_castsi256_pd(ix);
    y = _mm256_mul_pd(y, _mm256_sub_pd(three_half, _mm256_mul_pd(half_x, _mm256_mul_pd(y, y))));
    y = _mm256_mul_pd(y, _mm256_sub_pd(three_half, _mm256_mul_pd(half_x, _mm256_mul_pd(y, y))));
    return y;
}

/* 4-wide bit-exact exp(x) for pre-clamped x in [-50.0, 0.0] (degree-5 Estrin, 0-ULP identical) */
EPROC_ALWAYS_INLINE __m256d eproc_exp4_neg50_clamped_avx2(__m256d vx) {
    static const double INV_LN2 = 1.44269504088896340736;
    static const double LN2_VAL = 0.69314718055994530942;
    static const double C5      = 1.0 / 120.0;
    static const double C4      = 1.0 / 24.0;
    static const double C3      = 1.0 / 6.0;
    static const double C2      = 0.5;

    const __m256d kf = _mm256_floor_pd(
        _mm256_add_pd(_mm256_mul_pd(vx, _mm256_set1_pd(INV_LN2)), _mm256_set1_pd(0.5)));
    const __m256d r = _mm256_sub_pd(vx, _mm256_mul_pd(kf, _mm256_set1_pd(LN2_VAL)));

    const __m256d r2 = _mm256_mul_pd(r, r);
    const __m256d r4 = _mm256_mul_pd(r2, r2);

    const __m256d q0 = _mm256_add_pd(_mm256_set1_pd(1.0), r);
    const __m256d q1 = _mm256_add_pd(_mm256_set1_pd(C2), _mm256_mul_pd(_mm256_set1_pd(C3), r));
    const __m256d q2 = _mm256_add_pd(_mm256_set1_pd(C4), _mm256_mul_pd(_mm256_set1_pd(C5), r));

    const __m256d poly = _mm256_add_pd(_mm256_add_pd(q0, _mm256_mul_pd(q1, r2)),
                                       _mm256_mul_pd(q2, r4));

    const __m128i ki32 = _mm256_cvttpd_epi32(kf);
    const __m256i ki64 = _mm256_cvtepi32_epi64(ki32);
    const __m256i exp_bits = _mm256_slli_epi64(
        _mm256_add_epi64(ki64, _mm256_set1_epi64x(1023LL)), 52);
    return _mm256_mul_pd(poly, _mm256_castsi256_pd(exp_bits));
}

EPROC_ALWAYS_INLINE __m256d eproc_exp4_neg50_avx2(__m256d vx) {
    vx = _mm256_max_pd(vx, _mm256_set1_pd(-50.0));
    vx = _mm256_min_pd(vx, _mm256_setzero_pd());
    return eproc_exp4_neg50_clamped_avx2(vx);
}

EPROC_ALWAYS_INLINE double eproc_hsum4_avx2(__m256d v) {
    /* Deterministic pairwise sum: (v[0] + v[1]) + (v[2] + v[3]) matching scalar */
    const __m128d lo = _mm256_castpd256_pd128(v);
    const __m128d hi = _mm256_extractf128_pd(v, 1);
    const __m128d s_lo = _mm_add_sd(lo, _mm_unpackhi_pd(lo, lo));
    const __m128d s_hi = _mm_add_sd(hi, _mm_unpackhi_pd(hi, hi));
    return _mm_cvtsd_f64(_mm_add_sd(s_lo, s_hi));
}

/* Fast L1 scalar-indexed 4-expert sorted quantile load for pre-clamped alpha in [0, 1]. */
EPROC_ALWAYS_INLINE __m256d eproc_sorted_quantile4_avx2(const double *sorted,
                                                        __m256d v_len_p1,
                                                        __m256d v_len_m1,
                                                        __m256d valpha) {
    const __m256d v_zero = _mm256_setzero_pd();
    const __m256d v_one = _mm256_set1_pd(1.0);
    const __m256d vpos = _mm256_mul_pd(_mm256_sub_pd(v_one, valpha), v_len_p1);
    const __m256d vc = _mm256_sub_pd(_mm256_ceil_pd(vpos), v_one);
    const __m256d vclamped = _mm256_min_pd(_mm256_max_pd(vc, v_zero), v_len_m1);
    const __m128i vidx = _mm256_cvttpd_epi32(vclamped);
    const uint64_t lo2 = (uint64_t)_mm_cvtsi128_si64(vidx);
    const uint64_t hi2 = (uint64_t)_mm_extract_epi64(vidx, 1);
    const uint32_t i0 = (uint32_t)(lo2 & 0xFFFFFFFFULL);
    const uint32_t i1 = (uint32_t)(lo2 >> 32);
    const uint32_t i2 = (uint32_t)(hi2 & 0xFFFFFFFFULL);
    const uint32_t i3 = (uint32_t)(hi2 >> 32);
    return _mm256_set_pd(sorted[i3], sorted[i2], sorted[i1], sorted[i0]);
}
#endif

EPROC_ALWAYS_INLINE double eproc_rsqrt_scalar(double x) {
    const double half_x = 0.5 * x;
    uint64_t u = eproc_f64_to_u64(x);
    u = 0x5fe6eb50c7b537a9ULL - (u >> 1);
    double y = eproc_u64_to_f64(u);
    y = y * (1.5 - half_x * (y * y));
    y = y * (1.5 - half_x * (y * y));
    return y;
}

EPROC_ALWAYS_INLINE double eproc_exp_neg50_scalar(double x) {
    static const double INV_LN2 = 1.44269504088896340736;
    static const double LN2_VAL = 0.69314718055994530942;
    static const double C5      = 1.0 / 120.0;
    static const double C4      = 1.0 / 24.0;
    static const double C3      = 1.0 / 6.0;
    static const double C2      = 0.5;

    x = eproc_clamp(x, -50.0, 0.0);
    const double kf = floor(x * INV_LN2 + 0.5);
    const double r = x - kf * LN2_VAL;
    const double r2 = r * r;
    const double r4 = r2 * r2;
    const double q0 = 1.0 + r;
    const double q1 = C2 + C3 * r;
    const double q2 = C4 + C5 * r;
    const double poly = (q0 + q1 * r2) + q2 * r4;
    const int64_t ki = (int64_t)kf;
    const uint64_t exp_bits = (uint64_t)(ki + 1023LL) << 52;
    return poly * eproc_u64_to_f64(exp_bits);
}

/* Fast bit-exact tanh(z) for z in [-10.0, +10.0] using exp_neg50(-2|z|) */
EPROC_ALWAYS_INLINE double eproc_tanh_fast_scalar(double z) {
    const double az = fabs(z);
    if (az >= 20.0) {
        return (z >= 0.0) ? 1.0 : -1.0;
    }
    const double e = eproc_exp_neg50_scalar(-2.0 * az);
    const double mag = (1.0 - e) / (1.0 + e);
    return (z >= 0.0) ? mag : -mag;
}

EPROC_ALWAYS_INLINE void eproc_tanh_rates(const double *z, double *out, uint32_t n) {
#if defined(__AVX2__)
    if (n >= 4U) {
        const __m256d vz = _mm256_loadu_pd(z);
        const __m256d sign_mask = _mm256_set1_pd(-0.0);
        const __m256d vaz = _mm256_andnot_pd(sign_mask, vz);
        const __m256d ve = eproc_exp4_neg50_avx2(_mm256_mul_pd(_mm256_set1_pd(-2.0), vaz));
        const __m256d one = _mm256_set1_pd(1.0);
        const __m256d vmag = _mm256_div_pd(_mm256_sub_pd(one, ve), _mm256_add_pd(one, ve));
        const __m256d vres = _mm256_or_pd(vmag, _mm256_and_pd(vz, sign_mask));
        _mm256_storeu_pd(out, vres);
        for (uint32_t i = 4U; i < n; ++i) {
            out[i] = eproc_tanh_fast_scalar(z[i]);
        }
        return;
    }
#endif
    for (uint32_t i = 0; i < n; ++i) {
        out[i] = eproc_tanh_fast_scalar(z[i]);
    }
}

/* Exact overflow-safe L2-optimal PAVA isotonic projection for non-decreasing q[0..n-1] */
EPROC_ALWAYS_INLINE void eproc_pava_nondecreasing(double *q, uint32_t n) {
    if (n <= 1U) {
        return;
    }
    int ordered = 1;
    for (uint32_t i = 1U; i < n; ++i) {
        if (q[i] < q[i - 1U]) {
            ordered = 0;
            break;
        }
    }
    if (ordered) {
        return;
    }

    double blk_val[EPROC_MAX_RATES];
    uint32_t blk_cnt[EPROC_MAX_RATES];
    uint32_t nb = 0;

    for (uint32_t i = 0; i < n; ++i) {
        blk_val[nb] = q[i];
        blk_cnt[nb] = 1U;
        nb++;
        while (nb >= 2U && blk_val[nb - 2U] > blk_val[nb - 1U]) {
            const uint32_t c_tot = blk_cnt[nb - 2U] + blk_cnt[nb - 1U];
            const double w2 = (double)blk_cnt[nb - 1U] / (double)c_tot;
            const double w1 = 1.0 - w2;
            blk_val[nb - 2U] = blk_val[nb - 2U] * w1 + blk_val[nb - 1U] * w2;
            blk_cnt[nb - 2U] = c_tot;
            nb--;
        }
    }

    uint32_t pos = 0;
    for (uint32_t b = 0; b < nb; ++b) {
        for (uint32_t c = 0; c < blk_cnt[b]; ++c) {
            q[pos++] = blk_val[b];
        }
    }
    for (uint32_t i = 1; i < n; ++i) {
        if (q[i] < q[i - 1]) {
            q[i] = q[i - 1];
        }
    }
}

/* Branchless lower-bound binary search in sorted[0..len-1] (0 branch mispredictions) */
EPROC_ALWAYS_INLINE uint32_t eproc_lower_bound(const double *sorted, uint32_t len, double val) {
    uint32_t base = 0U;
    uint32_t n = len;
    while (n > 1U) {
        const uint32_t half = n >> 1;
        base += (sorted[base + half] < val) ? half : 0U;
        n -= half;
    }
    base += (n > 0U && sorted[base] < val) ? 1U : 0U;
    return base;
}

/* Inline 8-double AVX2 shift helpers (prevents GCC memmove@plt + vzeroupper substitution) */
EPROC_ALWAYS_INLINE void eproc_shift_right1(double *arr, uint32_t start, uint32_t count) {
#if defined(__AVX2__)
    uint32_t rem = count;
    while (rem >= 8U) {
        rem -= 8U;
        __m256d vhi = _mm256_loadu_pd(arr + start + rem + 4U);
        __m256d vlo = _mm256_loadu_pd(arr + start + rem);
        _mm256_storeu_pd(arr + start + rem + 5U, vhi);
        _mm256_storeu_pd(arr + start + rem + 1U, vlo);
    }
    if (rem >= 4U) {
        rem -= 4U;
        __m256d v = _mm256_loadu_pd(arr + start + rem);
        _mm256_storeu_pd(arr + start + rem + 1U, v);
    }
    if (rem == 3U) {
        arr[start + 3U] = arr[start + 2U];
        arr[start + 2U] = arr[start + 1U];
        arr[start + 1U] = arr[start];
    } else if (rem == 2U) {
        arr[start + 2U] = arr[start + 1U];
        arr[start + 1U] = arr[start];
    } else if (rem == 1U) {
        arr[start + 1U] = arr[start];
    }
#else
    if (count > 0U) {
        memmove(&arr[start + 1U], &arr[start], count * sizeof(double));
    }
#endif
}

EPROC_ALWAYS_INLINE void eproc_shift_left1(double *arr, uint32_t dst_start, uint32_t count) {
#if defined(__AVX2__)
    uint32_t i = 0U;
    for (; i + 8U <= count; i += 8U) {
        __m256d vlo = _mm256_loadu_pd(arr + dst_start + i + 1U);
        __m256d vhi = _mm256_loadu_pd(arr + dst_start + i + 5U);
        _mm256_storeu_pd(arr + dst_start + i, vlo);
        _mm256_storeu_pd(arr + dst_start + i + 4U, vhi);
    }
    if (i + 4U <= count) {
        __m256d v = _mm256_loadu_pd(arr + dst_start + i + 1U);
        _mm256_storeu_pd(arr + dst_start + i, v);
        i += 4U;
    }
    const uint32_t rem = count - i;
    if (rem == 3U) {
        arr[dst_start + i]      = arr[dst_start + i + 1U];
        arr[dst_start + i + 1U] = arr[dst_start + i + 2U];
        arr[dst_start + i + 2U] = arr[dst_start + i + 3U];
    } else if (rem == 2U) {
        arr[dst_start + i]      = arr[dst_start + i + 1U];
        arr[dst_start + i + 1U] = arr[dst_start + i + 2U];
    } else if (rem == 1U) {
        arr[dst_start + i]      = arr[dst_start + i + 1U];
    }
#else
    if (count > 0U) {
        memmove(&arr[dst_start], &arr[dst_start + 1U], count * sizeof(double));
    }
#endif
}

/* Single-pass O(W/3) L1-resident sorted ring maintenance with interleaved dual binary search */
EPROC_ALWAYS_INLINE void eproc_conformal_push_score(eproc_conformal_t *cp, double score) {
    if (cp->win_len == cp->win_cap) {
        const double old_val = cp->ring[cp->ring_pos];
        cp->ring[cp->ring_pos] = score;
        cp->ring_pos++;
        if (cp->ring_pos >= cp->win_cap) {
            cp->ring_pos = 0U;
        }
        if (old_val == score) {
            return;
        }
        const uint32_t len = cp->win_len;
        /* Dual interleaved branchless lower_bound exploits both Haswell L1 load ports */
        uint32_t rm_idx = 0U;
        uint32_t ins_idx = 0U;
        uint32_t n = len;
        while (n > 1U) {
            const uint32_t half = n >> 1;
            rm_idx  += (cp->sorted[rm_idx  + half] < old_val) ? half : 0U;
            ins_idx += (cp->sorted[ins_idx + half] < score)   ? half : 0U;
            n -= half;
        }
        rm_idx  += (cp->sorted[rm_idx]  < old_val) ? 1U : 0U;
        ins_idx += (cp->sorted[ins_idx] < score)   ? 1U : 0U;
        if (rm_idx >= len) {
            rm_idx = len - 1U;
        }
        if (ins_idx <= rm_idx) {
            eproc_shift_right1(cp->sorted, ins_idx, rm_idx - ins_idx);
            cp->sorted[ins_idx] = score;
        } else {
            const uint32_t dst = ins_idx - 1U;
            eproc_shift_left1(cp->sorted, rm_idx, dst - rm_idx);
            cp->sorted[dst] = score;
        }
        return;
    }

    cp->ring[cp->ring_pos] = score;
    cp->ring_pos++;
    if (cp->ring_pos >= cp->win_cap) {
        cp->ring_pos = 0U;
    }

    const uint32_t ins_idx = eproc_lower_bound(cp->sorted, cp->win_len, score);
    eproc_shift_right1(cp->sorted, ins_idx, cp->win_len - ins_idx);
    cp->sorted[ins_idx] = score;
    cp->win_len++;
}

static inline void eproc_conformal_fill_meta(const eproc_conformal_t *cp,
                                             eproc_conformal_out_t *meta_out) {
    if (meta_out == NULL) {
        return;
    }
    memset(meta_out, 0, sizeof(*meta_out));
    meta_out->n_rates = cp->n_rates;
    for (uint32_t r = 0; r < cp->n_rates; ++r) {
        meta_out->quantiles[r] = cp->current_q[r];
        meta_out->eff_alphas[r] = cp->eff_alpha[r];
        if (cp->n_updates > 0) {
            meta_out->emp_coverage[r] = (double)cp->hit_counts[r] / (double)cp->n_updates;
        } else {
            meta_out->emp_coverage[r] = 1.0 - cp->target_alphas[r];
        }
    }
}

int eproc_conformal_init(eproc_conformal_t *cp,
                         const double *alphas,
                         uint32_t n_rates,
                         uint32_t window_size,
                         double pid_ki) {
    if (cp == NULL || alphas == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (n_rates == 0 || n_rates > EPROC_MAX_RATES) {
        return EPROC_ERR_INVALID;
    }
    if (window_size == 0 || window_size > EPROC_CONFORMAL_MAX_WIN) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(pid_ki) || pid_ki < 0.0) {
        return EPROC_ERR_INVALID;
    }
    for (uint32_t r = 0; r < n_rates; ++r) {
        if (!isfinite(alphas[r]) || !(alphas[r] > 0.0 && alphas[r] < 1.0)) {
            return EPROC_ERR_INVALID;
        }
    }

    memset(cp, 0, sizeof(*cp));
    cp->n_rates = n_rates;
    cp->win_cap = window_size;
    cp->win_len = 0;
    cp->ring_pos = 0;
    cp->pid_ki = pid_ki;
    cp->pid_csat = 10.0;
    cp->n_updates = 0;

    /* Copy and sort target_alphas descending (ascending coverage 1 - alpha) */
    for (uint32_t r = 0; r < n_rates; ++r) {
        cp->target_alphas[r] = alphas[r];
    }
    for (uint32_t i = 0; i < n_rates; ++i) {
        for (uint32_t j = i + 1U; j < n_rates; ++j) {
            if (cp->target_alphas[j] > cp->target_alphas[i]) {
                double tmp = cp->target_alphas[i];
                cp->target_alphas[i] = cp->target_alphas[j];
                cp->target_alphas[j] = tmp;
            }
        }
    }

    cp->gammas[0] = 0.005;
    cp->gammas[1] = 0.02;
    cp->gammas[2] = 0.08;
    cp->gammas[3] = 0.20;

    for (uint32_t r = 0; r < n_rates; ++r) {
        cp->eff_alpha[r] = cp->target_alphas[r];
        cp->current_q[r] = 0.0;
        cp->emp_cov_ema[r] = 1.0 - cp->target_alphas[r];
        cp->pid_err_sum[r] = 0.0;
        cp->hit_counts[r] = 0;
        for (uint32_t k = 0; k < EPROC_AGACI_EXPERTS; ++k) {
            cp->alpha_t[r][k] = cp->target_alphas[r];
            cp->exp_log_w[r][k] = 0.0;
            cp->exp_var[r][k] = 0.0;
        }
    }

    return EPROC_OK;
}

void eproc_conformal_predict(const eproc_conformal_t *cp,
                             double point_pred,
                             double scale,
                             double *lows_out,
                             double *highs_out,
                             eproc_conformal_out_t *meta_out) {
    if (cp == NULL) {
        return;
    }
    const double pred = isfinite(point_pred) ? point_pred : 0.0;
    const double sc = (isfinite(scale) && scale > 0.0) ? scale : 1.0;

    for (uint32_t r = 0; r < cp->n_rates; ++r) {
        const double rad = cp->current_q[r] * sc;
        const double lo = pred - rad;
        const double hi = pred + rad;
        if (lows_out != NULL) {
            lows_out[r] = isfinite(lo) ? lo : pred;
        }
        if (highs_out != NULL) {
            highs_out[r] = isfinite(hi) ? hi : pred;
        }
    }
    eproc_conformal_fill_meta(cp, meta_out);
}

/* Compute exact normalized weights from exp_log_w using the SAME exp4_neg50
 * kernel as the inner step -> guarantees 0.0-ULP scalar vs batch identity! */
EPROC_ALWAYS_INLINE void eproc_conformal_load_weights(
    const eproc_conformal_t *cp,
    double w_cache[EPROC_MAX_RATES][EPROC_AGACI_EXPERTS]) {
    for (uint32_t r = 0; r < cp->n_rates; ++r) {
#if defined(__AVX2__)
        const __m256d vlw = _mm256_loadu_pd(cp->exp_log_w[r]);
        const __m256d vw = eproc_exp4_neg50_avx2(vlw);
        const double w_sum = eproc_hsum4_avx2(vw);
        const __m256d vnorm = (w_sum > 0.0)
            ? _mm256_mul_pd(vw, _mm256_set1_pd(1.0 / w_sum))
            : _mm256_set1_pd(0.25);
        _mm256_storeu_pd(w_cache[r], vnorm);
#else
        double w[4];
        for (int k = 0; k < 4; ++k) {
            w[k] = eproc_exp_neg50_scalar(cp->exp_log_w[r][k]);
        }
        const double w_sum = (w[0] + w[1]) + (w[2] + w[3]);
        const double inv_s = (w_sum > 0.0) ? (1.0 / w_sum) : 0.25;
        for (int k = 0; k < 4; ++k) {
            w_cache[r][k] = w[k] * inv_s;
        }
#endif
    }
}

/* 3-pass ROB-overlapping, zero-spill AVX2 4-expert AgACI BOA + PID step:
 * Parameterized by compile-time-specializable n_rates_arg so the common 5-rate
 * steady-state batch path unrolls cleanly and overlaps all 5 rates in Haswell's ROB. */
EPROC_ALWAYS_INLINE void eproc_conformal_step_cached(
    eproc_conformal_t *cp,
    double score,
    double w_cache[EPROC_MAX_RATES][EPROC_AGACI_EXPERTS],
    double inv_csat,
    double max_pid_sum,
    uint32_t n_rates) {
    static const double SQRT_LOG4 = 1.17741002251547469101;
    const uint32_t prev_len = cp->win_len;

    const double ema_beta = (cp->n_updates < 50ULL)
        ? (1.0 / (double)(cp->n_updates + 1ULL))
        : 0.02;

    if (prev_len == 0U) {
        for (uint32_t r = 0; r < n_rates; ++r) {
            cp->emp_cov_ema[r] = (1.0 - ema_beta) * cp->emp_cov_ema[r];
            cp->eff_alpha[r] = cp->target_alphas[r];
        }
        eproc_conformal_push_score(cp, score);
        cp->n_updates++;
        for (uint32_t r = 0; r < n_rates; ++r) {
            cp->current_q[r] = score;
        }
        return;
    }

    /* Overflow-safe F_{t-1}-predictable half-range (guaranteed finite for all finite doubles!) */
    double half_scale;
    if (prev_len >= 2U) {
        const double half_rng = 0.5 * cp->sorted[prev_len - 1U] - 0.5 * cp->sorted[0];
        half_scale = (half_rng > 0.5e-6) ? half_rng : 0.5e-6;
    } else {
        const double half_ab = 0.5 * fabs(score);
        half_scale = (half_ab > 0.5) ? half_ab : 0.5;
    }
    const double inv_half_scale = 1.0 / half_scale;
    const double half_score = 0.5 * score;

    double a_agg_arr[EPROC_MAX_RATES] = {0.0};
    double pid_z[EPROC_MAX_RATES] = {0.0};
    double w_sum_arr[EPROC_MAX_RATES] = {1.0, 1.0, 1.0, 1.0, 1.0};
    double wa_sum_arr[EPROC_MAX_RATES] = {0.0};
    const double pid_ki = cp->pid_ki;

#if defined(__AVX2__)
    double reg_buf[EPROC_MAX_RATES][4];
    double reg2_buf[EPROC_MAX_RATES][4];

    /* Vectorized per-rate coverage, EMA, and PID integral across rates 0..3 (+ rate 4) */
    {
        const __m256d v_one = _mm256_set1_pd(1.0);
        const __m256d v_score = _mm256_set1_pd(score);
        if (n_rates >= 4U) {
            const __m256d v_cq = _mm256_loadu_pd(cp->current_q);
            const __m256d v_ta = _mm256_loadu_pd(cp->target_alphas);
            const __m256d v_hit = _mm256_cmp_pd(v_score, v_cq, _CMP_LE_OQ);
            const __m256d v_cov = _mm256_and_pd(v_hit, v_one);
            const __m256d v_err = _mm256_andnot_pd(v_hit, v_one);
            const int hmask = _mm256_movemask_pd(v_hit);
            cp->hit_counts[0] += (uint64_t)((hmask >> 0) & 1);
            cp->hit_counts[1] += (uint64_t)((hmask >> 1) & 1);
            cp->hit_counts[2] += (uint64_t)((hmask >> 2) & 1);
            cp->hit_counts[3] += (uint64_t)((hmask >> 3) & 1);

            const __m256d v_ema = _mm256_add_pd(
                _mm256_mul_pd(_mm256_set1_pd(1.0 - ema_beta), _mm256_loadu_pd(cp->emp_cov_ema)),
                _mm256_mul_pd(_mm256_set1_pd(ema_beta), v_cov));
            _mm256_storeu_pd(cp->emp_cov_ema, v_ema);

            __m256d v_pid = _mm256_add_pd(_mm256_loadu_pd(cp->pid_err_sum),
                                          _mm256_sub_pd(v_err, v_ta));
            v_pid = _mm256_min_pd(_mm256_max_pd(v_pid, _mm256_set1_pd(-max_pid_sum)),
                                  _mm256_set1_pd(max_pid_sum));
            _mm256_storeu_pd(cp->pid_err_sum, v_pid);
            _mm256_storeu_pd(pid_z, _mm256_mul_pd(v_pid, _mm256_set1_pd(inv_csat)));

            for (uint32_t r = 4U; r < n_rates; ++r) {
                const double target_a = cp->target_alphas[r];
                const int covered = (score <= cp->current_q[r]);
                const double err = (double)(1 - covered);
                cp->hit_counts[r] += (uint64_t)covered;
                cp->emp_cov_ema[r] = (1.0 - ema_beta) * cp->emp_cov_ema[r] + ema_beta * (double)covered;
                cp->pid_err_sum[r] = eproc_clamp(
                    cp->pid_err_sum[r] + (err - target_a), -max_pid_sum, max_pid_sum);
                pid_z[r] = cp->pid_err_sum[r] * inv_csat;
            }
        } else {
            for (uint32_t r = 0; r < n_rates; ++r) {
                const double target_a = cp->target_alphas[r];
                const int covered = (score <= cp->current_q[r]);
                const double err = (double)(1 - covered);
                cp->hit_counts[r] += (uint64_t)covered;
                cp->emp_cov_ema[r] = (1.0 - ema_beta) * cp->emp_cov_ema[r] + ema_beta * (double)covered;
                cp->pid_err_sum[r] = eproc_clamp(
                    cp->pid_err_sum[r] + (err - target_a), -max_pid_sum, max_pid_sum);
                pid_z[r] = cp->pid_err_sum[r] * inv_csat;
            }
        }
    }

    /* Micro-pass 1: Quantile load, pinball regret, variance, ACI alpha_t */
    {
        const __m256d v_gam = _mm256_loadu_pd(cp->gammas);
        const __m256d v_half = _mm256_set1_pd(0.5);
        const __m256d v_one = _mm256_set1_pd(1.0);
        const __m256d v_neg1 = _mm256_set1_pd(-1.0);
        const __m256d v_zero = _mm256_setzero_pd();
        const __m256d v_lim100 = _mm256_set1_pd(1e100);
        const __m256d v_nlim100 = _mm256_set1_pd(-1e100);
        const __m256d v_score = _mm256_set1_pd(score);
        const __m256d v_hscore = _mm256_set1_pd(half_score);
        const __m256d v_ihs = _mm256_set1_pd(inv_half_scale);
        const __m256d v_len_p1 = _mm256_set1_pd((double)(prev_len + 1U));
        const __m256d v_len_m1 = _mm256_set1_pd((double)(prev_len - 1U));

        for (uint32_t r = 0; r < n_rates; ++r) {
            const double target_a = cp->target_alphas[r];
            const double tau = 1.0 - target_a;
            const __m256d valpha = _mm256_loadu_pd(cp->alpha_t[r]);
            const __m256d vq = eproc_sorted_quantile4_avx2(cp->sorted, v_len_p1, v_len_m1, valpha);
            const __m256d vwr = _mm256_loadu_pd(w_cache[r]);

            __m256d vsd = _mm256_mul_pd(_mm256_sub_pd(v_hscore, _mm256_mul_pd(v_half, vq)), v_ihs);
            vsd = _mm256_min_pd(_mm256_max_pd(vsd, v_nlim100), v_lim100);
            const __m256d v_ge0 = _mm256_cmp_pd(vsd, v_zero, _CMP_GE_OQ);
            const __m256d v_slope = _mm256_blendv_pd(_mm256_set1_pd(tau - 1.0),
                                                     _mm256_set1_pd(tau), v_ge0);
            const __m256d vloss = _mm256_mul_pd(v_slope, vsd);
            const double l_bar = eproc_hsum4_avx2(_mm256_mul_pd(vwr, vloss));

            __m256d vreg = _mm256_sub_pd(_mm256_set1_pd(l_bar), vloss);
            vreg = _mm256_min_pd(_mm256_max_pd(vreg, v_neg1), v_one);
            const __m256d vreg2 = _mm256_mul_pd(vreg, vreg);
            _mm256_storeu_pd(reg_buf[r], vreg);
            _mm256_storeu_pd(reg2_buf[r], vreg2);
            _mm256_storeu_pd(cp->exp_var[r], _mm256_add_pd(_mm256_loadu_pd(cp->exp_var[r]), vreg2));

            const __m256d v_hit = _mm256_cmp_pd(v_score, vq, _CMP_LE_OQ);
            const __m256d v_errk = _mm256_andnot_pd(v_hit, v_one);
            __m256d valpha_new = _mm256_add_pd(
                valpha, _mm256_mul_pd(v_gam, _mm256_sub_pd(_mm256_set1_pd(target_a), v_errk)));
            valpha_new = _mm256_min_pd(_mm256_max_pd(valpha_new, v_zero), v_one);
            _mm256_storeu_pd(cp->alpha_t[r], valpha_new);
        }
    }

    /* Micro-pass 2: Bit-exact rsqrt4 BOA learning rate + log-weight shift (0 spills!) */
    {
        const __m256d v_one = _mm256_set1_pd(1.0);
        const __m256d v_half = _mm256_set1_pd(0.5);
        const __m256d v_sqlog4 = _mm256_set1_pd(SQRT_LOG4);
        const __m256d v_neg50 = _mm256_set1_pd(-50.0);

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 5
#endif
        for (uint32_t r = 0; r < n_rates; ++r) {
            const __m256d vreg = _mm256_loadu_pd(reg_buf[r]);
            const __m256d vreg2 = _mm256_loadu_pd(reg2_buf[r]);
            const __m256d vdenom = _mm256_max_pd(_mm256_loadu_pd(cp->exp_var[r]), v_one);
            const __m256d veta = _mm256_min_pd(_mm256_mul_pd(v_sqlog4, eproc_rsqrt4_avx2(vdenom)), v_half);
            const __m256d vdelta = _mm256_sub_pd(_mm256_mul_pd(veta, vreg),
                                                 _mm256_mul_pd(_mm256_mul_pd(veta, veta), vreg2));
            __m256d vlw = _mm256_add_pd(_mm256_loadu_pd(cp->exp_log_w[r]), vdelta);
            const __m256d vmax1 = _mm256_max_pd(vlw, _mm256_permute_pd(vlw, 0x5));
            const __m256d vmax2 = _mm256_permute2f128_pd(vmax1, vmax1, 0x1);
            const __m256d vmax_all = _mm256_max_pd(vmax1, vmax2);
            vlw = _mm256_max_pd(_mm256_sub_pd(vlw, vmax_all), v_neg50);
            _mm256_storeu_pd(cp->exp_log_w[r], vlw);
        }
    }

    /* Micro-pass 3: Compact degree-5 exp4_neg50 (pre-clamped) + sums */
    {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 5
#endif
        for (uint32_t r = 0; r < n_rates; ++r) {
            const __m256d vw_raw = eproc_exp4_neg50_clamped_avx2(_mm256_loadu_pd(cp->exp_log_w[r]));
            _mm256_storeu_pd(w_cache[r], vw_raw);
            w_sum_arr[r] = eproc_hsum4_avx2(vw_raw);
            wa_sum_arr[r] = eproc_hsum4_avx2(_mm256_mul_pd(vw_raw, _mm256_loadu_pd(cp->alpha_t[r])));
        }
    }

    /* Micro-pass 4: Vectorized 1/w_sum normalization via dedicated divider unit */
    {
        double inv_w_arr[EPROC_MAX_RATES];
        const __m256d v_one = _mm256_set1_pd(1.0);
        if (n_rates >= 4U) {
            _mm256_storeu_pd(inv_w_arr, _mm256_div_pd(v_one, _mm256_loadu_pd(w_sum_arr)));
            for (uint32_t r = 4U; r < n_rates; ++r) {
                inv_w_arr[r] = 1.0 / w_sum_arr[r];
            }
        } else {
            for (uint32_t r = 0; r < n_rates; ++r) {
                inv_w_arr[r] = 1.0 / w_sum_arr[r];
            }
        }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 5
#endif
        for (uint32_t r = 0; r < n_rates; ++r) {
            const double inv_s = inv_w_arr[r];
            _mm256_storeu_pd(w_cache[r],
                             _mm256_mul_pd(_mm256_loadu_pd(w_cache[r]), _mm256_set1_pd(inv_s)));
            a_agg_arr[r] = wa_sum_arr[r] * inv_s;
        }
    }
#else
    double w_raw_arr[EPROC_MAX_RATES][4];
    for (uint32_t r = 0; r < n_rates; ++r) {
        const double target_a = cp->target_alphas[r];
        const double tau = 1.0 - target_a;
        const int covered = (score <= cp->current_q[r]);
        const double err = (double)(1 - covered);

        cp->hit_counts[r] += (uint64_t)covered;
        cp->emp_cov_ema[r] = (1.0 - ema_beta) * cp->emp_cov_ema[r] + ema_beta * (double)covered;
        cp->pid_err_sum[r] = eproc_clamp(
            cp->pid_err_sum[r] + (err - target_a), -max_pid_sum, max_pid_sum);
        pid_z[r] = cp->pid_err_sum[r] * inv_csat;

        const double *at = cp->alpha_t[r];
        double q_exp[4];
        q_exp[0] = eproc_sorted_quantile(cp->sorted, prev_len, at[0]);
        q_exp[1] = eproc_sorted_quantile(cp->sorted, prev_len, at[1]);
        q_exp[2] = eproc_sorted_quantile(cp->sorted, prev_len, at[2]);
        q_exp[3] = eproc_sorted_quantile(cp->sorted, prev_len, at[3]);

        const double *wr = w_cache[r];
        double loss_exp[4];
        for (int k = 0; k < 4; ++k) {
            double sd = (half_score - 0.5 * q_exp[k]) * inv_half_scale;
            sd = eproc_clamp(sd, -1e100, 1e100);
            loss_exp[k] = (sd >= 0.0) ? (tau * sd) : ((tau - 1.0) * sd);
        }
        const double l_bar = (wr[0] * loss_exp[0] + wr[1] * loss_exp[1])
                           + (wr[2] * loss_exp[2] + wr[3] * loss_exp[3]);

        double lw_new[4];
        double alpha_new[4];
        for (int k = 0; k < 4; ++k) {
            const double rk = eproc_clamp(l_bar - loss_exp[k], -1.0, 1.0);
            const double rk2 = rk * rk;
            const double v_new = cp->exp_var[r][k] + rk2;
            cp->exp_var[r][k] = v_new;
            const double denom = (v_new > 1.0) ? v_new : 1.0;
            const double e_raw = SQRT_LOG4 * eproc_rsqrt_scalar(denom);
            const double ek = (e_raw < 0.5) ? e_raw : 0.5;
            const double dk = ek * rk - (ek * ek) * rk2;
            lw_new[k] = cp->exp_log_w[r][k] + dk;

            const double err_k = (score <= q_exp[k]) ? 0.0 : 1.0;
            alpha_new[k] = eproc_clamp(at[k] + cp->gammas[k] * (target_a - err_k), 0.0, 1.0);
            cp->alpha_t[r][k] = alpha_new[k];
        }

        const double m01 = (lw_new[0] > lw_new[1]) ? lw_new[0] : lw_new[1];
        const double m23 = (lw_new[2] > lw_new[3]) ? lw_new[2] : lw_new[3];
        const double max_lw = (m01 > m23) ? m01 : m23;

        for (int k = 0; k < 4; ++k) {
            double s_lw = lw_new[k] - max_lw;
            if (s_lw < -50.0) s_lw = -50.0;
            cp->exp_log_w[r][k] = s_lw;
            w_raw_arr[r][k] = eproc_exp_neg50_scalar(s_lw);
        }
        w_sum_arr[r] = (w_raw_arr[r][0] + w_raw_arr[r][1]) + (w_raw_arr[r][2] + w_raw_arr[r][3]);
        wa_sum_arr[r] = (w_raw_arr[r][0] * alpha_new[0] + w_raw_arr[r][1] * alpha_new[1])
                      + (w_raw_arr[r][2] * alpha_new[2] + w_raw_arr[r][3] * alpha_new[3]);
    }
    for (uint32_t r = 0; r < n_rates; ++r) {
        const double inv_s = 1.0 / w_sum_arr[r];
        for (int k = 0; k < 4; ++k) {
            w_cache[r][k] = w_raw_arr[r][k] * inv_s;
        }
        a_agg_arr[r] = wa_sum_arr[r] * inv_s;
    }
#endif

    if (pid_ki > 0.0) {
        double th[EPROC_MAX_RATES];
        eproc_tanh_rates(pid_z, th, n_rates);
        for (uint32_t r = 0; r < n_rates; ++r) {
            a_agg_arr[r] -= pid_ki * th[r];
        }
    }

    /* Insert score into circular raw buffer + single-pass inline-AVX2 sorted ring */
    eproc_conformal_push_score(cp, score);
    cp->n_updates++;

#if defined(__AVX2__)
    if (n_rates >= 4U) {
        const __m256d v_zero = _mm256_setzero_pd();
        const __m256d v_one = _mm256_set1_pd(1.0);
        const __m256d v_new_lp1 = _mm256_set1_pd((double)(cp->win_len + 1U));
        const __m256d v_new_lm1 = _mm256_set1_pd((double)(cp->win_len - 1U));
        __m256d vea = _mm256_min_pd(_mm256_max_pd(_mm256_loadu_pd(a_agg_arr), v_zero), v_one);
        _mm256_storeu_pd(cp->eff_alpha, vea);
        __m256d vcq = eproc_sorted_quantile4_avx2(cp->sorted, v_new_lp1, v_new_lm1, vea);
        _mm256_storeu_pd(cp->current_q, vcq);
        for (uint32_t r = 4U; r < n_rates; ++r) {
            const double ea = eproc_clamp(a_agg_arr[r], 0.0, 1.0);
            cp->eff_alpha[r] = ea;
            cp->current_q[r] = eproc_sorted_quantile(cp->sorted, cp->win_len, ea);
        }
    } else
#endif
    {
        for (uint32_t r = 0; r < n_rates; ++r) {
            const double ea = eproc_clamp(a_agg_arr[r], 0.0, 1.0);
            cp->eff_alpha[r] = ea;
            cp->current_q[r] = eproc_sorted_quantile(cp->sorted, cp->win_len, ea);
        }
    }

    /* Crossing-free guarantee via overflow-safe PAVA isotonic projection */
    eproc_pava_nondecreasing(cp->current_q, n_rates);
}

int eproc_conformal_update(eproc_conformal_t *cp,
                           double nonconformity_score,
                           eproc_conformal_out_t *meta_out) {
    if (cp == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(nonconformity_score)) {
        eproc_conformal_fill_meta(cp, meta_out);
        return EPROC_OK;
    }
    const double inv_csat = (cp->pid_csat > 0.0) ? (1.0 / cp->pid_csat) : 0.0;
    const double max_pid_sum = 10.0 * cp->pid_csat;
    double w_cache[EPROC_MAX_RATES][EPROC_AGACI_EXPERTS];
    eproc_conformal_load_weights(cp, w_cache);
    eproc_conformal_step_cached(cp, nonconformity_score, w_cache, inv_csat, max_pid_sum, cp->n_rates);
    eproc_conformal_fill_meta(cp, meta_out);
    return EPROC_OK;
}

int eproc_conformal_update_batch(eproc_conformal_t *cp,
                                 const double *scores,
                                 size_t n,
                                 eproc_conformal_out_t *meta_out) {
    if (cp == NULL || (n > 0 && scores == NULL)) {
        return EPROC_ERR_INVALID;
    }
    const double inv_csat = (cp->pid_csat > 0.0) ? (1.0 / cp->pid_csat) : 0.0;
    const double max_pid_sum = 10.0 * cp->pid_csat;
    double w_cache[EPROC_MAX_RATES][EPROC_AGACI_EXPERTS];
    eproc_conformal_load_weights(cp, w_cache);
    const uint32_t nr = cp->n_rates;
    if (nr == 5U) {
        for (size_t i = 0; i < n; ++i) {
            const double s = scores[i];
            if (!isfinite(s)) {
                continue;
            }
            eproc_conformal_step_cached(cp, s, w_cache, inv_csat, max_pid_sum, 5U);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            const double s = scores[i];
            if (!isfinite(s)) {
                continue;
            }
            eproc_conformal_step_cached(cp, s, w_cache, inv_csat, max_pid_sum, nr);
        }
    }
    eproc_conformal_fill_meta(cp, meta_out);
    return EPROC_OK;
}

/* ============================================================================
 * 2B. STRICTLY CAUSAL DELAYED-FEEDBACK CONFORMAL WRAPPER
 * ============================================================================ */

int eproc_delayed_conformal_init(eproc_delayed_conformal_t *dc,
                                 const double *alphas,
                                 uint32_t n_rates,
                                 uint32_t window_size,
                                 double pid_ki) {
    if (dc == NULL) {
        return EPROC_ERR_INVALID;
    }
    memset(dc, 0, sizeof(*dc));
    int rc = eproc_conformal_init(&dc->core, alphas, n_rates, window_size, pid_ki);
    if (rc != EPROC_OK) {
        return rc;
    }
    dc->pending_count = 0;
    dc->next_ticket = 1ULL;
    return EPROC_OK;
}

int eproc_delayed_conformal_issue(eproc_delayed_conformal_t *dc,
                                  double pred,
                                  double scale,
                                  uint64_t *ticket_out,
                                  double *lows_out,
                                  double *highs_out) {
    if (dc == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (dc->pending_count >= EPROC_DELAY_MAX_PENDING) {
        return EPROC_ERR_CAPACITY;
    }

    /* Issue predictions using ONLY labels revealed <= t_issue */
    eproc_conformal_predict(&dc->core, pred, scale, lows_out, highs_out, NULL);

    const uint64_t tid = dc->next_ticket++;
    const uint32_t mask = EPROC_DELAY_MAX_PENDING - 1U;
    uint32_t idx = (uint32_t)(tid & mask);
    for (uint32_t probe = 0; probe < EPROC_DELAY_MAX_PENDING; ++probe) {
        if (!dc->slots[idx].occupied) {
            dc->slots[idx].ticket_id = tid;
            dc->slots[idx].pred = pred;
            dc->slots[idx].scale = (isfinite(scale) && scale > 0.0) ? scale : 1.0;
            dc->slots[idx].occupied = 1U;
            dc->pending_count++;
            if (ticket_out != NULL) {
                *ticket_out = tid;
            }
            return EPROC_OK;
        }
        idx = (idx + 1U) & mask;
    }
    return EPROC_ERR_CAPACITY;
}

int eproc_delayed_conformal_reveal(eproc_delayed_conformal_t *dc,
                                   uint64_t ticket_id,
                                   double y_true,
                                   eproc_conformal_out_t *meta_out) {
    if (dc == NULL || ticket_id == 0ULL) {
        return EPROC_ERR_INVALID;
    }
    const uint32_t mask = EPROC_DELAY_MAX_PENDING - 1U;
    uint32_t idx = (uint32_t)(ticket_id & mask);
    uint32_t found = EPROC_DELAY_MAX_PENDING;

    for (uint32_t probe = 0; probe < EPROC_DELAY_MAX_PENDING; ++probe) {
        if (!dc->slots[idx].occupied) {
            break;
        }
        if (dc->slots[idx].ticket_id == ticket_id) {
            found = idx;
            break;
        }
        idx = (idx + 1U) & mask;
    }

    if (found == EPROC_DELAY_MAX_PENDING) {
        return EPROC_ERR_INVALID;
    }

    const double pred = dc->slots[found].pred;
    const double scale = dc->slots[found].scale;

    /* Knuth Algorithm R backward-shift deletion (O(1), zero tombstones) */
    dc->slots[found].occupied = 0U;
    uint32_t hole = found;
    uint32_t cur = (found + 1U) & mask;
    while (dc->slots[cur].occupied) {
        const uint32_t home = (uint32_t)(dc->slots[cur].ticket_id & mask);
        const uint32_t d_hole = (hole - home) & mask;
        const uint32_t d_cur = (cur - home) & mask;
        if (d_hole < d_cur) {
            dc->slots[hole] = dc->slots[cur];
            dc->slots[cur].occupied = 0U;
            hole = cur;
        }
        cur = (cur + 1U) & mask;
    }
    dc->pending_count--;

    if (!isfinite(y_true) || !isfinite(pred)) {
        eproc_conformal_fill_meta(&dc->core, meta_out);
        return EPROC_OK;
    }

    const double denom = (scale > 1e-12) ? scale : 1e-12;
    const double score = fabs(y_true - pred) / denom;
    return eproc_conformal_update(&dc->core, score, meta_out);
}

/* ============================================================================
 * 2C. RIGHT-CENSORED AGENT BUDGET CONFORMAL CALIBRATOR (PROBE-STYLE UPB)
 * ============================================================================ */

int eproc_budget_conformal_init(eproc_budget_conformal_t *bc,
                                double alpha,
                                uint32_t window_size,
                                double censor_penalty) {
    if (bc == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(censor_penalty) || censor_penalty < 1.0) {
        return EPROC_ERR_INVALID;
    }
    memset(bc, 0, sizeof(*bc));
    int rc = eproc_conformal_init(&bc->core, &alpha, 1U, window_size, 0.0);
    if (rc != EPROC_OK) {
        return rc;
    }
    bc->censor_penalty = censor_penalty;
    bc->n_completed = 0;
    bc->n_censored = 0;
    return EPROC_OK;
}

double eproc_budget_conformal_bound(const eproc_budget_conformal_t *bc,
                                    double base_estimate) {
    if (bc == NULL) {
        return 0.0;
    }
    const double base = (isfinite(base_estimate) && base_estimate > 0.0) ? base_estimate : 0.0;
    const double q = bc->core.current_q[0];
    const double raw = base + q;
    if (!isfinite(raw)) {
        return base;
    }
    if (base > 0.0) {
        return (raw > 1e-12) ? raw : 1e-12;
    }
    return (raw > 0.0) ? raw : 0.0;
}

int eproc_budget_conformal_update(eproc_budget_conformal_t *bc,
                                  double base_estimate,
                                  double observed_cost,
                                  int32_t censored) {
    if (bc == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(base_estimate) || !isfinite(observed_cost)) {
        return EPROC_OK;
    }
    if (observed_cost < 0.0) {
        return EPROC_ERR_INVALID;
    }

    double score;
    if (censored != 0) {
        score = (observed_cost * bc->censor_penalty) - base_estimate;
        bc->n_censored++;
    } else {
        score = observed_cost - base_estimate;
        bc->n_completed++;
    }
    return eproc_conformal_update(&bc->core, score, NULL);
}

/* ============================================================================
 * 2D. EXACT ASSOCIATIVE & COMMUTATIVE DISTRIBUTED CONFORMAL MONOID (640 B)
 * ============================================================================ */

#define EPROC_CMONOID_WIRE_MAGIC 0x455052434D4F4E01ULL /* "EPRCMON\x01" */
#define EPROC_CMONOID_FLAG_SCALE_MISMATCH 1ULL

/* 63 exact interior edges spanning [-16, +32] with signed log spacing:
 *   bin  0: (-inf, -16)
 *   bin  1..15: [-16, 0) via -(17^u - 1)
 *   bin 16..62: [0, 32)  via +(33^u - 1) (~7.7% uniform relative resolution)
 *   bin 63: [32, +inf)
 */
static const double EPROC_CMONOID_EDGES[EPROC_CMONOID_BINS - 1] = {
    -16.0,
    -13.074046879278832,
    -10.651693856478721,
    -8.6462638564168639,
    -6.9859982191237586,
    -5.6114890184579442,
    -4.4735533169184887,
    -3.531473292855468,
    -2.7515392679910513,
    -2.1058435015977324,
    -1.5712815906582351,
    -1.1287257439265086,
    -0.76234034783231697,
    -0.45901533368450531,
    -0.20789707081543396,
    0.0,
    0.077230911718957573,
    0.16042643716285654,
    0.25004722888772535,
    0.34658951606648092,
    0.45058785210348495,
    0.56261807444988143,
    0.68330049300816764,
    0.8133033247801591,
    0.95334639377594788,
    1.1042051166702023,
    1.2667147962743375,
    1.4417752465974552,
    1.6303557751049591,
    1.8335005497615402,
    2.0523343805757919,
    2.2880689476587794,
    2.5420095102812605,
    2.8155621340775001,
    3.1102414764126367,
    3.4276791730210592,
    3.7696328723525152,
    4.1379959676490108,
    4.5348080806388706,
    4.9622663548960633,
    5.4227376213959522,
    5.9187715036280117,
    6.4531145348283463,
    7.0287253654989517,
    7.6487911454175563,
    8.3167451708450031,
    9.0362858946425533,
    9.8113974045579138,
    10.646371483067893,
    11.545831370922896,
    12.514757365971565,
    13.558514399006048,
    14.682881739314858,
    15.89408499442273,
    17.198830581199559,
    18.604342859204451,
    20.118404131871845,
    21.749397737025703,
    23.50635446531339,
    25.39900256357749,
    27.437821600033679,
    29.634100489505336,
    32.0
};

static inline uint32_t eproc_cmonoid_bin(double z) {
    uint32_t lo = 0U;
    uint32_t hi = EPROC_CMONOID_BINS - 1U;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (z < EPROC_CMONOID_EDGES[mid]) {
            hi = mid;
        } else {
            lo = mid + 1U;
        }
    }
    return lo;
}

static inline uint64_t eproc_fnv1a64(const uint8_t *data, size_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= (uint64_t)data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

void eproc_cmonoid_init(eproc_cmonoid_t *m, double scale_hint) {
    if (m == NULL) {
        return;
    }
    memset(m, 0, sizeof(*m));
    m->scale_hint = (isfinite(scale_hint) && scale_hint > 0.0) ? scale_hint : 1.0;
    m->min_val = INFINITY;
    m->max_val = -INFINITY;
}

/* Identical scaling (score * (1.0 / scale)) in BOTH scalar add and add_batch (fixes W07) */
void eproc_cmonoid_add(eproc_cmonoid_t *m, double score) {
    if (m == NULL || !isfinite(score)) {
        return;
    }
    const double scale = (isfinite(m->scale_hint) && m->scale_hint > 0.0) ? m->scale_hint : 1.0;
    const double inv_scale = 1.0 / scale;
    const uint32_t bin = eproc_cmonoid_bin(score * inv_scale);
    m->bin_counts[bin]++;
    m->count++;
    eproc_kahan_add(&m->sum, &m->sum_comp, score);
    const double sq = score * score;
    if (isfinite(sq)) {
        eproc_kahan_add(&m->sum_sq, &m->sum_sq_comp, sq);
    }
    if (score < m->min_val) {
        m->min_val = score;
    }
    if (score > m->max_val) {
        m->max_val = score;
    }
}

void eproc_cmonoid_add_batch(eproc_cmonoid_t *m, const double *scores, size_t n) {
    if (m == NULL || (n > 0 && scores == NULL)) {
        return;
    }
    const double scale = (isfinite(m->scale_hint) && m->scale_hint > 0.0) ? m->scale_hint : 1.0;
    const double inv_scale = 1.0 / scale;
    uint64_t count = m->count;
    double sum = m->sum;
    double sum_comp = m->sum_comp;
    double sum_sq = m->sum_sq;
    double sum_sq_comp = m->sum_sq_comp;
    double min_val = m->min_val;
    double max_val = m->max_val;

    for (size_t i = 0; i < n; ++i) {
        const double s = scores[i];
        if (!isfinite(s)) {
            continue;
        }
        const uint32_t bin = eproc_cmonoid_bin(s * inv_scale);
        m->bin_counts[bin]++;
        count++;
        eproc_kahan_add(&sum, &sum_comp, s);
        const double sq = s * s;
        if (isfinite(sq)) {
            eproc_kahan_add(&sum_sq, &sum_sq_comp, sq);
        }
        if (s < min_val) {
            min_val = s;
        }
        if (s > max_val) {
            max_val = s;
        }
    }

    m->count = count;
    m->sum = sum;
    m->sum_comp = sum_comp;
    m->sum_sq = sum_sq;
    m->sum_sq_comp = sum_sq_comp;
    m->min_val = min_val;
    m->max_val = max_val;
}

void eproc_cmonoid_merge(const eproc_cmonoid_t *a,
                         const eproc_cmonoid_t *b,
                         eproc_cmonoid_t *out) {
    if (a == NULL || b == NULL || out == NULL) {
        return;
    }
    if (a->count == 0ULL) {
        if (out != b) {
            *out = *b;
        }
        return;
    }
    if (b->count == 0ULL) {
        if (out != a) {
            *out = *a;
        }
        return;
    }

    /* Capture scalar fields first so out == a or out == b aliasing is 100% safe */
    const uint64_t new_count = a->count + b->count;
    double sum = 0.0, sum_comp = 0.0;
    eproc_kahan_merge_sym(a->sum, a->sum_comp, b->sum, b->sum_comp, &sum, &sum_comp);

    double sum_sq = 0.0, sum_sq_comp = 0.0;
    eproc_kahan_merge_sym(a->sum_sq, a->sum_sq_comp, b->sum_sq, b->sum_sq_comp, &sum_sq, &sum_sq_comp);

    const double min_val = (a->min_val < b->min_val) ? a->min_val : b->min_val;
    const double max_val = (a->max_val > b->max_val) ? a->max_val : b->max_val;
    const double log_wealth = a->log_wealth + b->log_wealth;

    const double scale_ref = fmax(1.0, fmax(fabs(a->scale_hint), fabs(b->scale_hint)));
    const int scale_mismatch = (fabs(a->scale_hint - b->scale_hint) > 1e-12 * scale_ref) ? 1 : 0;
    const double scale_hint = scale_mismatch
        ? ((a->scale_hint < b->scale_hint) ? a->scale_hint : b->scale_hint)
        : a->scale_hint;
    const uint64_t flags = a->flags | b->flags | (scale_mismatch ? EPROC_CMONOID_FLAG_SCALE_MISMATCH : 0ULL);

#if defined(__AVX2__)
    for (size_t i = 0; i < EPROC_CMONOID_BINS; i += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a->bin_counts + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(const void *)(b->bin_counts + i));
        __m256i vs = _mm256_add_epi64(va, vb);
        _mm256_storeu_si256((__m256i *)(void *)(out->bin_counts + i), vs);
    }
#else
    for (size_t i = 0; i < EPROC_CMONOID_BINS; ++i) {
        out->bin_counts[i] = a->bin_counts[i] + b->bin_counts[i];
    }
#endif

    out->count = new_count;
    out->sum = sum;
    out->sum_comp = sum_comp;
    out->sum_sq = sum_sq;
    out->sum_sq_comp = sum_sq_comp;
    out->min_val = min_val;
    out->max_val = max_val;
    out->log_wealth = log_wealth;
    out->scale_hint = scale_hint;
    out->flags = flags;
    memset(out->_reserved, 0, sizeof(out->_reserved));
}

int eproc_cmonoid_subtract(const eproc_cmonoid_t *global_m,
                           const eproc_cmonoid_t *local_m,
                           eproc_cmonoid_t *loo_out) {
    if (global_m == NULL || local_m == NULL || loo_out == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (local_m->count > global_m->count) {
        return EPROC_ERR_INVALID;
    }
    if (local_m->count > 0ULL && global_m->count > 0ULL) {
        const double ref = fmax(1.0, fabs(global_m->scale_hint));
        if (fabs(local_m->scale_hint - global_m->scale_hint) > 1e-12 * ref) {
            return EPROC_ERR_MISMATCH;
        }
    }
    /* Strict underflow guard BEFORE mutating loo_out */
    for (size_t i = 0; i < EPROC_CMONOID_BINS; ++i) {
        if (local_m->bin_counts[i] > global_m->bin_counts[i]) {
            return EPROC_ERR_INVALID;
        }
    }

    const uint64_t new_count = global_m->count - local_m->count;
    double sum = global_m->sum;
    double sum_comp = global_m->sum_comp;
    eproc_kahan_add(&sum, &sum_comp, -local_m->sum);
    eproc_kahan_add(&sum, &sum_comp, -local_m->sum_comp);

    double sum_sq = global_m->sum_sq;
    double sum_sq_comp = global_m->sum_sq_comp;
    eproc_kahan_add(&sum_sq, &sum_sq_comp, -local_m->sum_sq);
    eproc_kahan_add(&sum_sq, &sum_sq_comp, -local_m->sum_sq_comp);

    const double log_wealth = global_m->log_wealth - local_m->log_wealth;
    const double scale_hint = global_m->scale_hint;
    const uint64_t flags = global_m->flags;
    const double g_min = global_m->min_val;
    const double g_max = global_m->max_val;
    const double l_min = local_m->min_val;
    const double l_max = local_m->max_val;

#if defined(__AVX2__)
    for (size_t i = 0; i < EPROC_CMONOID_BINS; i += 4) {
        __m256i vg = _mm256_loadu_si256((const __m256i *)(const void *)(global_m->bin_counts + i));
        __m256i vl = _mm256_loadu_si256((const __m256i *)(const void *)(local_m->bin_counts + i));
        __m256i vd = _mm256_sub_epi64(vg, vl);
        _mm256_storeu_si256((__m256i *)(void *)(loo_out->bin_counts + i), vd);
    }
#else
    for (size_t i = 0; i < EPROC_CMONOID_BINS; ++i) {
        loo_out->bin_counts[i] = global_m->bin_counts[i] - local_m->bin_counts[i];
    }
#endif

    loo_out->count = new_count;
    loo_out->scale_hint = scale_hint;
    loo_out->flags = flags;
    loo_out->log_wealth = log_wealth;
    memset(loo_out->_reserved, 0, sizeof(loo_out->_reserved));

    if (new_count == 0ULL) {
        loo_out->sum = 0.0;
        loo_out->sum_comp = 0.0;
        loo_out->sum_sq = 0.0;
        loo_out->sum_sq_comp = 0.0;
        loo_out->min_val = INFINITY;
        loo_out->max_val = -INFINITY;
    } else {
        loo_out->sum = sum;
        loo_out->sum_comp = sum_comp;
        loo_out->sum_sq = sum_sq;
        loo_out->sum_sq_comp = sum_sq_comp;

        /* W04 fix: tighten min_val/max_val against occupied LOO bins so removed
         * outlier shards NEVER leak stale extreme values into loo_out. */
        uint32_t first_bin = 0U;
        while (first_bin < EPROC_CMONOID_BINS - 1U && loo_out->bin_counts[first_bin] == 0ULL) {
            first_bin++;
        }
        uint32_t last_bin = EPROC_CMONOID_BINS - 1U;
        while (last_bin > 0U && loo_out->bin_counts[last_bin] == 0ULL) {
            last_bin--;
        }
        const double sc = (isfinite(scale_hint) && scale_hint > 0.0) ? scale_hint : 1.0;

        double new_min = g_min;
        if (first_bin > 0U) {
            const double bin_lo = EPROC_CMONOID_EDGES[first_bin - 1U] * sc;
            if (!isfinite(new_min) || new_min < bin_lo || l_min <= g_min) {
                new_min = bin_lo;
            }
        }
        double new_max = g_max;
        if (last_bin < EPROC_CMONOID_BINS - 1U) {
            const double bin_hi = EPROC_CMONOID_EDGES[last_bin] * sc;
            if (!isfinite(new_max) || new_max > bin_hi || l_max >= g_max) {
                new_max = bin_hi;
            }
        }
        if (isfinite(new_min) && isfinite(new_max) && new_min > new_max) {
            new_min = new_max;
        }
        loo_out->min_val = new_min;
        loo_out->max_val = new_max;
    }

    return EPROC_OK;
}

double eproc_cmonoid_quantile(const eproc_cmonoid_t *m, double q) {
    if (m == NULL || m->count == 0ULL || !isfinite(q)) {
        return 0.0;
    }
    const double qc = eproc_clamp(q, 0.0, 1.0);
    const double scale = (isfinite(m->scale_hint) && m->scale_hint > 0.0) ? m->scale_hint : 1.0;

    /* Exact abelian histogram quantile: for all interior bins 1..62, depends strictly
     * on exact uint64 bin_counts (guaranteeing 0.0 LOO drift: subtract(merge(A,B),A) == B).
     * Tail overflow bins (0 and 63) use min_val / max_val. */
    if (qc == 0.0) {
        uint32_t first_bin = 0U;
        while (first_bin < EPROC_CMONOID_BINS - 1U && m->bin_counts[first_bin] == 0ULL) {
            first_bin++;
        }
        if (first_bin == 0U) {
            const double hi0 = EPROC_CMONOID_EDGES[0] * scale;
            return (isfinite(m->min_val) && m->min_val < hi0) ? m->min_val : hi0;
        }
        return EPROC_CMONOID_EDGES[first_bin - 1U] * scale;
    }
    if (qc == 1.0) {
        uint32_t last_bin = EPROC_CMONOID_BINS - 1U;
        while (last_bin > 0U && m->bin_counts[last_bin] == 0ULL) {
            last_bin--;
        }
        if (last_bin == EPROC_CMONOID_BINS - 1U) {
            const double lo63 = EPROC_CMONOID_EDGES[EPROC_CMONOID_BINS - 2U] * scale;
            return (isfinite(m->max_val) && m->max_val > lo63) ? m->max_val : lo63;
        }
        return EPROC_CMONOID_EDGES[last_bin] * scale;
    }

    const double target = qc * (double)m->count;
    uint64_t cum = 0ULL;
    uint32_t hit = EPROC_CMONOID_BINS - 1U;
    uint64_t prev_cum = 0ULL;

    for (uint32_t b = 0; b < EPROC_CMONOID_BINS; ++b) {
        const uint64_t c = m->bin_counts[b];
        if (c == 0ULL) {
            continue;
        }
        prev_cum = cum;
        cum += c;
        hit = b;
        if ((double)cum >= target) {
            break;
        }
    }

    const uint64_t bin_c = m->bin_counts[hit];
    double frac = 0.5;
    if (bin_c > 0ULL) {
        frac = eproc_clamp((target - (double)prev_cum) / (double)bin_c, 0.0, 1.0);
    }

    double lo, hi;
    if (hit == 0U) {
        hi = EPROC_CMONOID_EDGES[0] * scale;
        lo = (isfinite(m->min_val) && m->min_val < hi) ? m->min_val : hi;
    } else if (hit == EPROC_CMONOID_BINS - 1U) {
        lo = EPROC_CMONOID_EDGES[EPROC_CMONOID_BINS - 2U] * scale;
        hi = (isfinite(m->max_val) && m->max_val > lo) ? m->max_val : lo;
    } else {
        lo = EPROC_CMONOID_EDGES[hit - 1U] * scale;
        hi = EPROC_CMONOID_EDGES[hit] * scale;
    }

    return lo + frac * (hi - lo);
}

size_t eproc_cmonoid_wire_size(void) {
    return 640U;
}

int eproc_cmonoid_serialize(const eproc_cmonoid_t *m, uint8_t *buf, size_t cap) {
    if (m == NULL || buf == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (cap < 640U) {
        return EPROC_ERR_CAPACITY;
    }

    size_t off = 0;
    eproc_put_le64(buf + off, m->count);
    off += 8;
    for (size_t i = 0; i < EPROC_CMONOID_BINS; ++i) {
        eproc_put_le64(buf + off, m->bin_counts[i]);
        off += 8;
    }
    eproc_put_le64(buf + off, eproc_f64_to_u64(m->sum)); off += 8;
    eproc_put_le64(buf + off, eproc_f64_to_u64(m->sum_comp)); off += 8;
    eproc_put_le64(buf + off, eproc_f64_to_u64(m->sum_sq)); off += 8;
    eproc_put_le64(buf + off, eproc_f64_to_u64(m->sum_sq_comp)); off += 8;
    eproc_put_le64(buf + off, eproc_f64_to_u64(m->min_val)); off += 8;
    eproc_put_le64(buf + off, eproc_f64_to_u64(m->max_val)); off += 8;
    eproc_put_le64(buf + off, eproc_f64_to_u64(m->log_wealth)); off += 8;
    eproc_put_le64(buf + off, eproc_f64_to_u64(m->scale_hint)); off += 8;
    eproc_put_le64(buf + off, m->flags); off += 8;

    /* Reserved header: magic+version in slot 0, zeros in 1..4, FNV-1a checksum in slot 5 */
    eproc_put_le64(buf + off, EPROC_CMONOID_WIRE_MAGIC); off += 8;
    eproc_put_le64(buf + off, 0ULL); off += 8;
    eproc_put_le64(buf + off, 0ULL); off += 8;
    eproc_put_le64(buf + off, 0ULL); off += 8;
    eproc_put_le64(buf + off, 0ULL); off += 8;
    const uint64_t csum = eproc_fnv1a64(buf, off);
    eproc_put_le64(buf + off, csum); off += 8;
    (void)off;

    return EPROC_OK;
}

int eproc_cmonoid_deserialize(eproc_cmonoid_t *m, const uint8_t *buf, size_t len) {
    if (m == NULL || buf == NULL) {
        return EPROC_ERR_INVALID;
    }
    /* W11 fix: require exact 640-byte wire payload (reject both truncated & trailing garbage) */
    if (len != 640U) {
        return EPROC_ERR_INVALID;
    }

    const uint64_t magic = eproc_get_le64(buf + 592);
    if (magic != EPROC_CMONOID_WIRE_MAGIC) {
        return EPROC_ERR_MISMATCH;
    }
    const uint64_t stored_csum = eproc_get_le64(buf + 632);
    const uint64_t calc_csum = eproc_fnv1a64(buf, 632);
    if (stored_csum != calc_csum) {
        return EPROC_ERR_MISMATCH;
    }

    eproc_cmonoid_t tmp;
    memset(&tmp, 0, sizeof(tmp));

    size_t off = 0;
    tmp.count = eproc_get_le64(buf + off); off += 8;
    uint64_t bin_sum = 0ULL;
    for (size_t i = 0; i < EPROC_CMONOID_BINS; ++i) {
        tmp.bin_counts[i] = eproc_get_le64(buf + off);
        off += 8;
        if (UINT64_MAX - bin_sum < tmp.bin_counts[i]) {
            return EPROC_ERR_INVALID;
        }
        bin_sum += tmp.bin_counts[i];
    }
    if (bin_sum != tmp.count) {
        return EPROC_ERR_INVALID;
    }

    tmp.sum = eproc_u64_to_f64(eproc_get_le64(buf + off)); off += 8;
    tmp.sum_comp = eproc_u64_to_f64(eproc_get_le64(buf + off)); off += 8;
    tmp.sum_sq = eproc_u64_to_f64(eproc_get_le64(buf + off)); off += 8;
    tmp.sum_sq_comp = eproc_u64_to_f64(eproc_get_le64(buf + off)); off += 8;
    tmp.min_val = eproc_u64_to_f64(eproc_get_le64(buf + off)); off += 8;
    tmp.max_val = eproc_u64_to_f64(eproc_get_le64(buf + off)); off += 8;
    tmp.log_wealth = eproc_u64_to_f64(eproc_get_le64(buf + off)); off += 8;
    tmp.scale_hint = eproc_u64_to_f64(eproc_get_le64(buf + off)); off += 8;
    tmp.flags = eproc_get_le64(buf + off); off += 8;

    if (!isfinite(tmp.scale_hint) || !(tmp.scale_hint > 0.0)) {
        return EPROC_ERR_INVALID;
    }
    if (tmp.count > 0ULL) {
        if (!isfinite(tmp.min_val) || !isfinite(tmp.max_val) || tmp.min_val > tmp.max_val) {
            return EPROC_ERR_INVALID;
        }
    }

    *m = tmp;
    return EPROC_OK;
}
