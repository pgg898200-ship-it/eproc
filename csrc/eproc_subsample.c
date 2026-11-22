#if !defined(__GNUC__) || defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

#include "eproc.h"
#include "eproc_internal.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ============================================================================
 * 4A. COHEN-2011 VARIANCE-OPTIMAL WEIGHTED RESERVOIR SAMPLER (VAROPT)
 * ============================================================================ */

static inline void eproc_varopt_sift_up(eproc_varopt_item_t *heap, uint32_t idx) {
    while (idx > 0U) {
        const uint32_t parent = (idx - 1U) >> 1;
        if (heap[idx].orig_weight < heap[parent].orig_weight) {
            eproc_varopt_item_t tmp = heap[idx];
            heap[idx] = heap[parent];
            heap[parent] = tmp;
            idx = parent;
        } else {
            break;
        }
    }
}

static inline void eproc_varopt_sift_down(eproc_varopt_item_t *heap,
                                          uint32_t n,
                                          uint32_t idx) {
    for (;;) {
        const uint32_t left = (idx << 1) + 1U;
        const uint32_t right = left + 1U;
        uint32_t smallest = idx;
        if (left < n && heap[left].orig_weight < heap[smallest].orig_weight) {
            smallest = left;
        }
        if (right < n && heap[right].orig_weight < heap[smallest].orig_weight) {
            smallest = right;
        }
        if (smallest != idx) {
            eproc_varopt_item_t tmp = heap[idx];
            heap[idx] = heap[smallest];
            heap[smallest] = tmp;
            idx = smallest;
        } else {
            break;
        }
    }
}

int eproc_varopt_init(eproc_varopt_t *vo, uint32_t k, uint64_t seed) {
    if (vo == NULL || k == 0U || k > EPROC_VAROPT_MAX_K) {
        return EPROC_ERR_INVALID;
    }
    memset(vo, 0, sizeof(*vo));
    vo->k = k;
    vo->count = 0U;
    vo->n_heavy = 0U;
    vo->tau = 0.0;
    vo->total_weight = 0.0;
    vo->total_weight_comp = 0.0;
    vo->n_seen = 0ULL;
    eproc_rng_seed(vo->rng_state, seed);
    return EPROC_OK;
}

int eproc_varopt_push(eproc_varopt_t *vo,
                      uint64_t item_id,
                      double weight,
                      double value) {
    if (vo == NULL || !isfinite(weight) || !isfinite(value) || weight < 0.0) {
        return EPROC_ERR_INVALID;
    }
    vo->n_seen++;
    if (weight == 0.0) {
        return EPROC_OK;
    }

    eproc_kahan_add(&vo->total_weight, &vo->total_weight_comp, weight);

    eproc_varopt_item_t incoming;
    incoming.item_id = item_id;
    incoming.orig_weight = weight;
    incoming.adj_weight = weight;
    incoming.value = value;
    incoming.rand_u = 0.0;

    /* Phase 1: Reservoir not yet full -> keep deterministically in heavy min-heap */
    if (vo->count < vo->k) {
        const uint32_t pos = vo->count;
        vo->items[pos] = incoming;
        vo->count++;
        vo->n_heavy = vo->count;
        eproc_varopt_sift_up(vo->items, pos);
        return EPROC_OK;
    }

    /* Phase 2: Reservoir full (k items) -> exact Cohen-2011 VarOpt threshold tau'.
     * W06 fix: scale threshold arithmetic by 1e-150 when total_weight > 1e200 so
     * extreme 1e308 weights never overflow sum_light/tau_new to +Inf. */
    const uint32_t k = vo->k;
    uint32_t n_heavy = vo->n_heavy;
    const uint32_t prev_n_light = k - n_heavy;
    const double old_tau = vo->tau;

    const int use_scale = (!isfinite(vo->total_weight) || vo->total_weight > 1e200 ||
                           weight > 1e200 || old_tau > 1e200);
    const double sc = use_scale ? 1e-150 : 1.0;
    const double inv_sc = use_scale ? 1e150 : 1.0;

    const double w_s = weight * sc;
    const double old_tau_s = old_tau * sc;

    double sum_light_s = (double)prev_n_light * old_tau_s;
    uint32_t cnt_light = prev_n_light;
    int new_in_light = 0;

    if (prev_n_light > 0U && w_s < old_tau_s) {
        new_in_light = 1;
        sum_light_s += w_s;
        cnt_light++;
    }

    double tau_new_s = old_tau_s;
    for (;;) {
        const double cand_tau_s = (cnt_light >= 2U)
            ? (sum_light_s / (double)(cnt_light - 1U))
            : INFINITY;

        double min_h_s = INFINITY;
        int from_new = 0;
        if (n_heavy > 0U) {
            const double hs = vo->items[0].orig_weight * sc;
            if (hs < min_h_s) {
                min_h_s = hs;
                from_new = 0;
            }
        }
        if (!new_in_light && w_s < min_h_s) {
            min_h_s = w_s;
            from_new = 1;
        }

        if (cnt_light >= 2U && min_h_s >= cand_tau_s) {
            tau_new_s = cand_tau_s;
            break;
        }

        if (from_new) {
            new_in_light = 1;
            sum_light_s += w_s;
            cnt_light++;
        } else {
            /* Demote smallest heavy item into items[n_heavy - 1] (light partition) */
            eproc_varopt_item_t popped = vo->items[0];
            n_heavy--;
            if (n_heavy > 0U) {
                vo->items[0] = vo->items[n_heavy];
                eproc_varopt_sift_down(vo->items, n_heavy, 0U);
            }
            vo->items[n_heavy] = popped;
            sum_light_s += popped.orig_weight * sc;
            cnt_light++;
        }
    }

    if (tau_new_s < old_tau_s) {
        tau_new_s = old_tau_s;
    }

    /* Draw one item from L' to drop with exact probability q_i = 1 - c_i / tau_new */
    const double u = eproc_rng_uniform01(vo->rng_state);
    incoming.rand_u = u;

    int drop_new = 0;
    uint32_t drop_idx = k; /* sentinel */
    double rem_u = u;

    const uint32_t old_light_start = k - prev_n_light;
    if (prev_n_light > 0U && tau_new_s > 0.0) {
        double p_each = 1.0 - (old_tau_s / tau_new_s);
        if (p_each < 0.0) {
            p_each = 0.0;
        }
        const double p_old_block = (double)prev_n_light * p_each;
        if (rem_u < p_old_block && p_old_block > 0.0) {
            uint32_t offset = (uint32_t)floor((rem_u / p_old_block) * (double)prev_n_light);
            if (offset >= prev_n_light) {
                offset = prev_n_light - 1U;
            }
            drop_idx = old_light_start + offset;
        } else {
            rem_u -= p_old_block;
        }
    }

    if (drop_idx == k && tau_new_s > 0.0) {
        /* Check newly demoted heavy items at [n_heavy .. old_light_start - 1] */
        for (uint32_t j = n_heavy; j < old_light_start; ++j) {
            double q_j = 1.0 - ((vo->items[j].orig_weight * sc) / tau_new_s);
            if (q_j < 0.0) {
                q_j = 0.0;
            }
            if (rem_u < q_j) {
                drop_idx = j;
                break;
            }
            rem_u -= q_j;
        }
    }

    if (drop_idx == k) {
        if (new_in_light) {
            drop_new = 1;
        } else {
            /* FP rounding residual: drop last valid light element in [n_heavy .. k-1] */
            drop_idx = k - 1U;
        }
    }

    if (new_in_light) {
        if (!drop_new && drop_idx < k) {
            vo->items[drop_idx] = incoming;
        }
    } else {
        /* incoming is heavy -> move items[n_heavy] into dropped light slot and push to heap */
        if (drop_idx != n_heavy) {
            vo->items[drop_idx] = vo->items[n_heavy];
        }
        vo->items[n_heavy] = incoming;
        vo->items[n_heavy].adj_weight = weight;
        n_heavy++;
        eproc_varopt_sift_up(vo->items, n_heavy - 1U);
    }

    double tau_new = use_scale ? (tau_new_s * inv_sc) : tau_new_s;
    if (!isfinite(tau_new)) {
        tau_new = DBL_MAX;
    }

    vo->n_heavy = n_heavy;
    vo->tau = tau_new;

    /* Maintain exact Horvitz-Thompson adjusted weight = max(orig_weight, tau) on all light items */
    for (uint32_t i = n_heavy; i < k; ++i) {
        vo->items[i].adj_weight = tau_new;
    }

    return EPROC_OK;
}

double eproc_varopt_estimate_sum(const eproc_varopt_t *vo) {
    if (vo == NULL || vo->count == 0U) {
        return 0.0;
    }
    double sum = 0.0;
    double comp = 0.0;
    for (uint32_t i = 0; i < vo->count; ++i) {
        const double v = vo->items[i].value;
        if (v == 0.0) {
            continue; /* never evaluate (+Inf) * 0.0 = NaN */
        }
        const double prod = vo->items[i].adj_weight * v;
        if (isfinite(prod)) {
            eproc_kahan_add(&sum, &comp, prod);
        } else if (!isnan(prod)) {
            sum = prod;
            comp = 0.0;
        }
    }
    return eproc_kahan_total(sum, comp);
}

double eproc_varopt_estimate_mean(const eproc_varopt_t *vo) {
    if (vo == NULL || vo->count == 0U) {
        return 0.0;
    }
    const double tot_w = eproc_kahan_total(vo->total_weight, vo->total_weight_comp);
    const double est_s = eproc_varopt_estimate_sum(vo);
    if (isfinite(tot_w) && tot_w > 0.0 && isfinite(est_s)) {
        return est_s / tot_w;
    }
    /* Overflow-safe rescaled weighted mean when total_weight or sum exceeded DBL_MAX */
    double max_w = 0.0;
    for (uint32_t i = 0; i < vo->count; ++i) {
        const double w = isfinite(vo->items[i].adj_weight) ? vo->items[i].adj_weight : 0.0;
        if (w > max_w) {
            max_w = w;
        }
    }
    if (!(max_w > 0.0)) {
        return 0.0;
    }
    double num = 0.0, num_c = 0.0, den = 0.0, den_c = 0.0;
    for (uint32_t i = 0; i < vo->count; ++i) {
        const double w = isfinite(vo->items[i].adj_weight) ? (vo->items[i].adj_weight / max_w) : 0.0;
        eproc_kahan_add(&den, &den_c, w);
        if (vo->items[i].value != 0.0) {
            eproc_kahan_add(&num, &num_c, w * vo->items[i].value);
        }
    }
    const double den_tot = eproc_kahan_total(den, den_c);
    return (den_tot > 0.0) ? (eproc_kahan_total(num, num_c) / den_tot) : 0.0;
}

/* ============================================================================
 * 4B. ONLINE RIDGE LEVERAGE SCORE (RLS) CAUSAL SUBSAMPLER (AVX2 BIT-EXACT)
 * ============================================================================ */

static double eproc_dot_4lane(const double *a, const double *b, uint32_t n) {
#if defined(__AVX2__)
    __m256d vacc = _mm256_setzero_pd();
    uint32_t j = 0;
    for (; j + 4U <= n; j += 4U) {
        __m256d va = _mm256_loadu_pd(a + j);
        __m256d vb = _mm256_loadu_pd(b + j);
        vacc = _mm256_add_pd(vacc, _mm256_mul_pd(va, vb));
    }
    double lane[4];
    _mm256_storeu_pd(lane, vacc);
    double sum = (lane[0] + lane[1]) + (lane[2] + lane[3]);
    for (; j < n; ++j) {
        sum += a[j] * b[j];
    }
    return sum;
#else
    double lane[4] = {0.0, 0.0, 0.0, 0.0};
    uint32_t j = 0;
    for (; j + 4U <= n; j += 4U) {
        lane[0] += a[j + 0U] * b[j + 0U];
        lane[1] += a[j + 1U] * b[j + 1U];
        lane[2] += a[j + 2U] * b[j + 2U];
        lane[3] += a[j + 3U] * b[j + 3U];
    }
    double sum = (lane[0] + lane[1]) + (lane[2] + lane[3]);
    for (; j < n; ++j) {
        sum += a[j] * b[j];
    }
    return sum;
#endif
}

static void eproc_rls_rank1_sub(double *inv_a, const double *wv, uint32_t d) {
    for (uint32_t i = 0; i < d; ++i) {
        double *row = inv_a + (size_t)i * d;
        const double wi = wv[i];
        uint32_t j = 0;
#if defined(__AVX2__)
        const __m256d vwi = _mm256_set1_pd(wi);
        for (; j + 4U <= d; j += 4U) {
            __m256d vr = _mm256_loadu_pd(row + j);
            __m256d vw = _mm256_loadu_pd(wv + j);
            vr = _mm256_sub_pd(vr, _mm256_mul_pd(vwi, vw));
            _mm256_storeu_pd(row + j, vr);
        }
#endif
        for (; j < d; ++j) {
            row[j] -= wi * wv[j];
        }
        if (row[i] < 0.0) {
            row[i] = 0.0;
        }
    }
}

int eproc_rls_init(eproc_rls_t *rls,
                   uint32_t d,
                   double lambda,
                   double oversample,
                   double min_prob,
                   uint64_t seed) {
    if (rls == NULL || d == 0U || d > EPROC_RLS_MAX_D) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(lambda) || !(lambda > 0.0)) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(oversample) || !(oversample > 0.0)) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(min_prob) || !(min_prob > 0.0 && min_prob <= 1.0)) {
        return EPROC_ERR_INVALID;
    }

    memset(rls, 0, sizeof(*rls));
    rls->d = d;
    rls->lambda = lambda;
    rls->oversample = oversample;
    rls->min_prob = min_prob;
    rls->n_seen = 0;
    rls->n_kept = 0;
    eproc_rng_seed(rls->rng_state, seed);

    const double inv_lam = 1.0 / lambda;
    for (uint32_t i = 0; i < d; ++i) {
        rls->inv_a[(size_t)i * d + i] = inv_lam;
    }
    return EPROC_OK;
}

int eproc_rls_step(eproc_rls_t *rls, const double *x, eproc_rls_out_t *out) {
    if (rls == NULL || x == NULL || rls->d == 0U || rls->d > EPROC_RLS_MAX_D) {
        return EPROC_ERR_INVALID;
    }

    const uint32_t d = rls->d;
    double max_abs = 0.0;
    for (uint32_t i = 0; i < d; ++i) {
        if (!isfinite(x[i])) {
            if (out != NULL) {
                memset(out, 0, sizeof(*out));
            }
            return EPROC_OK;
        }
        const double a = fabs(x[i]);
        if (a > max_abs) {
            max_abs = a;
        }
    }

    double x_scaled[EPROC_RLS_MAX_D];
    double v[EPROC_RLS_MAX_D];
    int scaled = 0;
    double scale = 1.0;

    /* W05 fix: treat subnormal / ultra-tiny vectors (< 1e-150) as exact zero leverage
     * so 1.0 / max_abs NEVER overflows to +Inf and produces 0 * +Inf = NaN. */
    if (max_abs < 1e-150) {
        max_abs = 0.0;
        for (uint32_t i = 0; i < d; ++i) {
            v[i] = 0.0;
        }
    } else if (max_abs > 1e100 || max_abs < 1e-100) {
        scaled = 1;
        scale = max_abs;
        for (uint32_t i = 0; i < d; ++i) {
            x_scaled[i] = x[i] / scale;
        }
        for (uint32_t i = 0; i < d; ++i) {
            v[i] = eproc_dot_4lane(rls->inv_a + (size_t)i * d, x_scaled, d);
        }
    } else {
        for (uint32_t i = 0; i < d; ++i) {
            v[i] = eproc_dot_4lane(rls->inv_a + (size_t)i * d, x, d);
        }
    }

    double lev_base = 0.0;
    if (max_abs > 0.0) {
        lev_base = eproc_dot_4lane(scaled ? x_scaled : x, v, d);
        if (!isfinite(lev_base) || lev_base < 0.0) {
            lev_base = 0.0;
        }
    }

    double leverage = scaled ? ((lev_base * scale) * scale) : lev_base;
    if (isnan(leverage) || leverage < 0.0) {
        leverage = 0.0;
    } else if (!isfinite(leverage)) {
        leverage = INFINITY;
    }

    double prob = isfinite(leverage) ? (rls->oversample * leverage) : 1.0;
    prob = eproc_clamp(prob, rls->min_prob, 1.0);

    const double u = eproc_rng_uniform01(rls->rng_state);
    const int32_t keep = (u < prob) ? 1 : 0;
    const double ht_weight = keep ? (1.0 / prob) : 0.0;

    rls->n_seen++;
    if (keep) {
        rls->n_kept++;
        if (max_abs > 0.0 && lev_base > 0.0) {
            double denom;
            if (scaled) {
                const double inv_s = 1.0 / scale;
                const double inv_s2 = inv_s * inv_s;
                denom = inv_s2 + ht_weight * lev_base;
            } else {
                denom = 1.0 + ht_weight * lev_base;
            }
            if (isfinite(denom) && denom > 0.0) {
                const double coeff = ht_weight / denom;
                if (isfinite(coeff) && coeff > 0.0) {
                    const double root_c = sqrt(coeff);
                    double wv[EPROC_RLS_MAX_D];
                    for (uint32_t i = 0; i < d; ++i) {
                        wv[i] = root_c * v[i];
                    }
                    eproc_rls_rank1_sub(rls->inv_a, wv, d);
                }
            }
        }
    }

    if (out != NULL) {
        out->keep = keep;
        out->_pad0 = 0;
        out->leverage = isfinite(leverage) ? leverage : 1e300;
        out->prob = prob;
        out->ht_weight = ht_weight;
    }
    return EPROC_OK;
}

/* ============================================================================
 * 4C. HIERARCHICAL MULTI-RESOLUTION BAYES VOI SELECTOR (SIFT)
 *     WITH EXACT GAUSSIAN MAXIMUM-ENTROPY BRANCH-AND-BOUND (30x-100x FASTER)
 * ============================================================================ */

/* Exact binary entropy H_2(p) in nats */
static inline double eproc_bin_entropy(double p) {
    const double pc = eproc_clamp(p, 1e-15, 1.0 - 1e-15);
    return -(pc * log(pc) + (1.0 - pc) * log(1.0 - pc));
}

/* Exact Mutual Information I(Z; Y) in nats for Z ~ Bern(p), Y | Z ~ N(Z*d, 1)
 * via 5-point Gauss-Hermite quadrature, tightly bounded by the exact Shannon
 * Gaussian channel capacity 0.5 * log1p(p*(1-p)*d^2) to guarantee exact
 * branch-and-bound soundness. */
static double eproc_gauss_mi(double p, double d) {
    if (!(d > 1e-12) || !(p > 1e-12 && p < 1.0 - 1e-12)) {
        return 0.0;
    }
    static const double GH_X[5] = {
        -2.8569700138728056,
        -1.3556261799742659,
         0.0,
         1.3556261799742659,
         2.8569700138728056
    };
    static const double GH_W[5] = {
        0.011257411327720689,
        0.22207592200561264,
        0.53333333333333333,
        0.22207592200561264,
        0.011257411327720689
    };

    const double h_prior = eproc_bin_entropy(p);
    const double lo = log(p / (1.0 - p));
    const double half_d = 0.5 * d;
    double eh_post = 0.0;

    for (int i = 0; i < 5; ++i) {
        const double x = GH_X[i];
        const double p1 = eproc_sigmoid(lo + d * (x + half_d));
        const double p0 = eproc_sigmoid(lo + d * (x - half_d));
        eh_post += GH_W[i] * (p * eproc_bin_entropy(p1) + (1.0 - p) * eproc_bin_entropy(p0));
    }
    double mi = h_prior - eh_post;
    if (mi <= 0.0) {
        return 0.0;
    }
    const double cap = 0.5 * log1p(p * (1.0 - p) * d * d);
    return (mi < cap) ? mi : cap;
}

static inline double eproc_sift_site_sum_var(const eproc_sift_t *sf, uint32_t site_idx) {
    const uint32_t m = sf->per_site;
    const uint32_t base = site_idx * m;
    double sum_ns2 = 0.0;
    for (uint32_t j = 0; j < m; ++j) {
        const double sc = sf->chan_noise_scale[base + j];
        sum_ns2 += sc * sc;
    }
    return sum_ns2 * (sf->noise_sd * sf->noise_sd);
}

static void eproc_sift_site_params_from_sumvar(const eproc_sift_t *sf,
                                               double sum_var,
                                               double *mu0_out,
                                               double *var0_out,
                                               double *mu1_out,
                                               double *var1_out) {
    const double inv_m = 1.0 / (double)sf->per_site;
    const double noise_var_mean = sum_var * inv_m * inv_m;
    const double mu = sf->signal_mean;
    const double mu2 = mu * mu;

    const double pc = sf->prior_active_cold;
    const double ph = sf->prior_active_hot;

    *mu0_out = pc * mu;
    *mu1_out = ph * mu;
    *var0_out = fmax(noise_var_mean + pc * (1.0 - pc) * mu2 * inv_m, 1e-12);
    *var1_out = fmax(noise_var_mean + ph * (1.0 - ph) * mu2 * inv_m, 1e-12);
}

int eproc_sift_init(eproc_sift_t *sf,
                    uint32_t n_sites,
                    uint32_t per_site,
                    double signal_mean,
                    double noise_sd) {
    if (sf == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (n_sites == 0U || n_sites > EPROC_SIFT_MAX_SITES) {
        return EPROC_ERR_INVALID;
    }
    if (per_site == 0U || per_site > EPROC_SIFT_MAX_PER_SITE) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(signal_mean) || fabs(signal_mean) <= 0.0) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(noise_sd) || !(noise_sd > 0.0)) {
        return EPROC_ERR_INVALID;
    }

    memset(sf, 0, sizeof(*sf));
    sf->n_sites = n_sites;
    sf->per_site = per_site;
    sf->prior_hot_site = 0.05;
    sf->prior_active_cold = 0.005;
    sf->prior_active_hot = 0.30;
    sf->signal_mean = signal_mean;
    sf->noise_sd = noise_sd;

    const double log_p_hot = log(sf->prior_hot_site);
    const double p_chan_0 = sf->prior_hot_site * sf->prior_active_hot
                          + (1.0 - sf->prior_hot_site) * sf->prior_active_cold;

    for (uint32_t s = 0; s < n_sites; ++s) {
        sf->site_log_prob[s] = log_p_hot;
    }
    const uint32_t total_ch = n_sites * per_site;
    for (uint32_t c = 0; c < total_ch; ++c) {
        sf->chan_prob[c] = p_chan_0;
        sf->chan_noise_scale[c] = 1.0;
    }
    return EPROC_OK;
}

int eproc_sift_set_noise_scale(eproc_sift_t *sf,
                               uint32_t channel_idx,
                               double noise_scale) {
    if (sf == NULL || channel_idx >= sf->n_sites * sf->per_site) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(noise_scale) || !(noise_scale > 0.0)) {
        return EPROC_ERR_INVALID;
    }
    sf->chan_noise_scale[channel_idx] = noise_scale;
    return EPROC_OK;
}

int eproc_sift_observe_channel(eproc_sift_t *sf,
                               uint32_t channel_idx,
                               double obs_mean,
                               uint32_t n_ticks) {
    if (sf == NULL || channel_idx >= sf->n_sites * sf->per_site || n_ticks == 0U) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(obs_mean)) {
        return EPROC_OK;
    }

    const double sd = sf->noise_sd * sf->chan_noise_scale[channel_idx];
    const double var_obs = fmax((sd * sd) / (double)n_ticks, 1e-18);
    const double mu = sf->signal_mean;

    /* Exact Gaussian LLR = log(p(obs | Z=1) / p(obs | Z=0)) */
    const double llr = eproc_clamp((mu * (obs_mean - 0.5 * mu)) / var_obs, -500.0, 500.0);

    const double lo_new = eproc_logit(sf->chan_prob[channel_idx]) + llr;
    sf->chan_prob[channel_idx] = eproc_clamp(eproc_sigmoid(lo_new), 1e-9, 1.0 - 1e-9);

    /* Propagate channel evidence upward to parent site posterior */
    const uint32_t site_idx = channel_idx / sf->per_site;
    const double lp_hot_act = eproc_logaddexp(log(sf->prior_active_hot) + llr,
                                              log(1.0 - sf->prior_active_hot));
    const double lp_cold_act = eproc_logaddexp(log(sf->prior_active_cold) + llr,
                                               log(1.0 - sf->prior_active_cold));
    const double site_llr = lp_hot_act - lp_cold_act;

    const double old_p_hot = eproc_clamp(exp(sf->site_log_prob[site_idx]), 1e-9, 1.0 - 1e-9);
    const double new_p_hot = eproc_clamp(eproc_sigmoid(eproc_logit(old_p_hot) + site_llr),
                                         1e-9, 1.0 - 1e-9);
    sf->site_log_prob[site_idx] = log(new_p_hot);

    return EPROC_OK;
}

int eproc_sift_observe_site(eproc_sift_t *sf,
                            uint32_t site_idx,
                            double site_mean) {
    if (sf == NULL || site_idx >= sf->n_sites) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(site_mean)) {
        return EPROC_OK;
    }

    double mu0, var0, mu1, var1;
    const double sum_var = eproc_sift_site_sum_var(sf, site_idx);
    eproc_sift_site_params_from_sumvar(sf, sum_var, &mu0, &var0, &mu1, &var1);

    const double d0 = site_mean - mu0;
    const double d1 = site_mean - mu1;
    const double ll0 = -0.5 * log(var0) - 0.5 * (d0 * d0) / var0;
    const double ll1 = -0.5 * log(var1) - 0.5 * (d1 * d1) / var1;
    const double llr = eproc_clamp(ll1 - ll0, -500.0, 500.0);

    const double old_p_hot = eproc_clamp(exp(sf->site_log_prob[site_idx]), 1e-9, 1.0 - 1e-9);
    const double new_p_hot = eproc_clamp(eproc_sigmoid(eproc_logit(old_p_hot) + llr),
                                         1e-9, 1.0 - 1e-9);
    sf->site_log_prob[site_idx] = log(new_p_hot);

    /* Coherently propagate site posterior shift down to child channel priors */
    const double p_prior_old = old_p_hot * sf->prior_active_hot
                             + (1.0 - old_p_hot) * sf->prior_active_cold;
    const double p_prior_new = new_p_hot * sf->prior_active_hot
                             + (1.0 - new_p_hot) * sf->prior_active_cold;
    const double delta_lo = eproc_logit(p_prior_new) - eproc_logit(p_prior_old);

    const uint32_t base = site_idx * sf->per_site;
    for (uint32_t j = 0; j < sf->per_site; ++j) {
        const double lo = eproc_logit(sf->chan_prob[base + j]) + delta_lo;
        sf->chan_prob[base + j] = eproc_clamp(eproc_sigmoid(lo), 1e-9, 1.0 - 1e-9);
    }

    return EPROC_OK;
}

static inline void eproc_sift_eval_channel(uint32_t c,
                                           double p_c,
                                           double sd_c,
                                           double abs_mu,
                                           double mu2,
                                           double sqrt_ticks,
                                           double ticks_d,
                                           int ok_chan,
                                           double inv_c_chan,
                                           int ok_tick,
                                           double inv_c_tick,
                                           double *best_score,
                                           int32_t *best_type,
                                           uint32_t *best_target) {
    const double pq = p_c * (1.0 - p_c);
    const double sd_safe = (sd_c > 1e-12) ? sd_c : 1e-12;
    const double snr1_sq = mu2 / (sd_safe * sd_safe);

    if (ok_chan) {
        const double v1 = pq * snr1_sq;
        const double ub1 = 0.5 * v1 * inv_c_chan;
        if (ub1 * (1.0 + 1e-12) > *best_score) {
            const double ub1_log = 0.5 * log1p(v1) * inv_c_chan;
            if (ub1_log * (1.0 + 1e-12) > *best_score) {
                const double snr1 = abs_mu / sd_safe;
                const double score = eproc_gauss_mi(p_c, snr1) * inv_c_chan;
                if (score > *best_score) {
                    *best_score = score;
                    *best_type = 1;
                    *best_target = c;
                }
            }
        }
    }
    if (ok_tick) {
        const double v2 = pq * (snr1_sq * ticks_d);
        const double ub2 = 0.5 * v2 * inv_c_tick;
        if (ub2 * (1.0 + 1e-12) > *best_score) {
            const double ub2_log = 0.5 * log1p(v2) * inv_c_tick;
            if (ub2_log * (1.0 + 1e-12) > *best_score) {
                const double h_ub = eproc_bin_entropy(p_c) * inv_c_tick;
                if (h_ub * (1.0 + 1e-12) > *best_score) {
                    const double snr2 = (abs_mu / sd_safe) * sqrt_ticks;
                    const double score = eproc_gauss_mi(p_c, snr2) * inv_c_tick;
                    if (score > *best_score) {
                        *best_score = score;
                        *best_type = 2;
                        *best_target = c;
                    }
                }
            }
        }
    }
}

int eproc_sift_select_next(const eproc_sift_t *sf,
                           double cost_site,
                           double cost_channel,
                           double cost_ticks,
                           uint32_t ticks_per_read,
                           eproc_sift_action_t *out) {
    if (sf == NULL || out == NULL || ticks_per_read == 0U) {
        return EPROC_ERR_INVALID;
    }
    const int ok_site = isfinite(cost_site) && cost_site > 0.0;
    const int ok_chan = isfinite(cost_channel) && cost_channel > 0.0;
    const int ok_tick = isfinite(cost_ticks) && cost_ticks > 0.0;
    if (!ok_site && !ok_chan && !ok_tick) {
        return EPROC_ERR_INVALID;
    }

    int32_t best_type = ok_site ? 0 : (ok_chan ? 1 : 2);
    uint32_t best_target = 0U;
    double best_score = -1.0;

    const double abs_mu = fabs(sf->signal_mean);
    const double mu2 = abs_mu * abs_mu;
    const double ticks_d = (double)ticks_per_read;
    const double sqrt_ticks = sqrt(ticks_d);
    const double h_act_hot = eproc_bin_entropy(sf->prior_active_hot);
    const double h_act_cold = eproc_bin_entropy(sf->prior_active_cold);

    /* Action type 0: coarse site-mean reads with exact cache + Gaussian capacity B&B */
    if (ok_site) {
        const double inv_cost = 1.0 / cost_site;
        const double m = (double)sf->per_site;
        const double d_mu_sq = (sf->prior_active_hot - sf->prior_active_cold)
                             * (sf->prior_active_hot - sf->prior_active_cold) * mu2;

        /* Remember first/prior evaluated site state to skip identical cold sites in O(1) */
        int has_cache = 0;
        double cache_lp = 0.0;
        double cache_sumvar = 0.0;

        for (uint32_t s = 0; s < sf->n_sites; ++s) {
            const double lp = sf->site_log_prob[s];
            const double sum_var = eproc_sift_site_sum_var(sf, s);
            if (has_cache && lp == cache_lp && sum_var == cache_sumvar) {
                /* Identical (lp, sum_var) has exact same score <= best_score (ties keep lower index) */
                continue;
            }

            const double p_hot = eproc_clamp(exp(lp), 1e-9, 1.0 - 1e-9);
            double mu0, var0, mu1, var1;
            eproc_sift_site_params_from_sumvar(sf, sum_var, &mu0, &var0, &mu1, &var1);
            const double var_eff = fmax(0.5 * (var0 + var1), 1e-24);
            const double d_site_sq = d_mu_sq / var_eff;

            /* Exact upper bound: M * (I(Z_c; H_s) / H_2(H_s)) * I(H_s; Y_s)
             * <= M * I(H_s; Y_s) <= M * 0.5 * p_hot * (1 - p_hot) * d_site^2 */
            const double ub_site = m * 0.5 * p_hot * (1.0 - p_hot) * d_site_sq * inv_cost;
            if (ub_site * (1.0 + 1e-12) <= best_score) {
                continue;
            }

            const double h_site = eproc_bin_entropy(p_hot);
            if (!(h_site > 1e-12)) {
                continue;
            }

            const double p_ch_site = p_hot * sf->prior_active_hot
                                   + (1.0 - p_hot) * sf->prior_active_cold;
            const double h_ch_cond = p_hot * h_act_hot + (1.0 - p_hot) * h_act_cold;
            double mi_ch_given_h = eproc_bin_entropy(p_ch_site) - h_ch_cond;
            if (mi_ch_given_h <= 0.0) {
                continue;
            }

            const double ub_tight = m * mi_ch_given_h
                                  * fmin(1.0, 0.5 * log1p(p_hot * (1.0 - p_hot) * d_site_sq) / h_site)
                                  * inv_cost;
            if (ub_tight * (1.0 + 1e-12) <= best_score) {
                continue;
            }

            const double d_site = sqrt(d_site_sq);
            const double mi_site = eproc_gauss_mi(p_hot, d_site);
            const double evoi = m * mi_ch_given_h * (mi_site / h_site);
            const double score = evoi * inv_cost;
            if (!has_cache) {
                has_cache = 1;
                cache_lp = lp;
                cache_sumvar = sum_var;
            }
            if (score > best_score) {
                best_score = score;
                best_type = 0;
                best_target = s;
                cache_lp = lp;
                cache_sumvar = sum_var;
            }
        }
    }

    /* Action types 1 & 2: Exact Gaussian Max-Entropy Branch-and-Bound with AVX2 4-wide pre-filter */
    if (ok_chan || ok_tick) {
        const uint32_t total_ch = sf->n_sites * sf->per_site;
        const double inv_c_chan = ok_chan ? (1.0 / cost_channel) : 0.0;
        const double inv_c_tick = ok_tick ? (1.0 / cost_ticks) : 0.0;
        const double eff_inv = (ticks_d * inv_c_tick > inv_c_chan)
            ? (ticks_d * inv_c_tick)
            : inv_c_chan;
        const double noise_sd = sf->noise_sd;
        const double ns_var = fmax(noise_sd * noise_sd, 1e-24);
        /* Unified sound upper bound for both 1-tick & T-tick:
         * max(ub1, ub2) <= (0.5 * mu^2 * eff_inv / noise_sd^2) * p*(1-p) / chan_noise_scale^2 */
        const double coeff_ub = 0.5 * (mu2 / ns_var) * eff_inv * (1.0 + 1e-12);

        uint32_t c = 0U;
#if defined(__AVX2__)
        const __m256d v_one = _mm256_set1_pd(1.0);
        const __m256d v_coeff = _mm256_set1_pd(coeff_ub);
        const __m256d v_min_ns2 = _mm256_set1_pd(1e-24);
        __m256d v_best = _mm256_set1_pd(best_score);

        for (; c + 4U <= total_ch; c += 4U) {
            const __m256d vp = _mm256_loadu_pd(sf->chan_prob + c);
            const __m256d vns = _mm256_loadu_pd(sf->chan_noise_scale + c);
            const __m256d vpq = _mm256_mul_pd(vp, _mm256_sub_pd(v_one, vp));
            const __m256d vns2 = _mm256_max_pd(_mm256_mul_pd(vns, vns), v_min_ns2);
            const __m256d vub = _mm256_div_pd(_mm256_mul_pd(v_coeff, vpq), vns2);
            const int mask = _mm256_movemask_pd(_mm256_cmp_pd(vub, v_best, _CMP_GT_OQ));
            if (mask == 0) {
                continue;
            }
            for (uint32_t lane = 0U; lane < 4U; ++lane) {
                if ((mask & (1 << lane)) == 0) {
                    continue;
                }
                const uint32_t idx = c + lane;
                const double prev_b = best_score;
                eproc_sift_eval_channel(idx, sf->chan_prob[idx],
                                        noise_sd * sf->chan_noise_scale[idx],
                                        abs_mu, mu2, sqrt_ticks, ticks_d,
                                        ok_chan, inv_c_chan, ok_tick, inv_c_tick,
                                        &best_score, &best_type, &best_target);
                if (best_score > prev_b) {
                    v_best = _mm256_set1_pd(best_score);
                }
            }
        }
#endif
        for (; c < total_ch; ++c) {
            const double p_c = sf->chan_prob[c];
            const double ns = sf->chan_noise_scale[c];
            const double ns2 = fmax(ns * ns, 1e-24);
            const double ub = (coeff_ub * (p_c * (1.0 - p_c))) / ns2;
            if (ub <= best_score) {
                continue;
            }
            eproc_sift_eval_channel(c, p_c, noise_sd * ns,
                                    abs_mu, mu2, sqrt_ticks, ticks_d,
                                    ok_chan, inv_c_chan, ok_tick, inv_c_tick,
                                    &best_score, &best_type, &best_target);
        }
    }

    out->action_type = best_type;
    out->target_idx = best_target;
    out->evoi_per_cost = (best_score > 0.0) ? best_score : 0.0;
    return EPROC_OK;
}

/* ============================================================================
 * 4D. ABI VERSION & RUNTIME SIMD CAPABILITY FLAGS
 * ============================================================================ */

uint32_t eproc_abi_version(void) {
    return EPROC_ABI_VERSION;
}

uint32_t eproc_simd_flags(void) {
    uint32_t flags = 0U;
#if defined(__AVX2__)
    flags |= 1U;
#endif
#if defined(__FMA__)
    flags |= 2U;
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
    flags |= 4U;
#endif
    return flags;
}
