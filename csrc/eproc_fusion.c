#if !defined(__GNUC__) || defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

#include "eproc.h"
#include "eproc_internal.h"
#include "eproc_seqlock.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

EPROC_STATIC_ASSERT((EPROC_ECHO_TABLE_CAP & (EPROC_ECHO_TABLE_CAP - 1U)) == 0U,
                    "EPROC_ECHO_TABLE_CAP must be a power of 2");

/* ============================================================================
 * 3A. SCALE-FREE BOA + FIXED-SHARE + 3-GATE VILLE E-PROCESS + BOEHM-2012 SEQLOCK
 * ============================================================================ */

static void eproc_boa_publish_seqlock(eproc_boa_t *boa) {
    const uint32_t k = boa->n_experts;
    uint64_t snap[EPROC_BOA_MAX_EXPERTS];

    if (boa->fallback_active) {
        for (uint32_t i = 0; i < k; ++i) {
            const double w = (i == boa->anchor_idx) ? 1.0 : 0.0;
            snap[i] = eproc_f64_to_u64(w);
        }
    } else {
        for (uint32_t i = 0; i < k; ++i) {
            snap[i] = eproc_f64_to_u64(boa->weights[i]);
        }
    }
    const uint64_t meta =
        ((uint64_t)k << 32) | (uint64_t)(uint32_t)boa->fallback_active;
    const uint64_t n_upd = boa->n_updates;

    const uint64_t odd = eproc_seqlock_write_begin_u64(&boa->seqlock_seq);
    eproc_seqlock_store_words_u64(boa->seqlock_words, snap, k);
    eproc_seqlock_store_words_u64(&boa->seqlock_words[EPROC_BOA_MAX_EXPERTS], &meta, 1U);
    eproc_seqlock_store_words_u64(&boa->seqlock_words[EPROC_BOA_MAX_EXPERTS + 1U], &n_upd, 1U);
    eproc_seqlock_write_end_u64(&boa->seqlock_seq, odd);
}

static void eproc_sort_small(double *arr, uint32_t n) {
    for (uint32_t i = 1; i < n; ++i) {
        double key = arr[i];
        uint32_t j = i;
        while (j > 0 && arr[j - 1] > key) {
            arr[j] = arr[j - 1];
            j--;
        }
        arr[j] = key;
    }
}

/* Compute robust shadow BOA prediction (immune to tick-0 and steady-state Byzantine spikes) */
static double eproc_boa_shadow_predict(const eproc_boa_t *boa,
                                       const double *expert_preds,
                                       double *anchor_out) {
    const uint32_t k = boa->n_experts;
    double anc = expert_preds[boa->anchor_idx];

    if (!isfinite(anc)) {
        /* Anchor itself is non-finite: fall back to median of finite non-quarantined experts */
        double finite_vals[EPROC_BOA_MAX_EXPERTS];
        uint32_t n_fin = 0;
        for (uint32_t i = 0; i < k; ++i) {
            if (!boa->quarantined[i] && isfinite(expert_preds[i])) {
                finite_vals[n_fin++] = expert_preds[i];
            }
        }
        if (n_fin == 0U) {
            if (anchor_out != NULL) {
                *anchor_out = 0.0;
            }
            return 0.0;
        }
        eproc_sort_small(finite_vals, n_fin);
        anc = (n_fin & 1U)
            ? finite_vals[n_fin >> 1]
            : (0.5 * finite_vals[(n_fin >> 1) - 1U] + 0.5 * finite_vals[n_fin >> 1]);
    }

    if (anchor_out != NULL) {
        *anchor_out = anc;
    }

    /* Reference and strictly F_{t-1}-predictable poisoning gate anchored on trusted anchor_pred.
     * On tick 0 (n_updates == 0), NEVER allow unauthenticated non-anchor experts to inflate
     * pred_scale via MAD (prevents tick-0 Byzantine spread attacks like [1e12, 2e12, 3e12, 4e12]). */
    const double ref = anc;
    double pred_scale;
    if (boa->n_updates == 0ULL) {
        pred_scale = fmax(1.0, fabs(ref) * 0.1);
    } else {
        pred_scale = fmax(boa->scale_ema, 1e-6);
    }
    const double max_dev = boa->clip_mult * pred_scale;

    double w_sum = 0.0;
    double pred_sum = 0.0;
    double pred_comp = 0.0;

    for (uint32_t i = 0; i < k; ++i) {
        if (boa->quarantined[i]) {
            continue;
        }
        const double p = expert_preds[i];
        if (!isfinite(p)) {
            continue;
        }
        const double dev = fabs(p - ref);
        if (i != boa->anchor_idx && (!isfinite(dev) || dev > max_dev)) {
            /* Single-tick Byzantine spike suppression -> 0.00% blast radius */
            continue;
        }
        const double w = boa->weights[i];
        if (w > 0.0) {
            w_sum += w;
            eproc_kahan_add(&pred_sum, &pred_comp, w * p);
        }
    }

    if (w_sum > 0.0) {
        const double res = eproc_kahan_total(pred_sum, pred_comp) / w_sum;
        if (isfinite(res)) {
            return res;
        }
    }
    return anc;
}

int eproc_boa_init(eproc_boa_t *boa,
                   uint32_t n_experts,
                   uint32_t anchor_idx,
                   double alpha_deg) {
    if (boa == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (n_experts == 0 || n_experts > EPROC_BOA_MAX_EXPERTS) {
        return EPROC_ERR_INVALID;
    }
    if (anchor_idx >= n_experts) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(alpha_deg) || !(alpha_deg > 0.0 && alpha_deg < 1.0)) {
        return EPROC_ERR_INVALID;
    }

    memset(boa, 0, sizeof(*boa));
    boa->n_experts = n_experts;
    boa->anchor_idx = anchor_idx;
    boa->fallback_active = 0;
    boa->scale_ema = 1.0;
    boa->clip_mult = 25.0;
    boa->n_updates = 0;
    boa->n_poison_clipped = 0;
    boa->n_fallback_trips = 0;

    const double w0 = 1.0 / (double)n_experts;
    for (uint32_t i = 0; i < n_experts; ++i) {
        boa->weights[i] = w0;
        boa->log_w[i] = 0.0;
        boa->grad_sq_sum[i] = 0.0;
        boa->cum_loss[i] = 0.0;
        boa->quarantined[i] = 0;
        boa->poison_strikes[i] = 0;
    }

    int rc = eproc_bet_init(&boa->sup_e, 0.5, alpha_deg, 0.0, 1.0, EPROC_SIDE_UPPER);
    if (rc != EPROC_OK) return rc;
    rc = eproc_bet_init(&boa->deg_e, 0.5, alpha_deg, 0.0, 1.0, EPROC_SIDE_LOWER);
    if (rc != EPROC_OK) return rc;
    rc = eproc_bet_init(&boa->rec_e, 0.5, alpha_deg, 0.0, 1.0, EPROC_SIDE_UPPER);
    if (rc != EPROC_OK) return rc;

    /* Initialize atomic seqlock fields cleanly before publishing */
    eproc_seqlock_init_u64(&boa->seqlock_seq, boa->seqlock_words, EPROC_BOA_MAX_EXPERTS + 2U);
    eproc_boa_publish_seqlock(boa);
    return EPROC_OK;
}

double eproc_boa_predict(const eproc_boa_t *boa, const double *expert_preds) {
    if (boa == NULL || expert_preds == NULL || boa->n_experts == 0) {
        return 0.0;
    }
    double anc = 0.0;
    const double shadow = eproc_boa_shadow_predict(boa, expert_preds, &anc);
    return boa->fallback_active ? anc : shadow;
}

int eproc_boa_update(eproc_boa_t *boa,
                     const double *expert_preds,
                     double y_true,
                     eproc_boa_step_t *out) {
    if (boa == NULL || expert_preds == NULL || boa->n_experts == 0) {
        return EPROC_ERR_INVALID;
    }

    const uint32_t k = boa->n_experts;
    double anc_pred = 0.0;
    const double boa_pred = eproc_boa_shadow_predict(boa, expert_preds, &anc_pred);
    const double fused_pred = boa->fallback_active ? anc_pred : boa_pred;

    if (!isfinite(y_true)) {
        if (out != NULL) {
            uint32_t nq = 0;
            for (uint32_t i = 0; i < k; ++i) {
                nq += boa->quarantined[i] ? 1U : 0U;
            }
            out->fused_pred = fused_pred;
            out->anchor_pred = anc_pred;
            out->fallback_active = boa->fallback_active;
            out->n_quarantined = nq;
            out->sup_log_e = boa->sup_e.log_e;
            out->deg_log_e = boa->deg_e.log_e;
            out->n_updates = boa->n_updates;
        }
        return EPROC_OK;
    }

    /* Strictly F_{t-1}-predictable scale for Huber clipping & BOA normalization */
    const double anc_diff = fabs(y_true - anc_pred);
    const double prev_scale = (boa->n_updates == 0ULL)
        ? fmax(1.0, fmax(isfinite(anc_diff) ? anc_diff : 1.0, fabs(y_true) * 0.05))
        : fmax(boa->scale_ema, 1e-6);
    const double clip_cap = boa->clip_mult * prev_scale;
    const double extreme_cap = 1e4 * prev_scale;
    const double strike_cap = 15.0 * prev_scale;

    const double anc_loss = isfinite(anc_diff) ? anc_diff : clip_cap;
    const double boa_diff = fabs(y_true - boa_pred);
    const double boa_loss = isfinite(boa_diff) ? boa_diff : clip_cap;

    double eff_loss[EPROC_BOA_MAX_EXPERTS];
    double l_mix = 0.0;
    double w_active_sum = 0.0;

    for (uint32_t i = 0; i < k; ++i) {
        if (boa->quarantined[i]) {
            eff_loss[i] = clip_cap;
            continue;
        }

        const double p = expert_preds[i];
        if (!isfinite(p)) {
            boa->n_poison_clipped++;
            if (i != boa->anchor_idx) {
                boa->quarantined[i] = 1U;
            }
            eff_loss[i] = clip_cap;
            boa->cum_loss[i] += eff_loss[i];
            continue;
        }

        const double raw_err = fabs(y_true - p);
        if (i != boa->anchor_idx && (!isfinite(raw_err) || raw_err > extreme_cap)) {
            /* Immediate quarantine on extreme Byzantine spike (> 1e4 * scale_ema) */
            boa->quarantined[i] = 1U;
            boa->n_poison_clipped++;
            eff_loss[i] = clip_cap;
        } else if (i != boa->anchor_idx && boa->n_updates >= 3ULL && raw_err > strike_cap) {
            boa->poison_strikes[i]++;
            boa->n_poison_clipped++;
            if (boa->poison_strikes[i] >= 3U) {
                boa->quarantined[i] = 1U;
            }
            eff_loss[i] = (raw_err > clip_cap) ? clip_cap : raw_err;
        } else {
            boa->poison_strikes[i] = 0U;
            if (raw_err > clip_cap) {
                boa->n_poison_clipped++;
                eff_loss[i] = clip_cap;
            } else {
                eff_loss[i] = raw_err;
            }
        }

        boa->cum_loss[i] += eff_loss[i];
        if (!boa->quarantined[i]) {
            l_mix += boa->weights[i] * eff_loss[i];
            w_active_sum += boa->weights[i];
        }
    }

    if (w_active_sum > 0.0) {
        l_mix /= w_active_sum;
    } else {
        l_mix = (anc_loss > clip_cap) ? clip_cap : anc_loss;
    }

    /* Scale-free Wintenberger-2017 BOA second-order update with mild variance
     * forgetting + Fixed-Share exploration floor for fast post-shift recovery. */
    const double log_k = log((k > 2U) ? (double)k : 2.0);
    const double inv_scale = 1.0 / prev_scale;
    double max_lw = -INFINITY;
    uint32_t n_active = 0;

    for (uint32_t i = 0; i < k; ++i) {
        if (boa->quarantined[i]) {
            boa->weights[i] = 0.0;
            continue;
        }
        n_active++;
        const double reg = eproc_clamp((l_mix - eff_loss[i]) * inv_scale, -1.0, 1.0);
        boa->grad_sq_sum[i] = 0.98 * boa->grad_sq_sum[i] + reg * reg;
        const double denom_v = (boa->grad_sq_sum[i] > 1.0) ? boa->grad_sq_sum[i] : 1.0;
        double eta = sqrt(log_k / denom_v);
        if (eta > 0.5) {
            eta = 0.5;
        }
        boa->log_w[i] += eta * reg - (eta * eta) * (reg * reg);
        if (boa->log_w[i] > max_lw) {
            max_lw = boa->log_w[i];
        }
    }

    if (n_active == 0) {
        boa->quarantined[boa->anchor_idx] = 0U;
        for (uint32_t i = 0; i < k; ++i) {
            boa->weights[i] = (i == boa->anchor_idx) ? 1.0 : 0.0;
            boa->log_w[i] = (i == boa->anchor_idx) ? 0.0 : -50.0;
        }
    } else {
        double w_norm = 0.0;
        for (uint32_t i = 0; i < k; ++i) {
            if (boa->quarantined[i]) {
                boa->weights[i] = 0.0;
                continue;
            }
            boa->log_w[i] -= max_lw;
            if (boa->log_w[i] < -50.0) {
                boa->log_w[i] = -50.0;
            }
            boa->weights[i] = exp(boa->log_w[i]);
            w_norm += boa->weights[i];
        }
        if (w_norm > 0.0) {
            const double inv_norm = 1.0 / w_norm;
            for (uint32_t i = 0; i < k; ++i) {
                boa->weights[i] *= inv_norm;
            }
            /* Fixed-Share projection onto non-quarantined simplex (eps = 0.01) */
            if (n_active > 1U) {
                const double share_eps = 0.01;
                const double keep_scale = 1.0 - share_eps;
                const double floor_w = share_eps / (double)n_active;
                for (uint32_t i = 0; i < k; ++i) {
                    if (!boa->quarantined[i]) {
                        boa->weights[i] = keep_scale * boa->weights[i] + floor_w;
                        boa->log_w[i] = log(boa->weights[i]);
                    }
                }
            }
        } else {
            for (uint32_t i = 0; i < k; ++i) {
                boa->weights[i] = (i == boa->anchor_idx) ? 1.0 : 0.0;
                boa->log_w[i] = (i == boa->anchor_idx) ? 0.0 : -50.0;
            }
        }
    }

    /* 3-Gate Ville E-processes comparing shadow BOA vs anchor on bounded [0, 1].
     * deg_e and rec_e use Page-CUSUM resetting at 0.0 so sustained pre-shift
     * outperformance never sinks deg_e to -50 and delays post-shift protection. */
    const double c_boa = (boa_loss > clip_cap) ? clip_cap : boa_loss;
    const double c_anc = (anc_loss > clip_cap) ? clip_cap : anc_loss;
    const double norm_denom = fmax(2.0 * prev_scale, 1e-6);
    const double u_t = 0.5 + 0.5 * eproc_clamp((c_anc - c_boa) / norm_denom, -1.0, 1.0);

    eproc_bet_update(&boa->sup_e, u_t, NULL);
    if (!boa->fallback_active) {
        eproc_bet_update(&boa->deg_e, u_t, NULL);
        if (boa->deg_e.log_e_neg < 0.0) {
            boa->deg_e.log_e_neg = 0.0;
            boa->deg_e.log_e = 0.0;
            boa->deg_e.lam_neg = 0.0;
            boa->deg_e.a_neg = 1.0;
        } else if (boa->deg_e.log_e_neg == 0.0 && boa->deg_e.lam_neg == 0.0) {
            boa->deg_e.a_neg = 1.0;
        }
        if (boa->deg_e.log_e >= boa->deg_e.log_thresh) {
            boa->fallback_active = 1;
            boa->n_fallback_trips++;
            eproc_bet_init(&boa->rec_e, 0.5, boa->deg_e.alpha, 0.0, 1.0, EPROC_SIDE_UPPER);
        }
    } else {
        eproc_bet_update(&boa->rec_e, u_t, NULL);
        if (boa->rec_e.log_e_pos < 0.0) {
            boa->rec_e.log_e_pos = 0.0;
            boa->rec_e.log_e = 0.0;
            boa->rec_e.lam_pos = 0.0;
            boa->rec_e.a_pos = 1.0;
        } else if (boa->rec_e.log_e_pos == 0.0 && boa->rec_e.lam_pos == 0.0) {
            boa->rec_e.a_pos = 1.0;
        }
        if (boa->rec_e.log_e >= boa->rec_e.log_thresh) {
            boa->fallback_active = 0;
            eproc_bet_init(&boa->deg_e, 0.5, boa->deg_e.alpha, 0.0, 1.0, EPROC_SIDE_LOWER);
        }
    }

    /* Update F_t scale_ema strictly AFTER using prev_scale */
    const double obs_scale = fmax(fmin(fmax(anc_loss, boa_loss), clip_cap),
                                  fmax(1e-6 * fabs(y_true), 1e-6));
    if (boa->n_updates == 0ULL) {
        boa->scale_ema = obs_scale;
    } else {
        boa->scale_ema = 0.9 * boa->scale_ema + 0.1 * obs_scale;
    }

    boa->n_updates++;

    /* Publish lock-free snapshot for concurrent readers */
    eproc_boa_publish_seqlock(boa);

    if (out != NULL) {
        uint32_t nq = 0;
        for (uint32_t i = 0; i < k; ++i) {
            nq += boa->quarantined[i] ? 1U : 0U;
        }
        out->fused_pred = fused_pred;
        out->anchor_pred = anc_pred;
        out->fallback_active = boa->fallback_active;
        out->n_quarantined = nq;
        out->sup_log_e = boa->sup_e.log_e;
        out->deg_log_e = boa->deg_e.log_e;
        out->n_updates = boa->n_updates;
    }

    return EPROC_OK;
}

int eproc_boa_snapshot_weights(const eproc_boa_t *boa,
                               double *weights_out,
                               uint32_t max_experts,
                               int32_t *fallback_out) {
    if (boa == NULL || (max_experts > 0U && weights_out == NULL)) {
        return EPROC_ERR_INVALID;
    }

    uint64_t snap[EPROC_BOA_MAX_EXPERTS];
    uint64_t meta = 0ULL;
    const uint32_t lim = (max_experts < EPROC_BOA_MAX_EXPERTS) ? max_experts : EPROC_BOA_MAX_EXPERTS;
    eproc_seqlock_read_meta_words_u64(&boa->seqlock_seq,
                                      boa->seqlock_words,
                                      EPROC_BOA_MAX_EXPERTS,
                                      &meta,
                                      snap,
                                      lim);

    uint32_t n_exp = (uint32_t)(meta >> 32);
    if (n_exp > EPROC_BOA_MAX_EXPERTS) {
        n_exp = EPROC_BOA_MAX_EXPERTS;
    }
    const int32_t fb = (int32_t)(uint32_t)(meta & 0xFFFFFFFFULL);

    const uint32_t copy_n = (n_exp < lim) ? n_exp : lim;
    for (uint32_t i = 0; i < copy_n; ++i) {
        weights_out[i] = eproc_u64_to_f64(snap[i]);
    }
    if (fallback_out != NULL) {
        *fallback_out = fb;
    }
    return EPROC_OK;
}

/* ============================================================================
 * 3B. PROVENANCE-DEDUPLICATED MULTI-AGENT EVIDENCE POOL (ECHOPOOL)
 * ============================================================================ */

static inline uint64_t eproc_echo_hash(uint32_t claim_id, uint64_t evidence_id) {
    uint64_t h = evidence_id ^ ((uint64_t)claim_id * 0x9e3779b97f4a7c15ULL);
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
    h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
    h ^= (h >> 31);
    return (h == 0ULL) ? 1ULL : h;
}

static inline double eproc_echo_calc_rel(double alpha, double beta) {
    const double tot = alpha + beta;
    if (!(tot > 0.0)) {
        return 0.0;
    }
    const double acc = alpha / tot;
    return eproc_clamp(2.0 * acc - 1.0, 0.0, 1.0);
}

int eproc_echopool_init(eproc_echopool_t *pool,
                        uint32_t n_sources,
                        uint32_t n_claims,
                        double prior_prob) {
    if (pool == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (n_sources == 0 || n_sources > EPROC_ECHO_MAX_SOURCES) {
        return EPROC_ERR_INVALID;
    }
    if (n_claims == 0 || n_claims > EPROC_ECHO_MAX_CLAIMS) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(prior_prob) || !(prior_prob > 0.0 && prior_prob < 1.0)) {
        return EPROC_ERR_INVALID;
    }

    memset(pool, 0, sizeof(*pool));
    pool->n_sources = n_sources;
    pool->n_claims = n_claims;

    /* Optimistic Beta(2, 0) prior: fresh sources start at reliability = 1.0
     * and shrink smoothly toward 0 upon inaccurate claim resolutions. */
    for (uint32_t s = 0; s < n_sources; ++s) {
        pool->src_alpha[s] = 2.0;
        pool->src_beta[s] = 0.0;
        pool->src_reliability[s] = 1.0;
    }

    const double prior_lo = log(prior_prob / (1.0 - prior_prob));
    for (uint32_t c = 0; c < n_claims; ++c) {
        pool->claim_prior_lo[c] = prior_lo;
        pool->claim_lo[c] = prior_lo;
        pool->claim_lo_comp[c] = 0.0;
        pool->claim_unique[c] = 0;
    }

    return EPROC_OK;
}

int eproc_echopool_submit(eproc_echopool_t *pool,
                          uint32_t source_id,
                          uint32_t claim_id,
                          uint64_t evidence_id,
                          double llr,
                          eproc_echo_out_t *out) {
    if (pool == NULL || source_id >= pool->n_sources || claim_id >= pool->n_claims) {
        return EPROC_ERR_INVALID;
    }

    if (!isfinite(llr)) {
        if (out != NULL) {
            const double lo = eproc_kahan_total(pool->claim_lo[claim_id],
                                                pool->claim_lo_comp[claim_id]);
            out->accepted = 0;
            out->_pad0 = 0;
            out->log_odds = lo;
            out->posterior_prob = eproc_sigmoid(lo);
            out->source_reliability = pool->src_reliability[source_id];
            out->unique_count = pool->claim_unique[claim_id];
        }
        return EPROC_OK;
    }

    const double clamped_llr = eproc_clamp(llr, -50.0, 50.0);
    const uint64_t h = eproc_echo_hash(claim_id, evidence_id);
    const uint32_t mask = EPROC_ECHO_TABLE_CAP - 1U;
    uint32_t idx = (uint32_t)(h & mask);
    uint32_t free_idx = EPROC_ECHO_TABLE_CAP;

    for (uint32_t probe = 0; probe < EPROC_ECHO_TABLE_CAP; ++probe) {
        const eproc_echo_slot_t *slot = &pool->table[idx];
        if (slot->used == 0U) {
            free_idx = idx;
            break;
        }
        if (slot->key_hash == h &&
            slot->claim_id == claim_id &&
            slot->evidence_id == evidence_id) {
            /* Duplicate echo dropped! */
            pool->n_echo_dropped++;
            if (out != NULL) {
                const double lo = eproc_kahan_total(pool->claim_lo[claim_id],
                                                    pool->claim_lo_comp[claim_id]);
                out->accepted = 0;
                out->_pad0 = 0;
                out->log_odds = lo;
                out->posterior_prob = eproc_sigmoid(lo);
                out->source_reliability = pool->src_reliability[source_id];
                out->unique_count = pool->claim_unique[claim_id];
            }
            return EPROC_OK;
        }
        idx = (idx + 1U) & mask;
    }

    if (free_idx == EPROC_ECHO_TABLE_CAP) {
        pool->n_table_full++;
        if (out != NULL) {
            const double lo = eproc_kahan_total(pool->claim_lo[claim_id],
                                                pool->claim_lo_comp[claim_id]);
            out->accepted = 0;
            out->_pad0 = 0;
            out->log_odds = lo;
            out->posterior_prob = eproc_sigmoid(lo);
            out->source_reliability = pool->src_reliability[source_id];
            out->unique_count = pool->claim_unique[claim_id];
        }
        return EPROC_ERR_CAPACITY;
    }

    const double rel = pool->src_reliability[source_id];
    const double added = rel * clamped_llr;

    /* Preserve directional sign in slot even when rel == 0 so resolve can
     * still score a silenced peer for potential recovery. */
    eproc_echo_slot_t *dst = &pool->table[free_idx];
    dst->key_hash = h;
    dst->evidence_id = evidence_id;
    dst->claim_id = claim_id;
    dst->source_id = source_id;
    if (rel > 0.0 || clamped_llr == 0.0) {
        dst->weighted_llr = added;
    } else {
        dst->weighted_llr = copysign(1e-300, clamped_llr);
    }
    dst->used = 1U;

    eproc_kahan_add(&pool->claim_lo[claim_id], &pool->claim_lo_comp[claim_id], added);
    pool->claim_unique[claim_id]++;
    pool->n_accepted++;

    if (out != NULL) {
        const double lo = eproc_kahan_total(pool->claim_lo[claim_id],
                                            pool->claim_lo_comp[claim_id]);
        out->accepted = 1;
        out->_pad0 = 0;
        out->log_odds = lo;
        out->posterior_prob = eproc_sigmoid(lo);
        out->source_reliability = rel;
        out->unique_count = pool->claim_unique[claim_id];
    }
    return EPROC_OK;
}

int eproc_echopool_resolve(eproc_echopool_t *pool,
                           uint32_t claim_id,
                           int32_t outcome_positive) {
    if (pool == NULL || claim_id >= pool->n_claims) {
        return EPROC_ERR_INVALID;
    }

    const int truth_pos = (outcome_positive != 0) ? 1 : 0;
    for (uint32_t i = 0; i < EPROC_ECHO_TABLE_CAP; ++i) {
        eproc_echo_slot_t *slot = &pool->table[i];
        if (slot->used != 1U || slot->claim_id != claim_id) {
            continue;
        }
        slot->used = 2U; /* mark resolved while keeping echo deduplication active */
        if (slot->weighted_llr == 0.0) {
            continue;
        }
        const int pred_pos = (slot->weighted_llr > 0.0) ? 1 : 0;
        const uint32_t s = slot->source_id;
        if (s < pool->n_sources) {
            if (pred_pos == truth_pos) {
                pool->src_alpha[s] += 1.0;
            } else {
                pool->src_beta[s] += 1.0;
            }
            pool->src_reliability[s] = eproc_echo_calc_rel(pool->src_alpha[s],
                                                           pool->src_beta[s]);
        }
    }

    return EPROC_OK;
}

double eproc_echopool_posterior(const eproc_echopool_t *pool, uint32_t claim_id) {
    if (pool == NULL || claim_id >= pool->n_claims) {
        return 0.5;
    }
    return eproc_sigmoid(eproc_kahan_total(pool->claim_lo[claim_id],
                                           pool->claim_lo_comp[claim_id]));
}

double eproc_echopool_source_reliability(const eproc_echopool_t *pool, uint32_t source_id) {
    if (pool == NULL || source_id >= pool->n_sources) {
        return 0.0;
    }
    return pool->src_reliability[source_id];
}
