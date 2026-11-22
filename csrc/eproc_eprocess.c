#if !defined(__GNUC__) || defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

#include "eproc.h"
#include "eproc_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ============================================================================
 * 1A. ONS BETTING SUPERMARTINGALE E-PROCESS (Waudby-Smith & Ramdas 2024)
 * ============================================================================ */

static inline double eproc_bet_lam_pos_max(const eproc_bet_t *ep) {
    double denom = (ep->mu0_norm > 1e-9) ? ep->mu0_norm : 1e-9;
    return ep->c_max / denom;
}

static inline double eproc_bet_lam_neg_min(const eproc_bet_t *ep) {
    double one_minus = 1.0 - ep->mu0_norm;
    double denom = (one_minus > 1e-9) ? one_minus : 1e-9;
    return -ep->c_max / denom;
}

int eproc_bet_init(eproc_bet_t *ep,
                   double mu0,
                   double alpha,
                   double lower,
                   double upper,
                   int32_t side) {
    if (ep == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(mu0) || !isfinite(alpha) || !isfinite(lower) || !isfinite(upper)) {
        return EPROC_ERR_INVALID;
    }
    double span = upper - lower;
    if (!(span > 0.0) || !isfinite(span)) {
        return EPROC_ERR_INVALID;
    }
    if (!(alpha > 0.0 && alpha < 1.0)) {
        return EPROC_ERR_INVALID;
    }
    if (mu0 < lower || mu0 > upper) {
        return EPROC_ERR_INVALID;
    }
    if (side != EPROC_SIDE_UPPER &&
        side != EPROC_SIDE_LOWER &&
        side != EPROC_SIDE_TWO_SIDED) {
        return EPROC_ERR_INVALID;
    }

    memset(ep, 0, sizeof(*ep));
    ep->lower = lower;
    ep->upper = upper;
    ep->inv_span = 1.0 / span;
    ep->mu0_norm = eproc_clamp((mu0 - lower) * ep->inv_span, 0.0, 1.0);
    ep->alpha = alpha;
    ep->log_thresh = -log(alpha);
    ep->c_max = 0.75;
    ep->side = side;

    ep->lam_pos = 0.0;
    ep->a_pos = 1.0;
    ep->log_e_pos = 0.0;

    ep->lam_neg = 0.0;
    ep->a_neg = 1.0;
    ep->log_e_neg = 0.0;

    ep->log_e = 0.0;
    ep->max_log_e = 0.0;
    ep->sum_x = 0.0;
    ep->sum_x_comp = 0.0;
    ep->sum_sq = 0.0;
    ep->n = 0;
    ep->rejected_at = 0;
    ep->n_invalid = 0;

    return EPROC_OK;
}

void eproc_bet_query(const eproc_bet_t *ep, eproc_bet_result_t *out) {
    if (ep == NULL || out == NULL) {
        return;
    }
    out->n = ep->n;
    out->log_e = ep->log_e;
    out->max_log_e = ep->max_log_e;
    out->e_value = eproc_safe_exp(ep->log_e);
    if (ep->max_log_e <= 0.0) {
        out->p_value = 1.0;
    } else if (ep->max_log_e >= 745.0) {
        out->p_value = 0.0;
    } else {
        out->p_value = exp(-ep->max_log_e);
    }
    if (ep->n > 0) {
        out->mean = eproc_kahan_total(ep->sum_x, ep->sum_x_comp) / (double)ep->n;
    } else {
        out->mean = 0.0;
    }
    out->rejected = (ep->max_log_e >= ep->log_thresh) ? 1 : 0;
    out->_pad0 = 0;
}

int eproc_bet_update(eproc_bet_t *ep, double x, eproc_bet_result_t *out) {
    if (ep == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(x)) {
        ep->n_invalid++;
        if (out != NULL) {
            eproc_bet_query(ep, out);
        }
        return EPROC_OK;
    }

    const double x_eff = eproc_clamp(x, ep->lower, ep->upper);
    const double y = eproc_clamp((x_eff - ep->lower) * ep->inv_span, 0.0, 1.0);
    const double d = y - ep->mu0_norm;
    const int32_t side = ep->side;

    if (side == EPROC_SIDE_UPPER || side == EPROC_SIDE_TWO_SIDED) {
        const double lam_max = eproc_bet_lam_pos_max(ep);
        double step = ep->lam_pos * d;
        if (step < -0.999999999999) {
            step = -0.999999999999;
        }
        ep->log_e_pos += log1p(step);
        const double z = d / (1.0 + step);
        ep->a_pos += z * z;
        const double next_lam = ep->lam_pos + EPROC_ONS_ETA * (z / ep->a_pos);
        ep->lam_pos = eproc_clamp(next_lam, 0.0, lam_max);
    }

    if (side == EPROC_SIDE_LOWER || side == EPROC_SIDE_TWO_SIDED) {
        const double lam_min = eproc_bet_lam_neg_min(ep);
        double step = ep->lam_neg * d;
        if (step < -0.999999999999) {
            step = -0.999999999999;
        }
        ep->log_e_neg += log1p(step);
        const double z = d / (1.0 + step);
        ep->a_neg += z * z;
        const double next_lam = ep->lam_neg + EPROC_ONS_ETA * (z / ep->a_neg);
        ep->lam_neg = eproc_clamp(next_lam, lam_min, 0.0);
    }

    if (side == EPROC_SIDE_UPPER) {
        ep->log_e = ep->log_e_pos;
    } else if (side == EPROC_SIDE_LOWER) {
        ep->log_e = ep->log_e_neg;
    } else {
        ep->log_e = eproc_logaddexp(ep->log_e_pos, ep->log_e_neg) - EPROC_LN2;
    }

    if (ep->log_e > ep->max_log_e) {
        ep->max_log_e = ep->log_e;
    }

    eproc_kahan_add(&ep->sum_x, &ep->sum_x_comp, x_eff);
    ep->sum_sq += x_eff * x_eff;
    ep->n++;

    if (ep->rejected_at == 0 && ep->max_log_e >= ep->log_thresh) {
        ep->rejected_at = ep->n;
    }

    if (out != NULL) {
        eproc_bet_query(ep, out);
    }
    return EPROC_OK;
}

int eproc_bet_update_batch(eproc_bet_t *ep,
                           const double *xs,
                           size_t n,
                           eproc_bet_result_t *out) {
    if (ep == NULL || (n > 0 && xs == NULL)) {
        return EPROC_ERR_INVALID;
    }

    const double lower = ep->lower;
    const double upper = ep->upper;
    const double inv_span = ep->inv_span;
    const double mu0_norm = ep->mu0_norm;
    const double log_thresh = ep->log_thresh;
    const int32_t side = ep->side;
    const double lam_pos_max = eproc_bet_lam_pos_max(ep);
    const double lam_neg_min = eproc_bet_lam_neg_min(ep);

    double lam_pos = ep->lam_pos;
    double a_pos = ep->a_pos;
    double log_e_pos = ep->log_e_pos;
    double lam_neg = ep->lam_neg;
    double a_neg = ep->a_neg;
    double log_e_neg = ep->log_e_neg;
    double log_e = ep->log_e;
    double max_log_e = ep->max_log_e;
    double sum_x = ep->sum_x;
    double sum_x_comp = ep->sum_x_comp;
    double sum_sq = ep->sum_sq;
    uint64_t count = ep->n;
    uint64_t rejected_at = ep->rejected_at;
    uint64_t n_invalid = ep->n_invalid;

    if (side == EPROC_SIDE_UPPER) {
        for (size_t i = 0; i < n; ++i) {
            const double x = xs[i];
            if (!isfinite(x)) {
                n_invalid++;
                continue;
            }
            const double x_eff = eproc_clamp(x, lower, upper);
            const double y = eproc_clamp((x_eff - lower) * inv_span, 0.0, 1.0);
            const double d = y - mu0_norm;
            double step = lam_pos * d;
            if (step < -0.999999999999) {
                step = -0.999999999999;
            }
            log_e_pos += log1p(step);
            const double z = d / (1.0 + step);
            a_pos += z * z;
            lam_pos = eproc_clamp(lam_pos + EPROC_ONS_ETA * (z / a_pos), 0.0, lam_pos_max);

            log_e = log_e_pos;
            if (log_e > max_log_e) {
                max_log_e = log_e;
            }
            eproc_kahan_add(&sum_x, &sum_x_comp, x_eff);
            sum_sq += x_eff * x_eff;
            count++;
            if (rejected_at == 0 && max_log_e >= log_thresh) {
                rejected_at = count;
            }
        }
    } else if (side == EPROC_SIDE_LOWER) {
        for (size_t i = 0; i < n; ++i) {
            const double x = xs[i];
            if (!isfinite(x)) {
                n_invalid++;
                continue;
            }
            const double x_eff = eproc_clamp(x, lower, upper);
            const double y = eproc_clamp((x_eff - lower) * inv_span, 0.0, 1.0);
            const double d = y - mu0_norm;
            double step = lam_neg * d;
            if (step < -0.999999999999) {
                step = -0.999999999999;
            }
            log_e_neg += log1p(step);
            const double z = d / (1.0 + step);
            a_neg += z * z;
            lam_neg = eproc_clamp(lam_neg + EPROC_ONS_ETA * (z / a_neg), lam_neg_min, 0.0);

            log_e = log_e_neg;
            if (log_e > max_log_e) {
                max_log_e = log_e;
            }
            eproc_kahan_add(&sum_x, &sum_x_comp, x_eff);
            sum_sq += x_eff * x_eff;
            count++;
            if (rejected_at == 0 && max_log_e >= log_thresh) {
                rejected_at = count;
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            const double x = xs[i];
            if (!isfinite(x)) {
                n_invalid++;
                continue;
            }
            const double x_eff = eproc_clamp(x, lower, upper);
            const double y = eproc_clamp((x_eff - lower) * inv_span, 0.0, 1.0);
            const double d = y - mu0_norm;

            double step_p = lam_pos * d;
            if (step_p < -0.999999999999) {
                step_p = -0.999999999999;
            }
            log_e_pos += log1p(step_p);
            const double zp = d / (1.0 + step_p);
            a_pos += zp * zp;
            lam_pos = eproc_clamp(lam_pos + EPROC_ONS_ETA * (zp / a_pos), 0.0, lam_pos_max);

            double step_n = lam_neg * d;
            if (step_n < -0.999999999999) {
                step_n = -0.999999999999;
            }
            log_e_neg += log1p(step_n);
            const double zn = d / (1.0 + step_n);
            a_neg += zn * zn;
            lam_neg = eproc_clamp(lam_neg + EPROC_ONS_ETA * (zn / a_neg), lam_neg_min, 0.0);

            log_e = eproc_logaddexp(log_e_pos, log_e_neg) - EPROC_LN2;
            if (log_e > max_log_e) {
                max_log_e = log_e;
            }
            eproc_kahan_add(&sum_x, &sum_x_comp, x_eff);
            sum_sq += x_eff * x_eff;
            count++;
            if (rejected_at == 0 && max_log_e >= log_thresh) {
                rejected_at = count;
            }
        }
    }

    ep->lam_pos = lam_pos;
    ep->a_pos = a_pos;
    ep->log_e_pos = log_e_pos;
    ep->lam_neg = lam_neg;
    ep->a_neg = a_neg;
    ep->log_e_neg = log_e_neg;
    ep->log_e = log_e;
    ep->max_log_e = max_log_e;
    ep->sum_x = sum_x;
    ep->sum_x_comp = sum_x_comp;
    ep->sum_sq = sum_sq;
    ep->n = count;
    ep->rejected_at = rejected_at;
    ep->n_invalid = n_invalid;

    if (out != NULL) {
        eproc_bet_query(ep, out);
    }
    return EPROC_OK;
}

double eproc_combine_independent_log(double log_e_a, double log_e_b) {
    if (isnan(log_e_a)) {
        log_e_a = 0.0;
    }
    if (isnan(log_e_b)) {
        log_e_b = 0.0;
    }
    if ((log_e_a == INFINITY && log_e_b == -INFINITY) ||
        (log_e_a == -INFINITY && log_e_b == INFINITY)) {
        return 0.0;
    }
    return log_e_a + log_e_b;
}

double eproc_combine_dependent_log(const double *log_es, size_t k) {
    if (log_es == NULL || k == 0) {
        return 0.0;
    }

    double max_v = -INFINITY;
    int has_finite = 0;
    int has_neg_inf = 0;

    for (size_t i = 0; i < k; ++i) {
        const double v = log_es[i];
        if (isnan(v)) {
            continue;
        }
        if (v == INFINITY) {
            return INFINITY;
        }
        if (v == -INFINITY) {
            has_neg_inf = 1;
            continue;
        }
        if (!has_finite || v > max_v) {
            max_v = v;
            has_finite = 1;
        }
    }

    if (!has_finite) {
        return has_neg_inf ? -INFINITY : 0.0;
    }

    double sum_exp = 0.0;
    double sum_exp_comp = 0.0;
    for (size_t i = 0; i < k; ++i) {
        const double v = log_es[i];
        if (!isfinite(v)) {
            continue;
        }
        const double diff = v - max_v;
        if (diff >= -745.0) {
            eproc_kahan_add(&sum_exp, &sum_exp_comp, exp(diff));
        }
    }

    const double total = eproc_kahan_total(sum_exp, sum_exp_comp);
    if (!(total > 0.0)) {
        return -INFINITY;
    }
    return max_v + log(total) - log((double)k);
}

/* ============================================================================
 * 1B. CONFIDENCE SEQUENCES: EMPIRICAL-BERNSTEIN & NORMAL-MIXTURE
 * ============================================================================ */

static inline double eproc_psi_e(double lam) {
    if (lam < 1e-4) {
        const double l2 = lam * lam;
        return l2 * (0.5 + lam * ((1.0 / 3.0) + lam * (0.25 + lam * (0.2 + lam * (1.0 / 6.0)))));
    }
    return -log1p(-lam) - lam;
}

int eproc_cs_init_eb(eproc_cs_t *cs, double alpha, double lower, double upper) {
    if (cs == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(alpha) || !(alpha > 0.0 && alpha < 1.0)) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(lower) || !isfinite(upper)) {
        return EPROC_ERR_INVALID;
    }
    double span = upper - lower;
    if (!(span > 0.0) || !isfinite(span)) {
        return EPROC_ERR_INVALID;
    }

    memset(cs, 0, sizeof(*cs));
    cs->lower = lower;
    cs->upper = upper;
    cs->span = span;
    cs->inv_span = 1.0 / span;
    cs->alpha = alpha;
    cs->log_2_over_alpha = log(2.0 / alpha);
    cs->sigma = 0.0;
    cs->rho = 0.0;
    cs->mode = EPROC_CS_EMPIRICAL_BERNSTEIN;
    cs->cs_low = lower;
    cs->cs_high = upper;

    return EPROC_OK;
}

int eproc_cs_init_normal(eproc_cs_t *cs, double alpha, double sigma, double v_opt) {
    if (cs == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(alpha) || !(alpha > 0.0 && alpha < 1.0)) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(sigma) || !(sigma > 0.0)) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(v_opt) || !(v_opt > 0.0)) {
        return EPROC_ERR_INVALID;
    }

    memset(cs, 0, sizeof(*cs));
    cs->lower = -INFINITY;
    cs->upper = INFINITY;
    cs->span = INFINITY;
    cs->inv_span = 0.0;
    cs->alpha = alpha;
    cs->log_2_over_alpha = log(2.0 / alpha);
    cs->sigma = sigma;

    /* Optimal mixture prior parameter rho for target intrinsic time v_opt (Howard et al. 2021) */
    const double two_log_inv_alpha = -2.0 * log(alpha);
    const double opt_denom = two_log_inv_alpha + log(two_log_inv_alpha + 1.0);
    cs->rho = v_opt / ((opt_denom > 1.0) ? opt_denom : 1.0);
    if (!(cs->rho > 0.0) || !isfinite(cs->rho)) {
        cs->rho = 1.0;
    }

    cs->mode = EPROC_CS_NORMAL_MIXTURE;
    cs->cs_low = -INFINITY;
    cs->cs_high = INFINITY;

    return EPROC_OK;
}

void eproc_cs_query(const eproc_cs_t *cs, eproc_cs_result_t *out) {
    if (cs == NULL || out == NULL) {
        return;
    }
    out->n = cs->n;
    if (cs->n > 0) {
        out->mean = eproc_kahan_total(cs->sum_x, cs->sum_x_comp) / (double)cs->n;
    } else {
        out->mean = 0.0;
    }
    out->low = cs->cs_low;
    out->high = cs->cs_high;
    if (isfinite(cs->cs_low) && isfinite(cs->cs_high)) {
        double w = cs->cs_high - cs->cs_low;
        out->width = (w > 0.0) ? w : 0.0;
    } else {
        out->width = INFINITY;
    }
}

static inline void eproc_cs_step_eb(eproc_cs_t *cs, double x) {
    const double x_eff = eproc_clamp(x, cs->lower, cs->upper);
    const double y = eproc_clamp((x_eff - cs->lower) * cs->inv_span, 0.0, 1.0);

    /* Strictly F_{t-1}-predictable mean, variance, and betting parameter lambda_t */
    const double n_prev = (double)cs->n;
    const double sum_y_prev = eproc_kahan_total(cs->sum_y, cs->sum_y_comp);
    const double mu_hat = (0.5 + sum_y_prev) / (n_prev + 1.0);
    const double var_hat = (0.25 + cs->sum_sq_dev) / (n_prev + 1.0);

    const double denom = var_hat * (n_prev + 1.0) * log(2.0 + n_prev);
    double lam = 0.5;
    if (denom > 0.0) {
        const double raw_lam = sqrt((2.0 * cs->log_2_over_alpha) / denom);
        if (isfinite(raw_lam) && raw_lam < lam) {
            lam = raw_lam;
        }
    }
    if (lam < 0.0) {
        lam = 0.0;
    }

    const double psi_e = eproc_psi_e(lam);
    const double dev = y - mu_hat;
    const double dev_sq = dev * dev;

    /* Update accumulators strictly AFTER using old mu_hat */
    cs->sum_sq_dev += dev_sq;
    cs->sum_lam += lam;
    cs->sum_lam_y += lam * y;
    cs->sum_psi_v += psi_e * dev_sq;
    eproc_kahan_add(&cs->sum_y, &cs->sum_y_comp, y);
    eproc_kahan_add(&cs->sum_x, &cs->sum_x_comp, x_eff);
    cs->n++;

    if (cs->sum_lam > 0.0) {
        const double center = cs->sum_lam_y / cs->sum_lam;
        const double radius = (cs->log_2_over_alpha + cs->sum_psi_v) / cs->sum_lam;
        const double lo_y = eproc_clamp(center - radius, 0.0, 1.0);
        const double hi_y = eproc_clamp(center + radius, 0.0, 1.0);
        const double new_low = cs->lower + cs->span * lo_y;
        const double new_high = cs->lower + cs->span * hi_y;

        if (new_low > cs->cs_low) {
            cs->cs_low = new_low;
        }
        if (new_high < cs->cs_high) {
            cs->cs_high = new_high;
        }
        if (cs->cs_low > cs->cs_high) {
            const double mid = 0.5 * cs->cs_low + 0.5 * cs->cs_high;
            cs->cs_low = mid;
            cs->cs_high = mid;
        }
    }
}

static inline void eproc_cs_step_normal(eproc_cs_t *cs, double x) {
    eproc_kahan_add(&cs->sum_x, &cs->sum_x_comp, x);
    cs->n++;

    const double nd = (double)cs->n;
    const double mean = eproc_kahan_total(cs->sum_x, cs->sum_x_comp) / nd;
    if (!isfinite(mean)) {
        return;
    }
    const double log_inv_alpha = cs->log_2_over_alpha - EPROC_LN2;
    const double rho_plus_n = cs->rho + nd;
    const double inner = rho_plus_n * (2.0 * log_inv_alpha + log(rho_plus_n / cs->rho));
    const double radius = (inner > 0.0) ? (cs->sigma * sqrt(inner) / nd) : 0.0;

    const double new_low = mean - radius;
    const double new_high = mean + radius;
    if (isfinite(new_low) && new_low > cs->cs_low) {
        cs->cs_low = new_low;
    }
    if (isfinite(new_high) && new_high < cs->cs_high) {
        cs->cs_high = new_high;
    }
    if (cs->cs_low > cs->cs_high) {
        const double mid = 0.5 * cs->cs_low + 0.5 * cs->cs_high;
        if (isfinite(mid)) {
            cs->cs_low = mid;
            cs->cs_high = mid;
        }
    }
}

int eproc_cs_update(eproc_cs_t *cs, double x, eproc_cs_result_t *out) {
    if (cs == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(x)) {
        cs->n_invalid++;
        if (out != NULL) {
            eproc_cs_query(cs, out);
        }
        return EPROC_OK;
    }

    if (cs->mode == EPROC_CS_EMPIRICAL_BERNSTEIN) {
        eproc_cs_step_eb(cs, x);
    } else if (cs->mode == EPROC_CS_NORMAL_MIXTURE) {
        eproc_cs_step_normal(cs, x);
    } else {
        return EPROC_ERR_INVALID;
    }

    if (out != NULL) {
        eproc_cs_query(cs, out);
    }
    return EPROC_OK;
}

int eproc_cs_update_batch(eproc_cs_t *cs,
                          const double *xs,
                          size_t n,
                          eproc_cs_result_t *out) {
    if (cs == NULL || (n > 0 && xs == NULL)) {
        return EPROC_ERR_INVALID;
    }

    if (cs->mode == EPROC_CS_EMPIRICAL_BERNSTEIN) {
        for (size_t i = 0; i < n; ++i) {
            const double x = xs[i];
            if (!isfinite(x)) {
                cs->n_invalid++;
                continue;
            }
            eproc_cs_step_eb(cs, x);
        }
    } else if (cs->mode == EPROC_CS_NORMAL_MIXTURE) {
        for (size_t i = 0; i < n; ++i) {
            const double x = xs[i];
            if (!isfinite(x)) {
                cs->n_invalid++;
                continue;
            }
            eproc_cs_step_normal(cs, x);
        }
    } else {
        return EPROC_ERR_INVALID;
    }

    if (out != NULL) {
        eproc_cs_query(cs, out);
    }
    return EPROC_OK;
}

/* ============================================================================
 * 1C. PAGE-CUSUM RESETTABLE E-DETECTOR
 * ============================================================================ */

static inline void eproc_edetect_reset_inner(eproc_bet_t *inner) {
    inner->lam_pos = 0.0;
    inner->a_pos = 1.0;
    inner->log_e_pos = 0.0;
    inner->lam_neg = 0.0;
    inner->a_neg = 1.0;
    inner->log_e_neg = 0.0;
    inner->log_e = 0.0;
}

int eproc_edetect_init(eproc_edetect_t *ed,
                       double mu0,
                       double alpha,
                       double lower,
                       double upper,
                       int32_t side) {
    if (ed == NULL) {
        return EPROC_ERR_INVALID;
    }
    memset(ed, 0, sizeof(*ed));
    int rc = eproc_bet_init(&ed->inner, mu0, alpha, lower, upper, side);
    if (rc != EPROC_OK) {
        return rc;
    }
    ed->log_cusum = 0.0;
    ed->max_log_cusum = 0.0;
    ed->log_thresh = -log(alpha);
    ed->alarm_at = 0;
    ed->n_alarms = 0;
    ed->n = 0;
    return EPROC_OK;
}

int eproc_edetect_update(eproc_edetect_t *ed, double x, eproc_edetect_result_t *out) {
    if (ed == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(x)) {
        ed->inner.n_invalid++;
        if (out != NULL) {
            out->n = ed->n;
            out->log_stat = ed->log_cusum;
            out->stat = eproc_safe_exp(ed->log_cusum);
            out->alarm = (ed->log_cusum >= ed->log_thresh) ? 1 : 0;
            out->_pad0 = 0;
        }
        return EPROC_OK;
    }

    const double log_e_before = ed->inner.log_e;
    eproc_bet_update(&ed->inner, x, NULL);
    const double dl = ed->inner.log_e - log_e_before;

    const double base = (ed->log_cusum > 0.0) ? ed->log_cusum : 0.0;
    ed->log_cusum = base + dl;

    /* Page-CUSUM reset: when cumulative log e-statistic drops below 0,
     * restart at 0 and reset inner ONS bettor so post-reset steps form a
     * fresh valid supermartingale increment. */
    if (ed->log_cusum < 0.0) {
        ed->log_cusum = 0.0;
        eproc_edetect_reset_inner(&ed->inner);
    } else if (ed->log_cusum == 0.0 &&
               ed->inner.lam_pos == 0.0 &&
               ed->inner.lam_neg == 0.0) {
        eproc_edetect_reset_inner(&ed->inner);
    }

    ed->n++;
    if (ed->log_cusum > ed->max_log_cusum) {
        ed->max_log_cusum = ed->log_cusum;
    }

    const int32_t alarm = (ed->log_cusum >= ed->log_thresh) ? 1 : 0;
    if (alarm) {
        if (ed->alarm_at == 0) {
            ed->alarm_at = ed->n;
        }
        ed->n_alarms++;
    }

    if (out != NULL) {
        out->n = ed->n;
        out->log_stat = ed->log_cusum;
        out->stat = eproc_safe_exp(ed->log_cusum);
        out->alarm = alarm;
        out->_pad0 = 0;
    }
    return EPROC_OK;
}

void eproc_edetect_reset_alarm(eproc_edetect_t *ed) {
    if (ed == NULL) {
        return;
    }
    ed->log_cusum = 0.0;
    ed->alarm_at = 0;
    eproc_edetect_reset_inner(&ed->inner);
}

/* ============================================================================
 * 1D. ANYTIME-VALID MULTI-ARM RACER (BONFERRONI EB-CS + LUCB1 SAMPLING)
 * ============================================================================ */

int eproc_race_init(eproc_race_t *race,
                    uint32_t n_arms,
                    double alpha,
                    double lower,
                    double upper,
                    double margin) {
    if (race == NULL) {
        return EPROC_ERR_INVALID;
    }
    if (n_arms == 0 || n_arms > EPROC_MAX_ARMS) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(alpha) || !(alpha > 0.0 && alpha < 1.0)) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(lower) || !isfinite(upper) || !(upper > lower) || !isfinite(upper - lower)) {
        return EPROC_ERR_INVALID;
    }
    if (!isfinite(margin) || margin < 0.0) {
        return EPROC_ERR_INVALID;
    }

    memset(race, 0, sizeof(*race));
    race->n_arms = n_arms;
    race->n_active = n_arms;
    race->alpha = alpha;
    race->lower = lower;
    race->upper = upper;
    race->margin = margin;
    race->winner = (n_arms == 1) ? 0 : -1;
    race->total_obs = 0;

    const double arm_alpha = alpha / (double)n_arms;
    for (uint32_t i = 0; i < n_arms; ++i) {
        race->active[i] = 1;
        race->eliminated_at[i] = 0;
        int rc = eproc_cs_init_eb(&race->arms[i], arm_alpha, lower, upper);
        if (rc != EPROC_OK) {
            return rc;
        }
    }
    return EPROC_OK;
}

static inline void eproc_race_eliminate(eproc_race_t *race) {
    if (race->n_active <= 1) {
        return;
    }
    double max_active_low = -INFINITY;
    for (uint32_t i = 0; i < race->n_arms; ++i) {
        if (race->active[i] && race->arms[i].cs_low > max_active_low) {
            max_active_low = race->arms[i].cs_low;
        }
    }
    const double cutoff = max_active_low - race->margin;
    for (uint32_t j = 0; j < race->n_arms; ++j) {
        if (race->active[j] && race->n_active > 1 && race->arms[j].cs_high < cutoff) {
            race->active[j] = 0;
            race->eliminated_at[j] = race->total_obs;
            race->n_active--;
        }
    }
    if (race->n_active == 1 && race->winner < 0) {
        for (uint32_t k = 0; k < race->n_arms; ++k) {
            if (race->active[k]) {
                race->winner = (int32_t)k;
                break;
            }
        }
    }
}

void eproc_race_status(const eproc_race_t *race, eproc_race_status_t *out) {
    if (race == NULL || out == NULL) {
        return;
    }
    out->n_arms = race->n_arms;
    out->n_active = race->n_active;
    out->winner = race->winner;
    out->total_obs = race->total_obs;

    int32_t leader = 0;
    double best_low = -INFINITY;
    double best_mean = -INFINITY;
    for (uint32_t i = 0; i < race->n_arms; ++i) {
        if (!race->active[i]) {
            continue;
        }
        const double lo = race->arms[i].cs_low;
        const double mean = (race->arms[i].n > 0)
            ? (eproc_kahan_total(race->arms[i].sum_x, race->arms[i].sum_x_comp) / (double)race->arms[i].n)
            : 0.5 * (race->lower + race->upper);
        if (lo > best_low || (lo == best_low && mean > best_mean)) {
            best_low = lo;
            best_mean = mean;
            leader = (int32_t)i;
        }
    }

    double runner_up_high = race->lower;
    int found_other = 0;
    for (uint32_t j = 0; j < race->n_arms; ++j) {
        if ((int32_t)j == leader) {
            continue;
        }
        if (race->n_active > 1 && !race->active[j]) {
            continue;
        }
        if (!found_other || race->arms[j].cs_high > runner_up_high) {
            runner_up_high = race->arms[j].cs_high;
            found_other = 1;
        }
    }

    out->leader = leader;
    out->leader_low = race->arms[leader].cs_low;
    out->leader_high = race->arms[leader].cs_high;
    out->runner_up_high = runner_up_high;
}

int eproc_race_update(eproc_race_t *race,
                      uint32_t arm,
                      double reward,
                      eproc_race_status_t *out) {
    if (race == NULL || arm >= race->n_arms) {
        return EPROC_ERR_INVALID;
    }
    eproc_cs_update(&race->arms[arm], reward, NULL);
    if (isfinite(reward)) {
        race->total_obs++;
        eproc_race_eliminate(race);
    }
    if (out != NULL) {
        eproc_race_status(race, out);
    }
    return EPROC_OK;
}

int eproc_race_update_batch(eproc_race_t *race,
                            const uint32_t *arms,
                            const double *rewards,
                            size_t n,
                            eproc_race_status_t *out) {
    if (race == NULL || (n > 0 && (arms == NULL || rewards == NULL))) {
        return EPROC_ERR_INVALID;
    }
    for (size_t i = 0; i < n; ++i) {
        const uint32_t arm = arms[i];
        if (arm >= race->n_arms) {
            return EPROC_ERR_INVALID;
        }
        const double r = rewards[i];
        eproc_cs_update(&race->arms[arm], r, NULL);
        if (isfinite(r)) {
            race->total_obs++;
            eproc_race_eliminate(race);
        }
    }
    if (out != NULL) {
        eproc_race_status(race, out);
    }
    return EPROC_OK;
}

int eproc_race_arm_state(const eproc_race_t *race,
                         uint32_t arm,
                         eproc_cs_result_t *out,
                         int32_t *is_active) {
    if (race == NULL || arm >= race->n_arms) {
        return EPROC_ERR_INVALID;
    }
    if (out != NULL) {
        eproc_cs_query(&race->arms[arm], out);
    }
    if (is_active != NULL) {
        *is_active = (int32_t)race->active[arm];
    }
    return EPROC_OK;
}

int32_t eproc_race_suggest_arm(const eproc_race_t *race) {
    if (race == NULL || race->n_arms == 0) {
        return -1;
    }
    if (race->winner >= 0) {
        return race->winner;
    }

    /* Warm-start any unsampled active arm first */
    for (uint32_t i = 0; i < race->n_arms; ++i) {
        if (race->active[i] && race->arms[i].n == 0) {
            return (int32_t)i;
        }
    }

    /* LUCB1 active scheduler: identify empirical leader h_t and highest-UCB challenger l_t */
    int32_t leader = -1;
    double best_mean = -INFINITY;
    double best_low = -INFINITY;
    for (uint32_t i = 0; i < race->n_arms; ++i) {
        if (!race->active[i]) {
            continue;
        }
        const double mean = eproc_kahan_total(race->arms[i].sum_x, race->arms[i].sum_x_comp)
                            / (double)race->arms[i].n;
        const double lo = race->arms[i].cs_low;
        if (leader < 0 || mean > best_mean || (mean == best_mean && lo > best_low)) {
            leader = (int32_t)i;
            best_mean = mean;
            best_low = lo;
        }
    }

    if (leader < 0) {
        return 0;
    }

    int32_t challenger = -1;
    double best_challenger_high = -INFINITY;
    uint64_t best_challenger_n = UINT64_MAX;
    for (uint32_t j = 0; j < race->n_arms; ++j) {
        if (!race->active[j] || (int32_t)j == leader) {
            continue;
        }
        const double hi = race->arms[j].cs_high;
        if (challenger < 0 || hi > best_challenger_high ||
            (hi == best_challenger_high && race->arms[j].n < best_challenger_n)) {
            challenger = (int32_t)j;
            best_challenger_high = hi;
            best_challenger_n = race->arms[j].n;
        }
    }

    if (challenger < 0) {
        return leader;
    }

    /* Sample whichever of {leader, challenger} has wider confidence interval
     * (alternating on ties so leader LCB and challenger UCB tighten in lockstep) */
    const double w_leader = race->arms[leader].cs_high - race->arms[leader].cs_low;
    const double w_chall = race->arms[challenger].cs_high - race->arms[challenger].cs_low;
    if (w_leader > w_chall) {
        return leader;
    }
    if (w_chall > w_leader) {
        return challenger;
    }
    return (race->arms[leader].n <= race->arms[challenger].n) ? leader : challenger;
}
