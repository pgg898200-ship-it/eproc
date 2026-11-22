#ifndef EPROC_H
#define EPROC_H

/*
 * eproc — Sub-microsecond anytime-valid online statistical inference,
 * multi-rate conformal calibration, Byzantine-safe multi-agent evidence fusion,
 * and causal Horvitz-Thompson subsampling.
 *
 * Portable C11 ABI (zero runtime dependencies, zero hot-path malloc,
 * IEEE-754 binary64 deterministic with -ffp-contract=off).
 *
 * License: Apache-2.0
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EPROC_ABI_VERSION 1U

/* Status codes */
#define EPROC_OK            0
#define EPROC_ERR_INVALID  -1
#define EPROC_ERR_CAPACITY -2
#define EPROC_ERR_EMPTY    -3
#define EPROC_ERR_MISMATCH -4

/* Verdict codes */
#define EPROC_VERDICT_CONTINUE 0
#define EPROC_VERDICT_REJECT   1
#define EPROC_VERDICT_PROMOTE  1
#define EPROC_VERDICT_ROLLBACK -1

/* Test sides */
#define EPROC_SIDE_UPPER     0  /* H0: E[X] <= mu0, H1: E[X] > mu0 */
#define EPROC_SIDE_LOWER     1  /* H0: E[X] >= mu0, H1: E[X] < mu0 */
#define EPROC_SIDE_TWO_SIDED 2  /* H0: E[X] == mu0, H1: E[X] != mu0 */

/* Hard capacity bounds for zero-allocation hot paths */
#define EPROC_MAX_ARMS            64
#define EPROC_MAX_RATES           5
#define EPROC_AGACI_EXPERTS       4
#define EPROC_CONFORMAL_MAX_WIN   256
#define EPROC_CMONOID_BINS        64
#define EPROC_DELAY_MAX_PENDING   4096
#define EPROC_BOA_MAX_EXPERTS     64
#define EPROC_ECHO_MAX_SOURCES    256
#define EPROC_ECHO_MAX_CLAIMS     256
#define EPROC_ECHO_TABLE_CAP      4096   /* must be power of 2 */
#define EPROC_VAROPT_MAX_K        4096
#define EPROC_RLS_MAX_D           32
#define EPROC_SIFT_MAX_SITES      1024
#define EPROC_SIFT_MAX_PER_SITE   16

/* ============================================================================
 * 1. ANYTIME-VALID BETTING E-PROCESSES, CONFIDENCE SEQUENCES & AGENT RACING
 * ============================================================================ */

/*
 * ONS Betting Supermartingale E-Process for bounded streams X_t in [lower, upper].
 * Exact Ville inequality: P_{H0}(exists t >= 1: e_value_t >= 1/alpha) <= alpha.
 */
typedef struct {
    double lower;
    double upper;
    double inv_span;
    double mu0_norm;      /* normalized null mean in (0, 1) */
    double alpha;
    double log_thresh;    /* log(1 / alpha) */
    double c_max;         /* bet cap fraction in (0, 0.95], default 0.75 */
    int32_t side;         /* EPROC_SIDE_* */
    uint32_t _pad0;

    /* Upper-tail betting state (lambda_pos in [0, c_max / mu0_norm]) */
    double lam_pos;
    double a_pos;
    double log_e_pos;

    /* Lower-tail betting state (lambda_neg in [-c_max / (1 - mu0_norm), 0]) */
    double lam_neg;
    double a_neg;
    double log_e_neg;

    /* Running summaries (Kahan compensated) */
    double log_e;         /* current log E_t */
    double max_log_e;     /* running max log E_s for anytime-valid p-value */
    double sum_x;
    double sum_x_comp;
    double sum_sq;
    uint64_t n;
    uint64_t rejected_at; /* 0 if not yet rejected, else first tick t */
    uint64_t n_invalid;   /* count of sanitized NaN/Inf inputs */
} eproc_bet_t;

typedef struct {
    uint64_t n;
    double e_value;       /* exp(min(log_e, 700)) */
    double log_e;
    double max_log_e;
    double p_value;       /* min(1.0, exp(-max_log_e)) — anytime-valid */
    double mean;
    int32_t rejected;     /* 1 if max_log_e >= log(1/alpha) */
    uint32_t _pad0;
} eproc_bet_result_t;

int eproc_bet_init(eproc_bet_t *ep,
                   double mu0,
                   double alpha,
                   double lower,
                   double upper,
                   int32_t side);

int eproc_bet_update(eproc_bet_t *ep, double x, eproc_bet_result_t *out);

int eproc_bet_update_batch(eproc_bet_t *ep,
                           const double *xs,
                           size_t n,
                           eproc_bet_result_t *out);

void eproc_bet_query(const eproc_bet_t *ep, eproc_bet_result_t *out);

/* Combine two independent e-processes: log E_prod = log E_a + log E_b */
double eproc_combine_independent_log(double log_e_a, double log_e_b);

/* Combine K arbitrarily dependent e-processes: E_avg = (1/K) sum_k E_k (Vovk & Wang 2021) */
double eproc_combine_dependent_log(const double *log_es, size_t k);

/*
 * Exact Closed-Form Empirical-Bernstein Confidence Sequence (Waudby-Smith & Ramdas 2024)
 * for bounded streams X_t in [lower, upper], plus sub-Gaussian Normal-Mixture CS
 * (Howard et al. 2021) when mode == 1.
 *
 * Runs in O(1) time and O(1) memory (~25 ns/update) with running-intersection tightening.
 */
#define EPROC_CS_EMPIRICAL_BERNSTEIN 0
#define EPROC_CS_NORMAL_MIXTURE      1

typedef struct {
    double lower;
    double upper;
    double span;
    double inv_span;
    double alpha;
    double log_2_over_alpha;
    double sigma;         /* sub-Gaussian proxy for NORMAL_MIXTURE */
    double rho;           /* mixture prior variance for NORMAL_MIXTURE */
    int32_t mode;
    uint32_t _pad0;

    /* Empirical-Bernstein predictable accumulators (normalized to [0, 1]) */
    double sum_y;
    double sum_y_comp;
    double sum_sq_dev;    /* sum (y_s - mu_hat_{s-1})^2 */
    double sum_lam;       /* sum lambda_s */
    double sum_lam_y;     /* sum lambda_s * y_s */
    double sum_psi_v;     /* sum psi_E(lambda_s) * (y_s - mu_hat_{s-1})^2 */

    /* Running intersection bounds in original [lower, upper] scale */
    double cs_low;
    double cs_high;
    double sum_x;
    double sum_x_comp;
    uint64_t n;
    uint64_t n_invalid;
} eproc_cs_t;

typedef struct {
    uint64_t n;
    double mean;
    double low;
    double high;
    double width;
} eproc_cs_result_t;

int eproc_cs_init_eb(eproc_cs_t *cs, double alpha, double lower, double upper);
int eproc_cs_init_normal(eproc_cs_t *cs, double alpha, double sigma, double v_opt);
int eproc_cs_update(eproc_cs_t *cs, double x, eproc_cs_result_t *out);
int eproc_cs_update_batch(eproc_cs_t *cs,
                          const double *xs,
                          size_t n,
                          eproc_cs_result_t *out);
void eproc_cs_query(const eproc_cs_t *cs, eproc_cs_result_t *out);

/*
 * Anytime-Valid E-Detector (Page / Shiryaev-Roberts resettable e-process)
 * for streaming drift / silent regression detection.
 * Alarms when R_t = max(1, R_{t-1}) * e_t >= 1/alpha (ARL >= 1/alpha under H0).
 */
typedef struct {
    eproc_bet_t inner;
    double log_cusum;     /* Page-CUSUM resettable e-detector log R_t */
    double max_log_cusum;
    double log_thresh;
    uint64_t alarm_at;
    uint64_t n_alarms;
    uint64_t n;
} eproc_edetect_t;

typedef struct {
    uint64_t n;
    double log_stat;
    double stat;
    int32_t alarm;
    uint32_t _pad0;
} eproc_edetect_result_t;

int eproc_edetect_init(eproc_edetect_t *ed,
                       double mu0,
                       double alpha,
                       double lower,
                       double upper,
                       int32_t side);
int eproc_edetect_update(eproc_edetect_t *ed, double x, eproc_edetect_result_t *out);
void eproc_edetect_reset_alarm(eproc_edetect_t *ed);

/*
 * Anytime-Valid Multi-Arm Agent/Model Racer (Best-Arm Identification with
 * Automatic Early Elimination & Exact FWER Control).
 *
 * Each of K arms maintains an Empirical-Bernstein Confidence Sequence at
 * Bonferroni-adjusted alpha / K. Arm j is eliminated the instant
 *   cs_high[j] < max_i cs_low[i] - margin.
 * Winner is certified when only 1 active arm remains.
 */
typedef struct {
    uint32_t n_arms;
    uint32_t n_active;
    double alpha;
    double lower;
    double upper;
    double margin;        /* indifference / practical significance margin >= 0 */
    int32_t winner;       /* -1 until certified, else winning arm index */
    uint32_t _pad0;
    uint64_t total_obs;
    uint8_t active[EPROC_MAX_ARMS];
    uint64_t eliminated_at[EPROC_MAX_ARMS];
    eproc_cs_t arms[EPROC_MAX_ARMS];
} eproc_race_t;

typedef struct {
    uint32_t n_arms;
    uint32_t n_active;
    int32_t winner;
    int32_t leader;
    uint64_t total_obs;
    double leader_low;
    double leader_high;
    double runner_up_high;
} eproc_race_status_t;

int eproc_race_init(eproc_race_t *race,
                    uint32_t n_arms,
                    double alpha,
                    double lower,
                    double upper,
                    double margin);

int eproc_race_update(eproc_race_t *race,
                      uint32_t arm,
                      double reward,
                      eproc_race_status_t *out);

int eproc_race_update_batch(eproc_race_t *race,
                            const uint32_t *arms,
                            const double *rewards,
                            size_t n,
                            eproc_race_status_t *out);

void eproc_race_status(const eproc_race_t *race, eproc_race_status_t *out);
int eproc_race_arm_state(const eproc_race_t *race,
                         uint32_t arm,
                         eproc_cs_result_t *out,
                         int32_t *is_active);
/* Recommend next arm to sample (LCB/UCB active gap-maximizing scheduler) */
int32_t eproc_race_suggest_arm(const eproc_race_t *race);


/* ============================================================================
 * 2. MULTI-RATE ONLINE CONFORMAL, DELAYED FEEDBACK, BUDGET BOUNDS & MONOID
 * ============================================================================ */

/*
 * Multi-Rate Crossing-Free AgACI + Conformal PID Calibrator.
 * Simultaneously calibrates up to 5 coverage levels (e.g. 50/80/90/95/99%)
 * with zero quantile crossing (PAVA isotonic projection), O(W) L1-resident
 * sorted score ring, and 4-rate online BOA/Hedge expert aggregation per rate.
 */
typedef struct {
    uint32_t n_rates;
    uint32_t win_cap;
    uint32_t win_len;
    uint32_t ring_pos;
    double pid_ki;        /* integral saturation gain (0 = pure AgACI) */
    double pid_csat;      /* saturation scale */
    uint64_t n_updates;

    double target_alphas[EPROC_MAX_RATES]; /* descending alpha = ascending coverage */
    double gammas[EPROC_AGACI_EXPERTS];    /* {0.005, 0.02, 0.08, 0.20} */

    /* Per-rate, per-expert adaptive alpha_t & BOA weights */
    double alpha_t[EPROC_MAX_RATES][EPROC_AGACI_EXPERTS];
    double exp_log_w[EPROC_MAX_RATES][EPROC_AGACI_EXPERTS];
    double exp_var[EPROC_MAX_RATES][EPROC_AGACI_EXPERTS];
    double pid_err_sum[EPROC_MAX_RATES];
    double eff_alpha[EPROC_MAX_RATES];
    double current_q[EPROC_MAX_RATES];     /* guaranteed non-decreasing */
    double emp_cov_ema[EPROC_MAX_RATES];
    uint64_t hit_counts[EPROC_MAX_RATES];

    /* Circular raw buffer + incrementally maintained sorted buffer */
    double ring[EPROC_CONFORMAL_MAX_WIN];
    double sorted[EPROC_CONFORMAL_MAX_WIN];
} eproc_conformal_t;

typedef struct {
    uint32_t n_rates;
    uint32_t _pad0;
    double quantiles[EPROC_MAX_RATES];     /* monotone non-decreasing radii */
    double eff_alphas[EPROC_MAX_RATES];
    double emp_coverage[EPROC_MAX_RATES];
} eproc_conformal_out_t;

int eproc_conformal_init(eproc_conformal_t *cp,
                         const double *alphas,
                         uint32_t n_rates,
                         uint32_t window_size,
                         double pid_ki);

void eproc_conformal_predict(const eproc_conformal_t *cp,
                             double point_pred,
                             double scale,
                             double *lows_out,
                             double *highs_out,
                             eproc_conformal_out_t *meta_out);

int eproc_conformal_update(eproc_conformal_t *cp,
                           double nonconformity_score,
                           eproc_conformal_out_t *meta_out);

int eproc_conformal_update_batch(eproc_conformal_t *cp,
                                 const double *scores,
                                 size_t n,
                                 eproc_conformal_out_t *meta_out);

/*
 * Strictly Causal Delayed-Feedback Conformal Wrapper.
 * Issue predictions at t_issue; reveal ground truth y at arbitrary later t_reveal.
 * Predictions depend ONLY on labels revealed <= t_issue (assert_causal exact).
 */
typedef struct {
    uint64_t ticket_id;
    double pred;
    double scale;
    uint8_t occupied;
    uint8_t _pad[7];
} eproc_delay_slot_t;

typedef struct {
    eproc_conformal_t core;
    uint32_t pending_count;
    uint32_t _pad0;
    uint64_t next_ticket;
    eproc_delay_slot_t slots[EPROC_DELAY_MAX_PENDING];
} eproc_delayed_conformal_t;

int eproc_delayed_conformal_init(eproc_delayed_conformal_t *dc,
                                 const double *alphas,
                                 uint32_t n_rates,
                                 uint32_t window_size,
                                 double pid_ki);

int eproc_delayed_conformal_issue(eproc_delayed_conformal_t *dc,
                                  double pred,
                                  double scale,
                                  uint64_t *ticket_out,
                                  double *lows_out,
                                  double *highs_out);

int eproc_delayed_conformal_reveal(eproc_delayed_conformal_t *dc,
                                   uint64_t ticket_id,
                                   double y_true,
                                   eproc_conformal_out_t *meta_out);

/*
 * Right-Censored Agent Budget Conformal Calibrator (PROBE-style UPB).
 * Calibrates upper predictive bounds on agent latency/tokens/steps when
 * timed-out trajectories are right-censored at budget B_t.
 */
typedef struct {
    eproc_conformal_t core;
    double censor_penalty; /* conservative inflation factor >= 1.0 on censored runs */
    uint64_t n_completed;
    uint64_t n_censored;
} eproc_budget_conformal_t;

int eproc_budget_conformal_init(eproc_budget_conformal_t *bc,
                                double alpha,
                                uint32_t window_size,
                                double censor_penalty);

double eproc_budget_conformal_bound(const eproc_budget_conformal_t *bc,
                                    double base_estimate);

int eproc_budget_conformal_update(eproc_budget_conformal_t *bc,
                                  double base_estimate,
                                  double observed_cost,
                                  int32_t censored);

/*
 * Exact Associative & Commutative Distributed Conformal Monoid (640 bytes fixed!).
 *
 * Uses uint64_t exact bin counts over 64 signed log-spaced buckets spanning
 * [-16, +32] (with 2 tail overflow buckets), making merge(A, B) and
 * leave-one-out subtract(global, local) BIT-EXACT abelian group operations
 * (0.0 associativity drift across arbitrary tree-reduction orders!).
 */
typedef struct {
    uint64_t count;
    uint64_t bin_counts[EPROC_CMONOID_BINS];
    double sum;
    double sum_comp;
    double sum_sq;
    double sum_sq_comp;
    double min_val;
    double max_val;
    double log_wealth;
    double scale_hint;    /* optional reference scale (>0, default 1.0) */
    uint64_t flags;
    uint64_t _reserved[6];
} eproc_cmonoid_t;

void eproc_cmonoid_init(eproc_cmonoid_t *m, double scale_hint);
void eproc_cmonoid_add(eproc_cmonoid_t *m, double score);
void eproc_cmonoid_add_batch(eproc_cmonoid_t *m, const double *scores, size_t n);
void eproc_cmonoid_merge(const eproc_cmonoid_t *a,
                         const eproc_cmonoid_t *b,
                         eproc_cmonoid_t *out);
int  eproc_cmonoid_subtract(const eproc_cmonoid_t *global_m,
                            const eproc_cmonoid_t *local_m,
                            eproc_cmonoid_t *loo_out);
double eproc_cmonoid_quantile(const eproc_cmonoid_t *m, double q);
size_t eproc_cmonoid_wire_size(void);
int  eproc_cmonoid_serialize(const eproc_cmonoid_t *m, uint8_t *buf, size_t cap);
int  eproc_cmonoid_deserialize(eproc_cmonoid_t *m, const uint8_t *buf, size_t len);


/* ============================================================================
 * 3. BYZANTINE-SAFE MULTI-AGENT EVIDENCE & PREDICTION FUSION
 * ============================================================================ */

/*
 * Scale-Free Bernstein Online Aggregation (BOA, Wintenberger 2017) with
 * 3-Gate Ville E-Process Non-Inferiority Fallback & Poisoning Isolation.
 *
 * Also exposes a Boehm-2012 lock-free seqlocked snapshot of active weights
 * so concurrent inference threads read weights with 0 locks and 0 torn reads.
 */
typedef struct {
    uint32_t n_experts;
    uint32_t anchor_idx;      /* fallback baseline expert index */
    int32_t fallback_active;  /* 1 if deg_e tripped and BOA is quarantined */
    uint32_t _pad0;
    double scale_ema;         /* F_{t-1}-predictable loss scale */
    double clip_mult;         /* poisoning cap multiplier (default 25.0) */
    uint64_t n_updates;
    uint64_t n_poison_clipped;
    uint64_t n_fallback_trips;

    double weights[EPROC_BOA_MAX_EXPERTS];
    double log_w[EPROC_BOA_MAX_EXPERTS];
    double grad_sq_sum[EPROC_BOA_MAX_EXPERTS];
    double cum_loss[EPROC_BOA_MAX_EXPERTS];
    uint8_t quarantined[EPROC_BOA_MAX_EXPERTS];
    uint32_t poison_strikes[EPROC_BOA_MAX_EXPERTS];

    /* 3 Ville E-processes gating non-inferiority vs anchor */
    eproc_bet_t sup_e;        /* superiority vs anchor */
    eproc_bet_t deg_e;        /* degradation vs anchor -> trips fallback */
    eproc_bet_t rec_e;        /* recovery after fallback -> re-enables BOA */

    /* Lock-free Boehm-2012 seqlocked snapshot of published weights */
    uint64_t seqlock_seq;     /* accessed atomically in C implementation */
    uint64_t seqlock_words[EPROC_BOA_MAX_EXPERTS + 2]; /* weights + meta */
} eproc_boa_t;

typedef struct {
    double fused_pred;
    double anchor_pred;
    int32_t fallback_active;
    uint32_t n_quarantined;
    double sup_log_e;
    double deg_log_e;
    uint64_t n_updates;
} eproc_boa_step_t;

int eproc_boa_init(eproc_boa_t *boa,
                   uint32_t n_experts,
                   uint32_t anchor_idx,
                   double alpha_deg);

double eproc_boa_predict(const eproc_boa_t *boa, const double *expert_preds);

int eproc_boa_update(eproc_boa_t *boa,
                     const double *expert_preds,
                     double y_true,
                     eproc_boa_step_t *out);

int eproc_boa_snapshot_weights(const eproc_boa_t *boa,
                               double *weights_out,
                               uint32_t max_experts,
                               int32_t *fallback_out);

/*
 * Provenance-Deduplicated Multi-Agent Evidence Pool (EchoPool) + Online
 * Beta-Bernoulli Peer Reliability Tracker.
 *
 * Solves:
 *  - Echo double-counting: each unique (claim_id, evidence_id) counted ONCE.
 *  - Overwrite churn (84% at W=8 -> 0%): commutative log-odds accumulation.
 *  - Uncalibrated / sycophantic / Byzantine peers: Beta-Bernoulli reliability
 *    weighting shrinks noisy/adversarial agents toward 0 weight.
 */
typedef struct {
    uint64_t key_hash;
    uint64_t evidence_id;
    uint32_t claim_id;
    uint32_t source_id;
    double weighted_llr;
    uint8_t used;
    uint8_t _pad[7];
} eproc_echo_slot_t;

typedef struct {
    uint32_t n_sources;
    uint32_t n_claims;
    uint64_t n_accepted;
    uint64_t n_echo_dropped;
    uint64_t n_table_full;

    /* Per-source Beta-Bernoulli accuracy / reliability state */
    double src_alpha[EPROC_ECHO_MAX_SOURCES];
    double src_beta[EPROC_ECHO_MAX_SOURCES];
    double src_reliability[EPROC_ECHO_MAX_SOURCES]; /* in [0, 1] */

    /* Per-claim accumulated log-odds & unique evidence count */
    double claim_prior_lo[EPROC_ECHO_MAX_CLAIMS];
    double claim_lo[EPROC_ECHO_MAX_CLAIMS];
    double claim_lo_comp[EPROC_ECHO_MAX_CLAIMS];
    uint64_t claim_unique[EPROC_ECHO_MAX_CLAIMS];

    /* Open-addressing deduplication table */
    eproc_echo_slot_t table[EPROC_ECHO_TABLE_CAP];
} eproc_echopool_t;

typedef struct {
    int32_t accepted;       /* 1 if novel evidence, 0 if duplicate echo dropped */
    uint32_t _pad0;
    double posterior_prob;
    double log_odds;
    double source_reliability;
    uint64_t unique_count;
} eproc_echo_out_t;

int eproc_echopool_init(eproc_echopool_t *pool,
                        uint32_t n_sources,
                        uint32_t n_claims,
                        double prior_prob);

int eproc_echopool_submit(eproc_echopool_t *pool,
                          uint32_t source_id,
                          uint32_t claim_id,
                          uint64_t evidence_id,
                          double llr,
                          eproc_echo_out_t *out);

int eproc_echopool_resolve(eproc_echopool_t *pool,
                           uint32_t claim_id,
                           int32_t outcome_positive);

double eproc_echopool_posterior(const eproc_echopool_t *pool, uint32_t claim_id);
double eproc_echopool_source_reliability(const eproc_echopool_t *pool, uint32_t source_id);


/* ============================================================================
 * 4. CAUSAL ONLINE SUBSAMPLING & MULTI-RESOLUTION VALUE-OF-INFORMATION (VOI)
 * ============================================================================ */

/*
 * Variance-Optimal Weighted Reservoir Sampler (Cohen et al. 2011)
 * with exact unbiased Horvitz-Thompson adjusted weights.
 */
typedef struct {
    uint64_t item_id;
    double orig_weight;
    double adj_weight;    /* Horvitz-Thompson unbiased weight */
    double value;
    double rand_u;
} eproc_varopt_item_t;

typedef struct {
    uint32_t k;           /* reservoir capacity <= EPROC_VAROPT_MAX_K */
    uint32_t count;       /* current items in reservoir */
    uint32_t n_heavy;     /* count of heavy items (w >= tau) */
    uint32_t _pad0;
    double tau;           /* inclusion threshold */
    double total_weight;
    double total_weight_comp;
    uint64_t n_seen;
    uint64_t rng_state[2]; /* xoshiro128++ deterministic PRNG */
    eproc_varopt_item_t items[EPROC_VAROPT_MAX_K];
} eproc_varopt_t;

int eproc_varopt_init(eproc_varopt_t *vo, uint32_t k, uint64_t seed);
int eproc_varopt_push(eproc_varopt_t *vo, uint64_t item_id, double weight, double value);
double eproc_varopt_estimate_sum(const eproc_varopt_t *vo);
double eproc_varopt_estimate_mean(const eproc_varopt_t *vo);

/*
 * Online Ridge Leverage Score (RLS) Causal Subsampler
 * (Cohen-Musco-Pachocki, Sherman-Morrison rank-1 inverse updates).
 */
typedef struct {
    uint32_t d;           /* feature dimension <= EPROC_RLS_MAX_D */
    uint32_t _pad0;
    double lambda;        /* ridge regularization > 0 */
    double oversample;    /* multiplier c >= 1.0 */
    double min_prob;      /* heartbeat floor p_min in (0, 1] */
    uint64_t n_seen;
    uint64_t n_kept;
    uint64_t rng_state[2];
    double inv_a[EPROC_RLS_MAX_D * EPROC_RLS_MAX_D];
} eproc_rls_t;

typedef struct {
    int32_t keep;
    uint32_t _pad0;
    double leverage;
    double prob;
    double ht_weight;     /* 1.0 / prob if kept, else 0.0 */
} eproc_rls_out_t;

int eproc_rls_init(eproc_rls_t *rls,
                   uint32_t d,
                   double lambda,
                   double oversample,
                   double min_prob,
                   uint64_t seed);

int eproc_rls_step(eproc_rls_t *rls, const double *x, eproc_rls_out_t *out);

/*
 * Hierarchical Multi-Resolution Bayes Value-of-Information (VOI) Selector
 * (for massive sparse anomaly detection across S sites x M channels under
 * tight attention/tool-call budgets).
 */
typedef struct {
    uint32_t n_sites;
    uint32_t per_site;
    double prior_hot_site;
    double prior_active_cold;
    double prior_active_hot;
    double signal_mean;
    double noise_sd;
    /* Per-site log posterior of hot site, per-channel log odds of active */
    double site_log_prob[EPROC_SIFT_MAX_SITES];
    double chan_prob[EPROC_SIFT_MAX_SITES * EPROC_SIFT_MAX_PER_SITE];
    double chan_noise_scale[EPROC_SIFT_MAX_SITES * EPROC_SIFT_MAX_PER_SITE];
} eproc_sift_t;

typedef struct {
    int32_t action_type;  /* 0 = read site mean, 1 = read channel mean, 2 = read raw ticks */
    uint32_t target_idx;  /* site_id if type==0, else flat channel index */
    double evoi_per_cost;
} eproc_sift_action_t;

int eproc_sift_init(eproc_sift_t *sf,
                    uint32_t n_sites,
                    uint32_t per_site,
                    double signal_mean,
                    double noise_sd);

int eproc_sift_set_noise_scale(eproc_sift_t *sf,
                               uint32_t channel_idx,
                               double noise_scale);

int eproc_sift_observe_channel(eproc_sift_t *sf,
                               uint32_t channel_idx,
                               double obs_mean,
                               uint32_t n_ticks);

int eproc_sift_observe_site(eproc_sift_t *sf,
                            uint32_t site_idx,
                            double site_mean);

int eproc_sift_select_next(const eproc_sift_t *sf,
                           double cost_site,
                           double cost_channel,
                           double cost_ticks,
                           uint32_t ticks_per_read,
                           eproc_sift_action_t *out);

/* Runtime capability query (e.g., returns 1 if compiled with AVX2/FMA) */
uint32_t eproc_abi_version(void);
uint32_t eproc_simd_flags(void); /* bit 0: AVX2, bit 1: FMA, bit 2: NEON */

#ifdef __cplusplus
}
#endif

#endif /* EPROC_H */
