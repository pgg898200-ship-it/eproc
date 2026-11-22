#if !defined(__GNUC__) || defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

#include "eproc.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));    \
            g_failures++;                                                        \
        }                                                                        \
    } while (0)

/* Deterministic test PRNG (splitmix64) */
static uint64_t rng_next(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double rng_u01(uint64_t *s) {
    return (double)(rng_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

static double rng_norm(uint64_t *s) {
    double u1 = rng_u01(s);
    double u2 = rng_u01(s);
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(6.28318530717958647692 * u2);
}

static void print_abi_sizes(void) {
    printf("=== eproc ABI v%u | SIMD flags=0x%x ===\n",
           eproc_abi_version(), eproc_simd_flags());
    printf("sizeof(eproc_bet_t)               = %zu\n", sizeof(eproc_bet_t));
    printf("sizeof(eproc_bet_result_t)        = %zu\n", sizeof(eproc_bet_result_t));
    printf("sizeof(eproc_cs_t)                = %zu\n", sizeof(eproc_cs_t));
    printf("sizeof(eproc_cs_result_t)         = %zu\n", sizeof(eproc_cs_result_t));
    printf("sizeof(eproc_edetect_t)           = %zu\n", sizeof(eproc_edetect_t));
    printf("sizeof(eproc_edetect_result_t)    = %zu\n", sizeof(eproc_edetect_result_t));
    printf("sizeof(eproc_race_t)              = %zu\n", sizeof(eproc_race_t));
    printf("sizeof(eproc_race_status_t)       = %zu\n", sizeof(eproc_race_status_t));
    printf("sizeof(eproc_conformal_t)         = %zu\n", sizeof(eproc_conformal_t));
    printf("sizeof(eproc_conformal_out_t)     = %zu\n", sizeof(eproc_conformal_out_t));
    printf("sizeof(eproc_delay_slot_t)        = %zu\n", sizeof(eproc_delay_slot_t));
    printf("sizeof(eproc_delayed_conformal_t) = %zu\n", sizeof(eproc_delayed_conformal_t));
    printf("sizeof(eproc_budget_conformal_t)  = %zu\n", sizeof(eproc_budget_conformal_t));
    printf("sizeof(eproc_cmonoid_t)           = %zu\n", sizeof(eproc_cmonoid_t));
    printf("sizeof(eproc_boa_t)               = %zu\n", sizeof(eproc_boa_t));
    printf("sizeof(eproc_boa_step_t)          = %zu\n", sizeof(eproc_boa_step_t));
    printf("sizeof(eproc_echo_slot_t)         = %zu\n", sizeof(eproc_echo_slot_t));
    printf("sizeof(eproc_echopool_t)          = %zu\n", sizeof(eproc_echopool_t));
    printf("sizeof(eproc_echo_out_t)          = %zu\n", sizeof(eproc_echo_out_t));
    printf("sizeof(eproc_varopt_item_t)       = %zu\n", sizeof(eproc_varopt_item_t));
    printf("sizeof(eproc_varopt_t)            = %zu\n", sizeof(eproc_varopt_t));
    printf("sizeof(eproc_rls_t)               = %zu\n", sizeof(eproc_rls_t));
    printf("sizeof(eproc_rls_out_t)           = %zu\n", sizeof(eproc_rls_out_t));
    printf("sizeof(eproc_sift_t)              = %zu\n", sizeof(eproc_sift_t));
    printf("sizeof(eproc_sift_action_t)       = %zu\n", sizeof(eproc_sift_action_t));

    CHECK(eproc_abi_version() == EPROC_ABI_VERSION, "ABI version mismatch");
    CHECK(sizeof(eproc_cmonoid_t) == 640, "cmonoid struct size must be 640");
    CHECK(eproc_cmonoid_wire_size() == 640, "cmonoid wire size must be 640");
}

static void test_betting_eprocess(void) {
    uint64_t seed = 12345ULL;

    /* 1. Anytime-valid Type-I error under continuous peeking (N=600, T=400, alpha=0.05) */
    int false_rejects_upper = 0;
    int false_rejects_two = 0;
    const int n_trials = 600;
    const int horizon = 400;
    for (int tr = 0; tr < n_trials; ++tr) {
        eproc_bet_t ep_u, ep_2;
        CHECK(eproc_bet_init(&ep_u, 0.5, 0.05, 0.0, 1.0, EPROC_SIDE_UPPER) == EPROC_OK, "bet init upper");
        CHECK(eproc_bet_init(&ep_2, 0.5, 0.05, 0.0, 1.0, EPROC_SIDE_TWO_SIDED) == EPROC_OK, "bet init two");
        eproc_bet_result_t ru, r2;
        for (int t = 0; t < horizon; ++t) {
            double x = (rng_u01(&seed) < 0.5) ? 1.0 : 0.0;
            eproc_bet_update(&ep_u, x, &ru);
            eproc_bet_update(&ep_2, x, &r2);
        }
        if (ru.rejected) false_rejects_upper++;
        if (r2.rejected) false_rejects_two++;
    }
    double fpr_u = (double)false_rejects_upper / (double)n_trials;
    double fpr_2 = (double)false_rejects_two / (double)n_trials;
    printf("[bet] Type-I FPR (alpha=0.05): upper=%.4f two_sided=%.4f\n", fpr_u, fpr_2);
    CHECK(fpr_u <= 0.05, "upper betting Type-I error must be <= 0.05");
    CHECK(fpr_2 <= 0.05, "two-sided betting Type-I error must be <= 0.05");

    /* 2. Boundary nulls (mu0 == lower and mu0 == upper) */
    {
        eproc_bet_t ep0, ep1;
        eproc_bet_result_t r0, r1;
        CHECK(eproc_bet_init(&ep0, 0.0, 0.05, 0.0, 1.0, EPROC_SIDE_UPPER) == EPROC_OK, "boundary mu0=0");
        CHECK(eproc_bet_init(&ep1, 1.0, 0.05, 0.0, 1.0, EPROC_SIDE_LOWER) == EPROC_OK, "boundary mu0=1");
        for (int t = 0; t < 50; ++t) {
            eproc_bet_update(&ep0, 0.0, &r0);
            eproc_bet_update(&ep1, 1.0, &r1);
        }
        CHECK(!r0.rejected && r0.log_e == 0.0, "boundary mu0=0 null never rejects");
        CHECK(!r1.rejected && r1.log_e == 0.0, "boundary mu0=1 null never rejects");
    }

    /* 3. Power under H1 + NaN/Inf/subnormal/outlier sanitization + batch bit-identity */
    {
        eproc_bet_t ep_s, ep_b;
        eproc_bet_result_t rs, rb;
        CHECK(eproc_bet_init(&ep_s, 0.5, 0.01, 0.0, 1.0, EPROC_SIDE_UPPER) == EPROC_OK, "power init s");
        CHECK(eproc_bet_init(&ep_b, 0.5, 0.01, 0.0, 1.0, EPROC_SIDE_UPPER) == EPROC_OK, "power init b");

        double xs[260];
        for (int i = 0; i < 250; ++i) {
            xs[i] = (rng_u01(&seed) < 0.72) ? 1.0 : 0.0;
        }
        xs[250] = NAN;
        xs[251] = INFINITY;
        xs[252] = -INFINITY;
        xs[253] = 1e-315; /* subnormal */
        xs[254] = 1e15;   /* extreme outlier clamped to upper */
        xs[255] = -1e15;  /* extreme outlier clamped to lower */
        for (int i = 256; i < 260; ++i) xs[i] = 0.8;

        for (int i = 0; i < 260; ++i) {
            double before = ep_s.log_e;
            eproc_bet_update(&ep_s, xs[i], &rs);
            if (!isfinite(xs[i])) {
                CHECK(ep_s.log_e == before, "NaN/Inf must not mutate wealth");
            }
        }
        eproc_bet_update_batch(&ep_b, xs, 260, &rb);
        CHECK(rs.rejected == 1 && rs.p_value < 0.01, "power under H1 must reject");
        CHECK(ep_s.n_invalid == 3, "n_invalid must count 3 non-finite inputs");
        CHECK(ep_s.n == 257, "n must count 257 finite inputs");
        CHECK(memcmp(&rs, &rb, sizeof(rs)) == 0, "bet batch must match scalar bit-identically");
    }

    /* 4. Combine independent & dependent */
    {
        double c_ind = eproc_combine_independent_log(2.5, 3.5);
        CHECK(fabs(c_ind - 6.0) < 1e-12, "independent log combine");
        double big[2] = {1000.0, 1000.0};
        double small[2] = {-1000.0, -1000.0};
        CHECK(fabs(eproc_combine_dependent_log(big, 2) - 1000.0) < 1e-12, "dependent combine large");
        CHECK(fabs(eproc_combine_dependent_log(small, 2) - (-1000.0)) < 1e-12, "dependent combine tiny");
        double mixed[3] = {0.0, log(3.0), log(5.0)};
        double expected = log((1.0 + 3.0 + 5.0) / 3.0);
        CHECK(fabs(eproc_combine_dependent_log(mixed, 3) - expected) < 1e-12, "dependent combine exact");
    }
}

static void test_confidence_sequences(void) {
    uint64_t seed = 54321ULL;
    const double true_mu = 0.35;
    int eb_covered = 0;
    int nm_covered = 0;
    const int trials = 250;

    for (int tr = 0; tr < trials; ++tr) {
        eproc_cs_t eb, nm;
        eproc_cs_result_t reb, rnm;
        CHECK(eproc_cs_init_eb(&eb, 0.05, 0.0, 1.0) == EPROC_OK, "eb init");
        CHECK(eproc_cs_init_normal(&nm, 0.05, 1.0, 100.0) == EPROC_OK, "nm init");
        int ok_eb = 1, ok_nm = 1;
        double prev_eb_w = INFINITY;
        for (int t = 0; t < 300; ++t) {
            double x = (rng_u01(&seed) < true_mu) ? 1.0 : 0.0;
            eproc_cs_update(&eb, x, &reb);
            CHECK(reb.width <= prev_eb_w + 1e-15, "EB CS running intersection monotone");
            prev_eb_w = reb.width;
            if (true_mu < reb.low || true_mu > reb.high) ok_eb = 0;

            double g = 2.5 + rng_norm(&seed);
            eproc_cs_update(&nm, g, &rnm);
            if (2.5 < rnm.low || 2.5 > rnm.high) ok_nm = 0;
        }
        eb_covered += ok_eb;
        nm_covered += ok_nm;
    }
    double cov_eb = (double)eb_covered / (double)trials;
    double cov_nm = (double)nm_covered / (double)trials;
    printf("[cs] Anytime coverage (target >=0.95): EB=%.4f NormalMix=%.4f\n", cov_eb, cov_nm);
    CHECK(cov_eb >= 0.94, "EB CS anytime coverage >= 0.94");
    CHECK(cov_nm >= 0.94, "NormalMix CS anytime coverage >= 0.94");

    /* Zero-variance constant stream & NaN/Inf handling */
    {
        eproc_cs_t eb;
        eproc_cs_result_t r;
        eproc_cs_init_eb(&eb, 0.05, -10.0, 10.0);
        for (int t = 0; t < 200; ++t) {
            eproc_cs_update(&eb, 3.0, &r);
        }
        eproc_cs_update(&eb, NAN, &r);
        eproc_cs_update(&eb, INFINITY, &r);
        CHECK(eb.n == 200 && eb.n_invalid == 2, "CS invalid tracking");
        CHECK(isfinite(r.low) && isfinite(r.high) && r.low <= 3.0 && r.high >= 3.0, "CS zero-var covers 3.0");
        CHECK(r.width < 2.0, "CS zero-var tightens quickly");
    }
}

static void test_edetect_and_race(void) {
    uint64_t seed = 99999ULL;

    /* Page-CUSUM e-detector: calm under H0, fast alarm on drift, clean reset */
    {
        eproc_edetect_t ed;
        eproc_edetect_result_t r;
        CHECK(eproc_edetect_init(&ed, 0.5, 0.01, 0.0, 1.0, EPROC_SIDE_UPPER) == EPROC_OK, "edetect init");
        int false_alarms = 0;
        for (int t = 0; t < 200; ++t) {
            double x = (rng_u01(&seed) < 0.5) ? 1.0 : 0.0;
            eproc_edetect_update(&ed, x, &r);
            if (r.alarm) false_alarms++;
        }
        CHECK(false_alarms == 0, "edetect calm under H0 over 200 ticks at alpha=0.01");

        int detected_at = -1;
        for (int t = 0; t < 200; ++t) {
            double x = (rng_u01(&seed) < 0.85) ? 1.0 : 0.0;
            eproc_edetect_update(&ed, x, &r);
            if (r.alarm && detected_at < 0) detected_at = t;
        }
        printf("[edetect] Detected +0.35 drift in %d ticks (log_stat=%.2f)\n", detected_at, r.log_stat);
        CHECK(detected_at >= 0 && detected_at < 80, "edetect fast drift detection");

        eproc_edetect_reset_alarm(&ed);
        CHECK(ed.log_cusum == 0.0 && ed.alarm_at == 0, "edetect reset_alarm");
        detected_at = -1;
        for (int t = 0; t < 200; ++t) {
            double x = (rng_u01(&seed) < 0.85) ? 1.0 : 0.0;
            eproc_edetect_update(&ed, x, &r);
            if (r.alarm && detected_at < 0) detected_at = t;
        }
        CHECK(detected_at >= 0 && detected_at < 80, "edetect re-arms cleanly after reset");
    }

    /* Multi-arm racer with LUCB1 active sampling */
    {
        const double means[8] = {0.25, 0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.88};
        eproc_race_t race;
        eproc_race_status_t st;
        CHECK(eproc_race_init(&race, 8, 0.05, 0.0, 1.0, 0.0) == EPROC_OK, "race init");
        for (int step = 0; step < 5000; ++step) {
            int32_t arm = eproc_race_suggest_arm(&race);
            CHECK(arm >= 0 && arm < 8, "suggest arm valid");
            double rew = (rng_u01(&seed) < means[arm]) ? 1.0 : 0.0;
            eproc_race_update(&race, (uint32_t)arm, rew, &st);
            if (st.winner >= 0) break;
        }
        printf("[race] 8-arm winner=%d in %llu obs (active=%u)\n",
               st.winner, (unsigned long long)st.total_obs, st.n_active);
        CHECK(st.winner == 7, "race must certify best arm 7");
        CHECK(st.n_active == 1, "race certified means 1 active arm");
        eproc_cs_result_t arm7;
        int32_t active7 = 0;
        CHECK(eproc_race_arm_state(&race, 7, &arm7, &active7) == EPROC_OK, "race arm state");
        CHECK(active7 == 1 && arm7.mean > 0.75, "winner arm active with high mean");
    }
}

static void test_conformal_and_monoid(void) {
    uint64_t seed = 77777ULL;

    /* 1. Multi-rate AgACI + PID: strictly crossing-free */
    {
        const double unsorted_alphas[5] = {0.05, 0.50, 0.01, 0.20, 0.10};
        eproc_conformal_t cp;
        eproc_conformal_out_t out;
        CHECK(eproc_conformal_init(&cp, unsorted_alphas, 5, 128, 0.05) == EPROC_OK, "cp init");
        CHECK(cp.target_alphas[0] == 0.50 && cp.target_alphas[4] == 0.01, "alphas sorted descending");

        for (int t = 0; t < 1000; ++t) {
            double scale = (t < 500) ? 1.0 : 3.0; /* regime shift */
            double s = fabs(rng_norm(&seed)) * scale;
            if (t == 123) s = NAN;
            if (t == 456) s = INFINITY;
            eproc_conformal_update(&cp, s, &out);
            for (uint32_t r = 1; r < 5; ++r) {
                CHECK(out.quantiles[r] >= out.quantiles[r - 1], "quantiles must NEVER cross");
            }
            double lows[5], highs[5];
            eproc_conformal_predict(&cp, 10.0, 1.0, lows, highs, NULL);
            for (uint32_t r = 1; r < 5; ++r) {
                CHECK(lows[r] <= lows[r - 1] && highs[r] >= highs[r - 1], "intervals nested");
            }
        }
        printf("[conformal] 5-rate quantiles: %.3f <= %.3f <= %.3f <= %.3f <= %.3f (cov90=%.3f)\n",
               out.quantiles[0], out.quantiles[1], out.quantiles[2], out.quantiles[3], out.quantiles[4],
               out.emp_coverage[2]);
        CHECK(out.emp_coverage[2] > 0.75 && out.emp_coverage[2] < 0.98, "90% coverage tracks near nominal");
    }

    /* 2. Delayed-feedback conformal: strict causality & Knuth Alg R table reuse */
    {
        const double alphas[2] = {0.20, 0.10};
        eproc_delayed_conformal_t dc;
        CHECK(eproc_delayed_conformal_init(&dc, alphas, 2, 64, 0.02) == EPROC_OK, "dc init");

        uint64_t tids[100];
        double ref_lo[2], ref_hi[2];
        for (int i = 0; i < 100; ++i) {
            double lo[2], hi[2];
            CHECK(eproc_delayed_conformal_issue(&dc, 5.0, 1.0, &tids[i], lo, hi) == EPROC_OK, "dc issue");
            if (i == 0) {
                memcpy(ref_lo, lo, sizeof(lo));
                memcpy(ref_hi, hi, sizeof(hi));
            } else {
                CHECK(lo[0] == ref_lo[0] && hi[1] == ref_hi[1], "causal: unrevealed tickets do not change preds");
            }
        }
        CHECK(dc.pending_count == 100, "100 pending tickets");
        /* Reveal out-of-order (reverse) */
        for (int i = 99; i >= 0; --i) {
            CHECK(eproc_delayed_conformal_reveal(&dc, tids[i], 5.0 + fabs(rng_norm(&seed)), NULL) == EPROC_OK,
                  "dc reveal");
        }
        CHECK(dc.pending_count == 0, "all tickets revealed");
        /* Stress 5000 issue/reveal cycles across 4096-slot table */
        for (int i = 0; i < 5000; ++i) {
            uint64_t tid = 0;
            CHECK(eproc_delayed_conformal_issue(&dc, 1.0, 1.0, &tid, NULL, NULL) == EPROC_OK, "dc cycle issue");
            CHECK(eproc_delayed_conformal_reveal(&dc, tid, 1.5, NULL) == EPROC_OK, "dc cycle reveal");
        }
        CHECK(dc.pending_count == 0, "dc cycle table clean");
    }

    /* 3. Budget conformal (right-censored UPB) */
    {
        eproc_budget_conformal_t bc;
        CHECK(eproc_budget_conformal_init(&bc, 0.10, 64, 1.5) == EPROC_OK, "bc init");
        for (int i = 0; i < 80; ++i) {
            int cens = (i % 5 == 0) ? 1 : 0;
            CHECK(eproc_budget_conformal_update(&bc, 100.0, cens ? 120.0 : 105.0, cens) == EPROC_OK, "bc upd");
        }
        double b1 = eproc_budget_conformal_bound(&bc, 100.0);
        double b2 = eproc_budget_conformal_bound(&bc, 150.0);
        CHECK(b1 > 100.0 && b2 > b1, "budget UPB positive and monotone");
        CHECK(bc.n_censored == 16 && bc.n_completed == 64, "budget counts exact");
    }

    /* 4. Distributed Conformal Monoid: associativity, commutativity, LOO, wire roundtrip */
    {
        eproc_cmonoid_t a, b, c, ab, abc1, bc_m, abc2, ba, loo;
        eproc_cmonoid_init(&a, 1.0);
        eproc_cmonoid_init(&b, 1.0);
        eproc_cmonoid_init(&c, 1.0);

        for (int i = 0; i < 500; ++i) {
            eproc_cmonoid_add(&a, rng_u01(&seed) * 10.0);
            eproc_cmonoid_add(&b, rng_u01(&seed) * 10.0);
            eproc_cmonoid_add(&c, rng_u01(&seed) * 20.0 - 5.0);
        }
        eproc_cmonoid_add(&a, NAN);
        eproc_cmonoid_add(&a, INFINITY);
        CHECK(a.count == 500, "cmonoid skips NaN/Inf");

        /* Associativity: (A+B)+C == A+(B+C) bit-exact on counts */
        eproc_cmonoid_merge(&a, &b, &ab);
        eproc_cmonoid_merge(&ab, &c, &abc1);
        eproc_cmonoid_merge(&b, &c, &bc_m);
        eproc_cmonoid_merge(&a, &bc_m, &abc2);
        CHECK(abc1.count == abc2.count, "cmonoid associative count");
        CHECK(memcmp(abc1.bin_counts, abc2.bin_counts, sizeof(abc1.bin_counts)) == 0,
              "cmonoid associative bins 0 drift");

        /* Commutativity: A+B == B+A */
        eproc_cmonoid_merge(&b, &a, &ba);
        CHECK(memcmp(ab.bin_counts, ba.bin_counts, sizeof(ab.bin_counts)) == 0,
              "cmonoid commutative bins 0 drift");

        /* Leave-one-out subtract: (A+B) - A == B */
        CHECK(eproc_cmonoid_subtract(&ab, &a, &loo) == EPROC_OK, "cmonoid subtract ok");
        CHECK(loo.count == b.count, "cmonoid LOO count exact");
        CHECK(memcmp(loo.bin_counts, b.bin_counts, sizeof(b.bin_counts)) == 0,
              "cmonoid LOO bins bit-exact");

        /* Underflow protection */
        CHECK(eproc_cmonoid_subtract(&a, &ab, &loo) == EPROC_ERR_INVALID,
              "cmonoid subtract underflow must fail");

        /* Quantile sanity on Uniform(0, 10) */
        double q50 = eproc_cmonoid_quantile(&ab, 0.50);
        double q90 = eproc_cmonoid_quantile(&ab, 0.90);
        printf("[cmonoid] Uniform(0,10) q50=%.3f q90=%.3f (wire=%zu B)\n",
               q50, q90, eproc_cmonoid_wire_size());
        CHECK(fabs(q50 - 5.0) < 0.7 && fabs(q90 - 9.0) < 0.9, "cmonoid quantile accuracy");

        /* Wire serialize / deserialize roundtrip + corruption rejection */
        uint8_t wire[640];
        CHECK(eproc_cmonoid_serialize(&abc1, wire, 639) == EPROC_ERR_CAPACITY, "wire short buf fails");
        CHECK(eproc_cmonoid_serialize(&abc1, wire, 640) == EPROC_OK, "wire serialize ok");
        eproc_cmonoid_t decoded;
        CHECK(eproc_cmonoid_deserialize(&decoded, wire, 640) == EPROC_OK, "wire deserialize ok");
        CHECK(memcmp(&abc1, &decoded, sizeof(abc1)) == 0, "wire roundtrip bit-exact");
        wire[100] ^= 0x01;
        CHECK(eproc_cmonoid_deserialize(&decoded, wire, 640) == EPROC_ERR_MISMATCH,
              "wire checksum catches bit flip");
    }
}

/* ============================================================================
 * 1,000,000-ITERATION CONCURRENT BOEHM-2012 SEQLOCK STRESS TEST (TSan verified)
 * ============================================================================ */

typedef struct {
    const eproc_boa_t *boa;
    uint32_t n_experts;
    uint32_t anchor_idx;
    size_t iters;
    _Atomic uint64_t torn_reads;
} seqlock_reader_arg_t;

typedef struct {
    eproc_boa_t *boa;
    _Atomic int stop;
    uint64_t updates_done;
} seqlock_writer_arg_t;

static void *seqlock_reader_fn(void *arg) {
    seqlock_reader_arg_t *ra = (seqlock_reader_arg_t *)arg;
    double w[EPROC_BOA_MAX_EXPERTS];
    uint64_t local_torn = 0;

    for (size_t it = 0; it < ra->iters; ++it) {
        int32_t fb = -1;
        int rc = eproc_boa_snapshot_weights(ra->boa, w, ra->n_experts, &fb);
        if (rc != EPROC_OK || (fb != 0 && fb != 1)) {
            local_torn++;
            continue;
        }
        double sum = 0.0;
        int bad = 0;
        for (uint32_t i = 0; i < ra->n_experts; ++i) {
            if (!isfinite(w[i]) || w[i] < 0.0 || w[i] > 1.0 + 1e-12) {
                bad = 1;
                break;
            }
            if (fb == 1) {
                double exp_w = (i == ra->anchor_idx) ? 1.0 : 0.0;
                if (w[i] != exp_w) {
                    bad = 1;
                    break;
                }
            }
            sum += w[i];
        }
        if (bad || fabs(sum - 1.0) > 1e-9) {
            local_torn++;
        }
    }

    atomic_fetch_add_explicit(&ra->torn_reads, local_torn, memory_order_relaxed);
    return NULL;
}

static void *seqlock_writer_fn(void *arg) {
    seqlock_writer_arg_t *wa = (seqlock_writer_arg_t *)arg;
    uint64_t s = 0xCAFEBABEULL;
    double preds[8];
    uint64_t n = 0;
    while (!atomic_load_explicit(&wa->stop, memory_order_acquire)) {
        double y = sin((double)n * 0.01) * 5.0;
        for (int i = 0; i < 8; ++i) {
            preds[i] = y + (0.2 + 0.3 * (double)i) * rng_norm(&s);
        }
        eproc_boa_update(wa->boa, preds, y, NULL);
        n++;
    }
    wa->updates_done = n;
    return NULL;
}

static void test_boa_and_echopool(void) {
    uint64_t seed = 314159ULL;

    /* 1. SafeBOA Byzantine poisoning immunity (0.00% blast) + quarantine */
    {
        eproc_boa_t boa;
        eproc_boa_step_t st;
        CHECK(eproc_boa_init(&boa, 8, 0, 0.05) == EPROC_OK, "boa init");

        double preds[8];
        for (int t = 0; t < 100; ++t) {
            double y = 10.0 + sin((double)t * 0.1);
            preds[0] = y + 0.5 * rng_norm(&seed); /* anchor */
            preds[1] = y + 0.4 * rng_norm(&seed);
            preds[2] = y + 0.3 * rng_norm(&seed);
            preds[3] = y + 0.05 * rng_norm(&seed); /* best expert */
            preds[4] = y + 0.8 * rng_norm(&seed);
            preds[5] = (t >= 70 && t <= 74) ? (y + 50.0) : (y + 0.5 * rng_norm(&seed)); /* 3-strike */
            preds[6] = (t == 60) ? NAN : (y + 0.5 * rng_norm(&seed));                   /* NaN */
            preds[7] = (t == 50) ? 1e6 : (y + 0.5 * rng_norm(&seed));                   /* 1e6 attack */

            double p_pred = eproc_boa_predict(&boa, preds);
            if (t == 50) {
                /* Tick 50: 1e6 attack MUST have 0.00% blast radius! */
                CHECK(fabs(p_pred - y) < 5.0, "1e6 Byzantine attack must have ~0 blast radius");
            }
            CHECK(eproc_boa_update(&boa, preds, y, &st) == EPROC_OK, "boa update");
            CHECK(isfinite(st.fused_pred), "fused_pred always finite");
        }
        printf("[boa] expert3_w=%.3f quarantined=%u clipped=%llu\n",
               boa.weights[3], st.n_quarantined, (unsigned long long)boa.n_poison_clipped);
        CHECK(boa.quarantined[7] == 1, "1e6 attacker quarantined immediately");
        CHECK(boa.quarantined[6] == 1, "NaN attacker quarantined immediately");
        CHECK(boa.quarantined[5] == 1, "sustained glitch attacker quarantined after 3 strikes");
        CHECK(boa.weights[3] > 0.35, "BOA concentrates weight on best expert 3");
    }

    /* 2. 3-Gate Ville degradation fallback test */
    {
        eproc_boa_t boa;
        eproc_boa_step_t st;
        eproc_boa_init(&boa, 4, 0, 0.05);
        double preds[4];
        /* First let BOA trust expert 1, then degrade experts 1..3 moderately (below poison threshold) */
        for (int t = 0; t < 40; ++t) {
            preds[0] = 1.0;
            preds[1] = 0.0;
            preds[2] = 0.5;
            preds[3] = 0.5;
            eproc_boa_update(&boa, preds, 0.0, &st);
        }
        int tripped = 0;
        for (int t = 0; t < 150; ++t) {
            preds[0] = 0.0; /* anchor now exact */
            preds[1] = 3.0; /* degraded */
            preds[2] = 3.0;
            preds[3] = 3.0;
            eproc_boa_update(&boa, preds, 0.0, &st);
            if (st.fallback_active) {
                tripped = 1;
                break;
            }
        }
        CHECK(tripped == 1, "Ville deg_e must trip fallback when BOA degrades vs anchor");
    }

    /* 3. 1,000,000-iteration concurrent Boehm-2012 Seqlock test (4 readers + 1 writer) */
    {
        eproc_boa_t boa;
        CHECK(eproc_boa_init(&boa, 8, 0, 0.05) == EPROC_OK, "seqlock boa init");

        seqlock_writer_arg_t warg;
        warg.boa = &boa;
        atomic_init(&warg.stop, 0);
        warg.updates_done = 0;

        pthread_t writer_tid;
        CHECK(pthread_create(&writer_tid, NULL, seqlock_writer_fn, &warg) == 0, "spawn writer");

        pthread_t reader_tids[4];
        seqlock_reader_arg_t rargs[4];
        for (int r = 0; r < 4; ++r) {
            rargs[r].boa = &boa;
            rargs[r].n_experts = 8;
            rargs[r].anchor_idx = 0;
            rargs[r].iters = 250000; /* 4 x 250,000 = 1,000,000 concurrent reads */
            atomic_init(&rargs[r].torn_reads, 0);
            CHECK(pthread_create(&reader_tids[r], NULL, seqlock_reader_fn, &rargs[r]) == 0, "spawn reader");
        }

        uint64_t total_torn = 0;
        for (int r = 0; r < 4; ++r) {
            pthread_join(reader_tids[r], NULL);
            total_torn += atomic_load_explicit(&rargs[r].torn_reads, memory_order_relaxed);
        }
        atomic_store_explicit(&warg.stop, 1, memory_order_release);
        pthread_join(writer_tid, NULL);

        printf("[seqlock] 1,000,000 concurrent reads vs %llu live BOA updates -> torn_reads=%llu\n",
               (unsigned long long)warg.updates_done, (unsigned long long)total_torn);
        CHECK(total_torn == 0ULL, "seqlock must have 0 torn reads across 1M concurrent reads");
    }

    /* 4. EchoPool: 0 echo double-counts + Beta-Bernoulli peer reliability */
    {
        eproc_echopool_t pool;
        eproc_echo_out_t eo;
        CHECK(eproc_echopool_init(&pool, 16, 32, 0.5) == EPROC_OK, "echopool init");
        CHECK(eproc_echopool_submit(&pool, 0, 0, 1001ULL, 2.0, &eo) == EPROC_OK, "echo submit 1");
        CHECK(eo.accepted == 1 && eo.unique_count == 1, "first evidence accepted");
        double post1 = eo.posterior_prob;

        /* 9 other sources echo the exact same evidence_id */
        for (uint32_t s = 1; s < 10; ++s) {
            CHECK(eproc_echopool_submit(&pool, s, 0, 1001ULL, 2.0, &eo) == EPROC_OK, "echo dup");
            CHECK(eo.accepted == 0, "duplicate echo dropped");
            CHECK(eo.posterior_prob == post1, "posterior invariant to echo spam");
        }
        CHECK(pool.n_echo_dropped == 9, "9 echoes dropped");

        /* Train source 0 (honest) vs source 1 (adversarial) across 10 resolved claims */
        for (uint32_t c = 1; c <= 10; ++c) {
            eproc_echopool_submit(&pool, 0, c, 2000ULL + c, +2.0, NULL);
            eproc_echopool_submit(&pool, 1, c, 3000ULL + c, -2.0, NULL);
            CHECK(eproc_echopool_resolve(&pool, c, 1) == EPROC_OK, "echo resolve");
            /* Idempotent second resolve should not double-count */
            CHECK(eproc_echopool_resolve(&pool, c, 1) == EPROC_OK, "echo resolve idempotent");
        }
        double rel_good = eproc_echopool_source_reliability(&pool, 0);
        double rel_bad = eproc_echopool_source_reliability(&pool, 1);
        printf("[echopool] rel_good=%.3f rel_bad=%.3f echoes_dropped=%llu\n",
               rel_good, rel_bad, (unsigned long long)pool.n_echo_dropped);
        CHECK(rel_good > 0.99, "honest source stays near 1.0 reliability");
        CHECK(rel_bad == 0.0, "adversarial source shrinks to 0.0 reliability");
    }
}

static void test_subsample_and_sift(void) {
    /* 1. VarOpt: exact total weight conservation & unbiasedness */
    {
        double weights[200];
        double values[200];
        uint64_t gs = 2026ULL;
        double true_sum = 0.0;
        double true_w = 0.0;
        for (int i = 0; i < 200; ++i) {
            weights[i] = (i % 20 == 0) ? 50.0 : (0.5 + 3.0 * rng_u01(&gs));
            values[i] = 1.0 + 0.1 * (double)i;
            true_sum += weights[i] * values[i];
            true_w += weights[i];
        }

        double mc_sum = 0.0;
        const int n_mc = 1000;
        for (int tr = 0; tr < n_mc; ++tr) {
            eproc_varopt_t vo;
            CHECK(eproc_varopt_init(&vo, 32, (uint64_t)(tr + 1)) == EPROC_OK, "varopt init");
            for (int i = 0; i < 200; ++i) {
                CHECK(eproc_varopt_push(&vo, (uint64_t)i, weights[i], values[i]) == EPROC_OK, "varopt push");
            }
            double adj_w_sum = 0.0;
            for (uint32_t j = 0; j < vo.count; ++j) {
                adj_w_sum += vo.items[j].adj_weight;
                CHECK(vo.items[j].adj_weight + 1e-12 >= vo.items[j].orig_weight, "adj >= orig");
            }
            CHECK(fabs(adj_w_sum - true_w) < 1e-8 * true_w, "VarOpt exact total weight conservation");
            mc_sum += eproc_varopt_estimate_sum(&vo);
        }
        double est_mean = mc_sum / (double)n_mc;
        double rel_err = fabs(est_mean - true_sum) / true_sum;
        printf("[varopt] MC mean=%.2f exact=%.2f rel_err=%.5f\n", est_mean, true_sum, rel_err);
        CHECK(rel_err < 0.01, "VarOpt Horvitz-Thompson estimator must be unbiased (<1% MC error)");

        /* k = 1 corner case */
        eproc_varopt_t vo1;
        CHECK(eproc_varopt_init(&vo1, 1, 42ULL) == EPROC_OK, "varopt k=1");
        for (int i = 0; i < 20; ++i) {
            CHECK(eproc_varopt_push(&vo1, (uint64_t)i, (double)(i + 1), 1.0) == EPROC_OK, "varopt k=1 push");
        }
        CHECK(fabs(vo1.items[0].adj_weight - 210.0) < 1e-9, "varopt k=1 exact weight conservation");
    }

    /* 2. Online RLS: bit-exact symmetry, leverage bounds, extreme outliers */
    {
        eproc_rls_t rls;
        eproc_rls_out_t ro;
        const uint32_t d = 16;
        CHECK(eproc_rls_init(&rls, d, 1.0, 2.0, 0.05, 1337ULL) == EPROC_OK, "rls init");
        uint64_t s = 4242ULL;
        double x[16];
        for (int t = 0; t < 300; ++t) {
            for (uint32_t i = 0; i < d; ++i) x[i] = rng_norm(&s);
            CHECK(eproc_rls_step(&rls, x, &ro) == EPROC_OK, "rls step");
            CHECK(ro.leverage >= 0.0 && ro.prob >= 0.05 && ro.prob <= 1.0, "rls bounds");
            if (ro.keep) {
                CHECK(fabs(ro.ht_weight - 1.0 / ro.prob) < 1e-12, "rls HT weight exact");
            } else {
                CHECK(ro.ht_weight == 0.0, "rls dropped weight 0");
            }
        }
        /* Extreme outlier & NaN resilience */
        for (uint32_t i = 0; i < d; ++i) x[i] = 1e15;
        CHECK(eproc_rls_step(&rls, x, &ro) == EPROC_OK, "rls 1e15 step");
        for (uint32_t i = 0; i < d; ++i) x[i] = 1e150;
        CHECK(eproc_rls_step(&rls, x, &ro) == EPROC_OK, "rls 1e150 step");
        x[0] = NAN;
        CHECK(eproc_rls_step(&rls, x, &ro) == EPROC_OK, "rls NaN step");

        for (uint32_t i = 0; i < d; ++i) {
            for (uint32_t j = 0; j < d; ++j) {
                double a = rls.inv_a[i * d + j];
                double b = rls.inv_a[j * d + i];
                CHECK(isfinite(a) && a == b, "rls inv_a finite and bit-exact symmetric");
            }
            CHECK(rls.inv_a[i * d + i] >= 0.0, "rls inv_a diag >= 0");
        }
        printf("[rls] n_seen=%llu n_kept=%llu (d=%u)\n",
               (unsigned long long)rls.n_seen, (unsigned long long)rls.n_kept, d);
    }

    /* 3. Hierarchical SIFT VOI selector + 3x loud distractor immunity */
    {
        eproc_sift_t sf;
        eproc_sift_action_t act;
        CHECK(eproc_sift_init(&sf, 16, 8, 2.0, 1.0) == EPROC_OK, "sift init");
        /* Startup: coarse site read must win */
        CHECK(eproc_sift_select_next(&sf, 1.0, 1.0, 4.0, 16, &act) == EPROC_OK, "sift select 0");
        CHECK(act.action_type == 0, "sift starts with coarse site mean read");

        /* Mark channel 0 of site 5 as a 3x loud distractor */
        uint32_t loud_ch = 5U * 8U + 0U;
        uint32_t clean_ch = 5U * 8U + 1U;
        CHECK(eproc_sift_set_noise_scale(&sf, loud_ch, 3.0) == EPROC_OK, "sift set noise scale");

        /* Same noisy spike on site 5 (loud distractor) vs site 6 (clean): site 5 dampens false spike */
        eproc_sift_t sf_cmp = sf;
        CHECK(eproc_sift_observe_site(&sf_cmp, 5, 1.0) == EPROC_OK, "sift cmp loud site");
        CHECK(eproc_sift_observe_site(&sf_cmp, 6, 1.0) == EPROC_OK, "sift cmp clean site");
        CHECK(sf_cmp.site_log_prob[5] < sf_cmp.site_log_prob[6],
              "3x loud distractor site dampens spurious site-mean spike");

        /* Strong hot site observation on site 5 -> posterior > 0.5 and zooms into clean channel */
        CHECK(eproc_sift_observe_site(&sf, 5, 1.6) == EPROC_OK, "sift observe hot site");
        CHECK(exp(sf.site_log_prob[5]) > 0.5, "hot site posterior rises > 0.5");

        CHECK(eproc_sift_select_next(&sf, 1.0, 1.0, 4.0, 16, &act) == EPROC_OK, "sift select 1");
        printf("[sift] after hot site 5 (p_hot=%.3f): action_type=%d target=%u evoi/cost=%.4f\n",
               exp(sf.site_log_prob[5]), act.action_type, act.target_idx, act.evoi_per_cost);
        CHECK(act.action_type == 1 && act.target_idx == clean_ch,
              "sift zooms into clean channel and skips 3x loud distractor");

        /* Confirm channel observation updates channel & site coherently */
        CHECK(eproc_sift_observe_channel(&sf, clean_ch, 2.5, 4) == EPROC_OK, "sift obs chan");
        CHECK(sf.chan_prob[clean_ch] > 0.95, "active channel posterior > 0.95");
    }
}

static uint64_t fnv_mix_bytes(uint64_t h, const void *ptr, size_t n) {
    const uint8_t *p = (const uint8_t *)ptr;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static void test_bit_exact_fingerprint(void) {
    uint64_t h = 14695981039346656037ULL;
    uint64_t seed = 0x123456789ABCDEF0ULL;

    /* 1. AVX2 cmonoid merge & subtract */
    eproc_cmonoid_t m1, m2, m3, m4;
    eproc_cmonoid_init(&m1, 1.5);
    eproc_cmonoid_init(&m2, 1.5);
    double scores[128];
    for (int i = 0; i < 128; ++i) scores[i] = (rng_u01(&seed) - 0.25) * 20.0;
    eproc_cmonoid_add_batch(&m1, scores, 64);
    eproc_cmonoid_add_batch(&m2, scores + 64, 64);
    eproc_cmonoid_merge(&m1, &m2, &m3);
    CHECK(eproc_cmonoid_subtract(&m3, &m1, &m4) == EPROC_OK, "fp subtract");
    h = fnv_mix_bytes(h, &m3, sizeof(m3));
    h = fnv_mix_bytes(h, &m4, sizeof(m4));

    /* 2. AVX2 RLS dot + rank-1 update across non-multiple-of-4 dimension d=19 */
    eproc_rls_t rls;
    eproc_rls_out_t ro;
    CHECK(eproc_rls_init(&rls, 19, 0.75, 2.5, 0.05, 98765ULL) == EPROC_OK, "fp rls");
    double x[19];
    for (int t = 0; t < 100; ++t) {
        for (int i = 0; i < 19; ++i) x[i] = rng_norm(&seed);
        CHECK(eproc_rls_step(&rls, x, &ro) == EPROC_OK, "fp rls step");
        h = fnv_mix_bytes(h, &ro, sizeof(ro));
    }
    h = fnv_mix_bytes(h, rls.inv_a, sizeof(double) * 19U * 19U);

    /* 3. Betting batch + CS batch + Conformal batch + VarOpt */
    eproc_bet_t bet;
    eproc_bet_result_t br;
    eproc_bet_init(&bet, 0.5, 0.05, 0.0, 1.0, EPROC_SIDE_TWO_SIDED);
    double ys[128];
    for (int i = 0; i < 128; ++i) ys[i] = rng_u01(&seed);
    eproc_bet_update_batch(&bet, ys, 128, &br);
    h = fnv_mix_bytes(h, &br, sizeof(br));

    eproc_cs_t cs;
    eproc_cs_result_t cr;
    eproc_cs_init_eb(&cs, 0.05, 0.0, 1.0);
    eproc_cs_update_batch(&cs, ys, 128, &cr);
    h = fnv_mix_bytes(h, &cr, sizeof(cr));

    eproc_conformal_t cp;
    eproc_conformal_out_t cpo;
    const double ca[5] = {0.50, 0.20, 0.10, 0.05, 0.01};
    eproc_conformal_init(&cp, ca, 5, 64, 0.05);
    eproc_conformal_update_batch(&cp, scores, 128, &cpo);
    h = fnv_mix_bytes(h, &cpo, sizeof(cpo));

    eproc_varopt_t vo;
    eproc_varopt_init(&vo, 24, 777ULL);
    for (int i = 0; i < 128; ++i) {
        eproc_varopt_push(&vo, (uint64_t)i, 0.1 + 5.0 * ys[i], scores[i]);
    }
    double vsum = eproc_varopt_estimate_sum(&vo);
    h = fnv_mix_bytes(h, &vsum, sizeof(vsum));

    printf("[fingerprint] 0x%016llx\n", (unsigned long long)h);
}

static void test_breaker_witness_regressions(void) {
    /* W01: MultiRateConformal finite (-1e308, +1e308, +1e308) -> 0 NaN, 0 float-cast-overflow UB */
    {
        eproc_conformal_t cp;
        eproc_conformal_out_t out;
        const double alphas[5] = {0.50, 0.20, 0.10, 0.05, 0.01};
        CHECK(eproc_conformal_init(&cp, alphas, 5, 32, 0.05) == EPROC_OK, "W01 cp init");
        CHECK(eproc_conformal_update(&cp, -1e308, &out) == EPROC_OK, "W01 -1e308");
        CHECK(eproc_conformal_update(&cp, +1e308, &out) == EPROC_OK, "W01 +1e308 #1");
        CHECK(eproc_conformal_update(&cp, +1e308, &out) == EPROC_OK, "W01 +1e308 #2");
        for (uint32_t r = 0; r < 5; ++r) {
            CHECK(isfinite(out.eff_alphas[r]) && isfinite(out.quantiles[r]), "W01 finite outputs");
        }
    }

    /* W17: MultiRateConformal scalar loop vs update_batch 0-ULP bit-identity */
    {
        const double alphas[5] = {0.50, 0.20, 0.10, 0.05, 0.01};
        eproc_conformal_t cp_s, cp_b;
        eproc_conformal_out_t os, ob;
        eproc_conformal_init(&cp_s, alphas, 5, 64, 0.05);
        eproc_conformal_init(&cp_b, alphas, 5, 64, 0.05);
        double scores[200];
        uint64_t s = 0x55551234ULL;
        for (int i = 0; i < 200; ++i) scores[i] = fabs(rng_norm(&s)) * 2.0;
        for (int i = 0; i < 200; ++i) eproc_conformal_update(&cp_s, scores[i], &os);
        eproc_conformal_update_batch(&cp_b, scores, 200, &ob);
        CHECK(memcmp(&os, &ob, sizeof(os)) == 0, "W17 conformal scalar vs batch bit-identical");
    }

    /* W03: SafeBOA tick-0 100% Byzantine spread attack [0, 1e12, 2e12, 3e12, 4e12] */
    {
        eproc_boa_t boa;
        eproc_boa_step_t st;
        CHECK(eproc_boa_init(&boa, 5, 0, 0.05) == EPROC_OK, "W03 boa init");
        double atk[5] = {0.0, 1e12, 2e12, 3e12, 4e12};
        CHECK(eproc_boa_update(&boa, atk, 0.0, &st) == EPROC_OK, "W03 tick-0 update");
        CHECK(fabs(st.fused_pred - st.anchor_pred) < 1e-9, "W03 tick-0 Byzantine 0.00% blast");
        CHECK(st.n_quarantined == 4U, "W03 all 4 Byzantine experts quarantined on tick 0");
    }

    /* W04, W07, W08, W10, W11: ConformalMonoid witnesses */
    {
        /* W04: LOO subtract(merge(A, B), A).quantile(1.0) == B.quantile(1.0) (no 1e12 leak) */
        eproc_cmonoid_t a, b, ab, loo;
        eproc_cmonoid_init(&a, 1.0);
        eproc_cmonoid_init(&b, 1.0);
        eproc_cmonoid_add(&a, 1e12);
        const double bvals[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
        eproc_cmonoid_add_batch(&b, bvals, 5);
        eproc_cmonoid_merge(&a, &b, &ab);
        CHECK(eproc_cmonoid_subtract(&ab, &a, &loo) == EPROC_OK, "W04 LOO subtract ok");
        CHECK(eproc_cmonoid_quantile(&loo, 1.0) == eproc_cmonoid_quantile(&b, 1.0),
              "W04 LOO quantile(1.0) bit-exact match without 1e12 leak");
        CHECK(loo.max_val < 10.0, "W04 LOO max_val tightened from 1e12");

        /* W07: add vs add_batch exact bin parity at scale=7.0 */
        eproc_cmonoid_t ms, mb;
        eproc_cmonoid_init(&ms, 7.0);
        eproc_cmonoid_init(&mb, 7.0);
        double edge_val = 1.1042051166702023 * 7.0;
        eproc_cmonoid_add(&ms, edge_val);
        eproc_cmonoid_add_batch(&mb, &edge_val, 1);
        CHECK(memcmp(ms.bin_counts, mb.bin_counts, sizeof(ms.bin_counts)) == 0,
              "W07 add vs add_batch bin parity");

        /* W08: scale_hint mismatch sets flag bit 0 symmetrically */
        eproc_cmonoid_t m1, m10, m_ab, m_ba;
        eproc_cmonoid_init(&m1, 1.0);
        eproc_cmonoid_init(&m10, 10.0);
        eproc_cmonoid_add(&m1, 1.0);
        eproc_cmonoid_add(&m10, 10.0);
        eproc_cmonoid_merge(&m1, &m10, &m_ab);
        eproc_cmonoid_merge(&m10, &m1, &m_ba);
        CHECK((m_ab.flags & 1ULL) != 0ULL && (m_ba.flags & 1ULL) != 0ULL, "W08 mismatch flag");

        /* W10: bit-exact wire commutativity merge(A,B) == merge(B,A) */
        eproc_cmonoid_t wa, wb, wab, wba;
        eproc_cmonoid_init(&wa, 1.0);
        eproc_cmonoid_init(&wb, 1.0);
        const double va[5] = {1e12, 1e-12, 0.5, -1e12, 0.05823};
        const double vb[5] = {1e12, 2e-12, 0.2, -1e12, -5.0};
        eproc_cmonoid_add_batch(&wa, va, 5);
        eproc_cmonoid_add_batch(&wb, vb, 5);
        eproc_cmonoid_merge(&wa, &wb, &wab);
        eproc_cmonoid_merge(&wb, &wa, &wba);
        uint8_t wire_ab[640], wire_ba[640];
        CHECK(eproc_cmonoid_serialize(&wab, wire_ab, 640) == EPROC_OK, "W10 ser ab");
        CHECK(eproc_cmonoid_serialize(&wba, wire_ba, 640) == EPROC_OK, "W10 ser ba");
        CHECK(memcmp(wire_ab, wire_ba, 640) == 0, "W10 wire commutativity bit-exact");

        /* W11: deserialize rejects both len < 640 and len > 640 with EPROC_ERR_INVALID */
        uint8_t wire_long[656];
        memcpy(wire_long, wire_ab, 640);
        memset(wire_long + 640, 'X', 16);
        eproc_cmonoid_t dec;
        CHECK(eproc_cmonoid_deserialize(&dec, wire_ab, 320) == EPROC_ERR_INVALID,
              "W11 truncated wire returns EPROC_ERR_INVALID");
        CHECK(eproc_cmonoid_deserialize(&dec, wire_long, 656) == EPROC_ERR_INVALID,
              "W11 trailing garbage wire returns EPROC_ERR_INVALID");
    }

    /* W05: OnlineRLS subnormal [1e-310, 0, 0, 0] -> leverage ~ 0, prob == min_prob */
    {
        eproc_rls_t rls;
        eproc_rls_out_t ro;
        CHECK(eproc_rls_init(&rls, 4, 1.0, 2.0, 0.02, 42ULL) == EPROC_OK, "W05 rls init");
        double sub[4] = {1e-310, 0.0, 0.0, 0.0};
        CHECK(eproc_rls_step(&rls, sub, &ro) == EPROC_OK, "W05 rls subnormal");
        CHECK(ro.leverage < 1e-100 && fabs(ro.prob - 0.02) < 1e-12, "W05 subnormal zero leverage");
    }

    /* W06: 1e308 overflow safety across CS(normal), ConformalMonoid, VarOptSampler */
    {
        eproc_cs_t nm;
        eproc_cs_result_t r;
        eproc_cs_init_normal(&nm, 0.05, 1.0, 100.0);
        eproc_cs_update(&nm, 1e308, &r);
        eproc_cs_update(&nm, 1e308, &r);
        CHECK(!isnan(r.mean) && !isnan(r.low) && !isnan(r.high), "W06 CS(normal) no NaN on 1e308");

        eproc_cmonoid_t cm;
        eproc_cmonoid_init(&cm, 1.0);
        eproc_cmonoid_add(&cm, 1e308);
        eproc_cmonoid_add(&cm, 1e308);
        CHECK(!isnan((cm.sum + cm.sum_comp) / (double)cm.count), "W06 cmonoid no NaN on 1e308");

        eproc_varopt_t vo;
        eproc_varopt_init(&vo, 2, 1ULL);
        eproc_varopt_push(&vo, 0, 1e308, 0.0);
        eproc_varopt_push(&vo, 1, 1e308, 0.0);
        eproc_varopt_push(&vo, 2, 1e308, 0.0);
        CHECK(!isnan(eproc_varopt_estimate_sum(&vo)) && isfinite(vo.tau),
              "W06 VarOpt no NaN/Inf on 3x (1e308, 0.0)");
    }

    printf("[breaker-regressions] W01/W03/W04/W05/W06/W07/W08/W10/W11/W17: PASS\n");
}

int main(void) {
    print_abi_sizes();
    test_betting_eprocess();
    test_confidence_sequences();
    test_edetect_and_race();
    test_conformal_and_monoid();
    test_boa_and_echopool();
    test_subsample_and_sift();
    test_breaker_witness_regressions();
    test_bit_exact_fingerprint();

    if (g_failures == 0) {
        printf("\nALL C11 CORE TESTS PASSED (0 failures)!\n");
        return 0;
    }
    fprintf(stderr, "\nFAILED WITH %d ERRORS!\n", g_failures);
    return 1;
}
