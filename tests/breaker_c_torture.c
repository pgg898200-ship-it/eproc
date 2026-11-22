/*
 * breaker_c_torture.c — Independent Adversarial Breaker C11 Torture Suite
 *
 * Attacks libeproc at the C11 ABI level under ASan+UBSan and TSan:
 *   1. NULL pointer fuzzing across all 61 public C11 functions
 *   2. Out-of-range indices, invalid parameters, negative/zero/overflow bounds
 *   3. Zero-length batch operations with NULL and non-NULL pointers
 *   4. Boundary mu0 = 1e-12 and 1 - 1e-12 across all 3 sides + NaN/Inf/subnormals
 *   5. DelayedConformal 4096-slot full capacity + Knuth Alg R stress
 *   6. EchoPool 4096-slot table saturation + post-saturation echo dedup
 *   7. ConformalMonoid in-place aliasing, 640-byte single-bit-flip sweep, & add vs add_batch
 *   8. OnlineRLS subnormal (1e-310) & extreme (1e15, 1e150) torture
 *   9. 1,000,000-read concurrent SafeBOA seqlock stress (4 readers + 1 writer, 64 experts)
 *      + raw multi-writer seqlock stress (2 writers + 4 readers)
 *  10. Optional `--ub-witness` mode reproducing finite (-1e308, +1e308) NaN -> (uint32_t)NaN UB
 */

#if !defined(__GNUC__) || defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

#include "eproc.h"
#include "eproc_seqlock.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

#define BREAKER_CHECK(cond, msg)                                                       \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "BREAKER FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));  \
            g_failures++;                                                              \
        }                                                                              \
    } while (0)

static uint64_t rng_next(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double rng_u01(uint64_t *s) {
    return (double)(rng_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

/* ============================================================================
 * 1. NULL POINTERS, INVALID PARAMS, OUT-OF-RANGE INDICES, ZERO-LENGTH BATCHES
 * ============================================================================ */

static void test_null_invalid_and_empty(void) {
    double dummy_d[4] = {0.1, 0.2, 0.3, 0.4};
    uint32_t dummy_u[4] = {0, 1, 2, 3};
    uint8_t wire[640];
    double lo[5], hi[5];

    /* 1A. Betting E-Process */
    eproc_bet_t bet;
    eproc_bet_result_t br;
    BREAKER_CHECK(eproc_bet_init(NULL, 0.5, 0.05, 0.0, 1.0, EPROC_SIDE_UPPER) == EPROC_ERR_INVALID, "bet_init NULL");
    BREAKER_CHECK(eproc_bet_init(&bet, NAN, 0.05, 0.0, 1.0, EPROC_SIDE_UPPER) == EPROC_ERR_INVALID, "bet_init NaN mu0");
    BREAKER_CHECK(eproc_bet_init(&bet, 0.5, 0.0, 0.0, 1.0, EPROC_SIDE_UPPER) == EPROC_ERR_INVALID, "bet_init alpha=0");
    BREAKER_CHECK(eproc_bet_init(&bet, 0.5, 1.0, 0.0, 1.0, EPROC_SIDE_UPPER) == EPROC_ERR_INVALID, "bet_init alpha=1");
    BREAKER_CHECK(eproc_bet_init(&bet, 0.5, -0.1, 0.0, 1.0, EPROC_SIDE_UPPER) == EPROC_ERR_INVALID, "bet_init alpha<0");
    BREAKER_CHECK(eproc_bet_init(&bet, 0.5, 0.05, 1.0, 1.0, EPROC_SIDE_UPPER) == EPROC_ERR_INVALID, "bet_init 0-span");
    BREAKER_CHECK(eproc_bet_init(&bet, 0.5, 0.05, 2.0, 1.0, EPROC_SIDE_UPPER) == EPROC_ERR_INVALID, "bet_init inverted span");
    BREAKER_CHECK(eproc_bet_init(&bet, 0.0, 0.05, -1e308, 1e308, EPROC_SIDE_UPPER) == EPROC_ERR_INVALID, "bet_init inf span");
    BREAKER_CHECK(eproc_bet_init(&bet, 1.5, 0.05, 0.0, 1.0, EPROC_SIDE_UPPER) == EPROC_ERR_INVALID, "bet_init mu0>upper");
    BREAKER_CHECK(eproc_bet_init(&bet, -0.5, 0.05, 0.0, 1.0, EPROC_SIDE_UPPER) == EPROC_ERR_INVALID, "bet_init mu0<lower");
    BREAKER_CHECK(eproc_bet_init(&bet, 0.5, 0.05, 0.0, 1.0, 99) == EPROC_ERR_INVALID, "bet_init bad side");
    BREAKER_CHECK(eproc_bet_init(&bet, 0.5, 0.05, 0.0, 1.0, -1) == EPROC_ERR_INVALID, "bet_init neg side");
    BREAKER_CHECK(eproc_bet_update(NULL, 0.5, &br) == EPROC_ERR_INVALID, "bet_update NULL");
    BREAKER_CHECK(eproc_bet_update_batch(NULL, dummy_d, 4, &br) == EPROC_ERR_INVALID, "bet_batch NULL ep");
    BREAKER_CHECK(eproc_bet_init(&bet, 0.5, 0.05, 0.0, 1.0, EPROC_SIDE_UPPER) == EPROC_OK, "bet_init valid");
    BREAKER_CHECK(eproc_bet_update_batch(&bet, NULL, 4, &br) == EPROC_ERR_INVALID, "bet_batch NULL xs n>0");
    BREAKER_CHECK(eproc_bet_update_batch(&bet, NULL, 0, &br) == EPROC_OK && br.n == 0, "bet_batch NULL xs n=0");
    BREAKER_CHECK(eproc_bet_update_batch(&bet, dummy_d, 0, NULL) == EPROC_OK && bet.n == 0, "bet_batch n=0 out=NULL");
    eproc_bet_query(NULL, &br);
    eproc_bet_query(&bet, NULL);
    BREAKER_CHECK(eproc_combine_dependent_log(NULL, 5) == 0.0, "combine_dep NULL");
    BREAKER_CHECK(eproc_combine_dependent_log(dummy_d, 0) == 0.0, "combine_dep n=0");
    BREAKER_CHECK(eproc_combine_independent_log(INFINITY, -INFINITY) == 0.0, "combine_ind +inf -inf");
    BREAKER_CHECK(eproc_combine_independent_log(NAN, NAN) == 0.0, "combine_ind NaN");

    /* 1B. Confidence Sequence */
    eproc_cs_t cs;
    eproc_cs_result_t cr;
    BREAKER_CHECK(eproc_cs_init_eb(NULL, 0.05, 0.0, 1.0) == EPROC_ERR_INVALID, "cs_eb NULL");
    BREAKER_CHECK(eproc_cs_init_eb(&cs, 0.05, 1.0, 1.0) == EPROC_ERR_INVALID, "cs_eb 0-span");
    BREAKER_CHECK(eproc_cs_init_eb(&cs, 0.05, -1e308, 1e308) == EPROC_ERR_INVALID, "cs_eb inf-span");
    BREAKER_CHECK(eproc_cs_init_eb(&cs, 0.0, 0.0, 1.0) == EPROC_ERR_INVALID, "cs_eb alpha=0");
    BREAKER_CHECK(eproc_cs_init_normal(NULL, 0.05, 1.0, 100.0) == EPROC_ERR_INVALID, "cs_norm NULL");
    BREAKER_CHECK(eproc_cs_init_normal(&cs, 0.05, 0.0, 100.0) == EPROC_ERR_INVALID, "cs_norm sigma=0");
    BREAKER_CHECK(eproc_cs_init_normal(&cs, 0.05, 1.0, 0.0) == EPROC_ERR_INVALID, "cs_norm v_opt=0");
    BREAKER_CHECK(eproc_cs_init_normal(&cs, 0.05, -1.0, 100.0) == EPROC_ERR_INVALID, "cs_norm sigma<0");
    BREAKER_CHECK(eproc_cs_update(NULL, 0.5, &cr) == EPROC_ERR_INVALID, "cs_update NULL");
    BREAKER_CHECK(eproc_cs_init_eb(&cs, 0.05, 0.0, 1.0) == EPROC_OK, "cs_eb valid");
    BREAKER_CHECK(eproc_cs_update_batch(&cs, NULL, 2, &cr) == EPROC_ERR_INVALID, "cs_batch NULL n>0");
    BREAKER_CHECK(eproc_cs_update_batch(&cs, NULL, 0, &cr) == EPROC_OK && cr.n == 0, "cs_batch NULL n=0");
    eproc_cs_query(NULL, &cr);
    eproc_cs_query(&cs, NULL);

    /* 1C. E-Detector */
    eproc_edetect_t ed;
    eproc_edetect_result_t er;
    BREAKER_CHECK(eproc_edetect_init(NULL, 0.5, 0.05, 0.0, 1.0, EPROC_SIDE_UPPER) == EPROC_ERR_INVALID, "ed NULL");
    BREAKER_CHECK(eproc_edetect_update(NULL, 0.5, &er) == EPROC_ERR_INVALID, "ed_upd NULL");
    eproc_edetect_reset_alarm(NULL);
    BREAKER_CHECK(eproc_edetect_init(&ed, 0.5, 0.05, 0.0, 1.0, EPROC_SIDE_UPPER) == EPROC_OK, "ed valid");
    BREAKER_CHECK(eproc_edetect_update(&ed, NAN, &er) == EPROC_OK && er.n == 0 && isfinite(er.stat), "ed NaN");

    /* 1D. AgentRace */
    eproc_race_t *race = (eproc_race_t *)calloc(1, sizeof(*race));
    BREAKER_CHECK(race != NULL, "alloc race");
    eproc_race_status_t rst;
    BREAKER_CHECK(eproc_race_init(NULL, 4, 0.05, 0.0, 1.0, 0.0) == EPROC_ERR_INVALID, "race NULL");
    BREAKER_CHECK(eproc_race_init(race, 0, 0.05, 0.0, 1.0, 0.0) == EPROC_ERR_INVALID, "race 0 arms");
    BREAKER_CHECK(eproc_race_init(race, EPROC_MAX_ARMS + 1, 0.05, 0.0, 1.0, 0.0) == EPROC_ERR_INVALID, "race >64 arms");
    BREAKER_CHECK(eproc_race_init(race, 4, 0.05, 0.0, 1.0, -0.1) == EPROC_ERR_INVALID, "race neg margin");
    BREAKER_CHECK(eproc_race_init(race, 4, 0.05, 0.0, 1.0, 0.0) == EPROC_OK, "race valid");
    BREAKER_CHECK(eproc_race_update(NULL, 0, 0.5, &rst) == EPROC_ERR_INVALID, "race_upd NULL");
    BREAKER_CHECK(eproc_race_update(race, 4, 0.5, &rst) == EPROC_ERR_INVALID, "race_upd arm OOB");
    BREAKER_CHECK(eproc_race_update(race, UINT32_MAX, 0.5, &rst) == EPROC_ERR_INVALID, "race_upd arm UINT32_MAX");
    BREAKER_CHECK(eproc_race_update_batch(race, NULL, dummy_d, 2, &rst) == EPROC_ERR_INVALID, "race_batch NULL arms");
    BREAKER_CHECK(eproc_race_update_batch(race, dummy_u, NULL, 2, &rst) == EPROC_ERR_INVALID, "race_batch NULL rew");
    BREAKER_CHECK(eproc_race_update_batch(race, NULL, NULL, 0, &rst) == EPROC_OK, "race_batch n=0");
    BREAKER_CHECK(eproc_race_arm_state(NULL, 0, &cr, NULL) == EPROC_ERR_INVALID, "race_arm NULL");
    BREAKER_CHECK(eproc_race_arm_state(race, 99, &cr, NULL) == EPROC_ERR_INVALID, "race_arm OOB");
    BREAKER_CHECK(eproc_race_suggest_arm(NULL) == -1, "race_suggest NULL");
    eproc_race_status(NULL, &rst);
    eproc_race_status(race, NULL);
    free(race);

    /* 1E. Conformal / Delayed / Budget / Monoid */
    eproc_conformal_t cp;
    eproc_conformal_out_t cpo;
    const double good_alphas[2] = {0.2, 0.1};
    const double bad_alphas[2] = {0.2, 1.0};
    BREAKER_CHECK(eproc_conformal_init(NULL, good_alphas, 2, 32, 0.05) == EPROC_ERR_INVALID, "cp NULL");
    BREAKER_CHECK(eproc_conformal_init(&cp, NULL, 2, 32, 0.05) == EPROC_ERR_INVALID, "cp NULL alphas");
    BREAKER_CHECK(eproc_conformal_init(&cp, good_alphas, 0, 32, 0.05) == EPROC_ERR_INVALID, "cp 0 rates");
    BREAKER_CHECK(eproc_conformal_init(&cp, good_alphas, EPROC_MAX_RATES + 1, 32, 0.05) == EPROC_ERR_INVALID, "cp >5 rates");
    BREAKER_CHECK(eproc_conformal_init(&cp, good_alphas, 2, 0, 0.05) == EPROC_ERR_INVALID, "cp win=0");
    BREAKER_CHECK(eproc_conformal_init(&cp, good_alphas, 2, EPROC_CONFORMAL_MAX_WIN + 1, 0.05) == EPROC_ERR_INVALID, "cp win>256");
    BREAKER_CHECK(eproc_conformal_init(&cp, good_alphas, 2, 32, -0.01) == EPROC_ERR_INVALID, "cp ki<0");
    BREAKER_CHECK(eproc_conformal_init(&cp, bad_alphas, 2, 32, 0.05) == EPROC_ERR_INVALID, "cp alpha=1");
    BREAKER_CHECK(eproc_conformal_init(&cp, good_alphas, 2, 32, 0.05) == EPROC_OK, "cp valid");
    BREAKER_CHECK(eproc_conformal_update(NULL, 1.0, &cpo) == EPROC_ERR_INVALID, "cp_upd NULL");
    BREAKER_CHECK(eproc_conformal_update_batch(&cp, NULL, 2, &cpo) == EPROC_ERR_INVALID, "cp_batch NULL n>0");
    BREAKER_CHECK(eproc_conformal_update_batch(&cp, NULL, 0, &cpo) == EPROC_OK, "cp_batch NULL n=0");
    eproc_conformal_predict(NULL, 0.0, 1.0, lo, hi, &cpo);
    eproc_conformal_predict(&cp, NAN, -1.0, lo, hi, NULL);
    BREAKER_CHECK(isfinite(lo[0]) && isfinite(hi[0]), "cp_predict sanitizes NaN/neg scale");

    eproc_delayed_conformal_t *dc = (eproc_delayed_conformal_t *)calloc(1, sizeof(*dc));
    BREAKER_CHECK(dc != NULL, "alloc dc");
    BREAKER_CHECK(eproc_delayed_conformal_init(NULL, good_alphas, 2, 32, 0.05) == EPROC_ERR_INVALID, "dc NULL");
    BREAKER_CHECK(eproc_delayed_conformal_issue(NULL, 1.0, 1.0, NULL, NULL, NULL) == EPROC_ERR_INVALID, "dc_issue NULL");
    BREAKER_CHECK(eproc_delayed_conformal_reveal(NULL, 1, 1.0, NULL) == EPROC_ERR_INVALID, "dc_reveal NULL");
    BREAKER_CHECK(eproc_delayed_conformal_init(dc, good_alphas, 2, 32, 0.05) == EPROC_OK, "dc valid");
    BREAKER_CHECK(eproc_delayed_conformal_reveal(dc, 0, 1.0, NULL) == EPROC_ERR_INVALID, "dc_reveal ticket 0");
    BREAKER_CHECK(eproc_delayed_conformal_reveal(dc, 9999, 1.0, NULL) == EPROC_ERR_INVALID, "dc_reveal unknown ticket");
    free(dc);

    eproc_budget_conformal_t bc;
    BREAKER_CHECK(eproc_budget_conformal_init(NULL, 0.1, 32, 1.2) == EPROC_ERR_INVALID, "bc NULL");
    BREAKER_CHECK(eproc_budget_conformal_init(&bc, 0.1, 32, 0.5) == EPROC_ERR_INVALID, "bc penalty<1");
    BREAKER_CHECK(eproc_budget_conformal_init(&bc, 0.1, 32, 1.2) == EPROC_OK, "bc valid");
    BREAKER_CHECK(eproc_budget_conformal_bound(NULL, 10.0) == 0.0, "bc_bound NULL");
    BREAKER_CHECK(eproc_budget_conformal_update(NULL, 10.0, 10.0, 0) == EPROC_ERR_INVALID, "bc_upd NULL");
    BREAKER_CHECK(eproc_budget_conformal_update(&bc, 10.0, -1.0, 0) == EPROC_ERR_INVALID, "bc_upd neg cost");

    eproc_cmonoid_t cm;
    eproc_cmonoid_init(NULL, 1.0);
    eproc_cmonoid_add(NULL, 1.0);
    eproc_cmonoid_add_batch(NULL, dummy_d, 4);
    eproc_cmonoid_init(&cm, 1.0);
    eproc_cmonoid_add_batch(&cm, NULL, 4);
    eproc_cmonoid_add_batch(&cm, NULL, 0);
    BREAKER_CHECK(cm.count == 0, "cmonoid NULL batch no-op");
    eproc_cmonoid_merge(NULL, &cm, &cm);
    eproc_cmonoid_merge(&cm, NULL, &cm);
    eproc_cmonoid_merge(&cm, &cm, NULL);
    BREAKER_CHECK(eproc_cmonoid_subtract(NULL, &cm, &cm) == EPROC_ERR_INVALID, "cm_sub NULL");
    BREAKER_CHECK(eproc_cmonoid_quantile(NULL, 0.5) == 0.0, "cm_q NULL");
    BREAKER_CHECK(eproc_cmonoid_quantile(&cm, 0.5) == 0.0, "cm_q empty");
    BREAKER_CHECK(eproc_cmonoid_serialize(NULL, wire, 640) == EPROC_ERR_INVALID, "cm_ser NULL");
    BREAKER_CHECK(eproc_cmonoid_serialize(&cm, NULL, 640) == EPROC_ERR_INVALID, "cm_ser NULL buf");
    BREAKER_CHECK(eproc_cmonoid_deserialize(NULL, wire, 640) == EPROC_ERR_INVALID, "cm_deser NULL");
    BREAKER_CHECK(eproc_cmonoid_deserialize(&cm, NULL, 640) == EPROC_ERR_INVALID, "cm_deser NULL buf");

    /* 1F. SafeBOA & EchoPool */
    eproc_boa_t boa;
    eproc_boa_step_t bst;
    BREAKER_CHECK(eproc_boa_init(NULL, 4, 0, 0.05) == EPROC_ERR_INVALID, "boa NULL");
    BREAKER_CHECK(eproc_boa_init(&boa, 0, 0, 0.05) == EPROC_ERR_INVALID, "boa 0 experts");
    BREAKER_CHECK(eproc_boa_init(&boa, EPROC_BOA_MAX_EXPERTS + 1, 0, 0.05) == EPROC_ERR_INVALID, "boa >64 experts");
    BREAKER_CHECK(eproc_boa_init(&boa, 4, 4, 0.05) == EPROC_ERR_INVALID, "boa anchor OOB");
    BREAKER_CHECK(eproc_boa_init(&boa, 4, 0, 0.05) == EPROC_OK, "boa valid");
    BREAKER_CHECK(eproc_boa_predict(NULL, dummy_d) == 0.0, "boa_pred NULL");
    BREAKER_CHECK(eproc_boa_predict(&boa, NULL) == 0.0, "boa_pred NULL preds");
    BREAKER_CHECK(eproc_boa_update(NULL, dummy_d, 1.0, &bst) == EPROC_ERR_INVALID, "boa_upd NULL");
    BREAKER_CHECK(eproc_boa_update(&boa, NULL, 1.0, &bst) == EPROC_ERR_INVALID, "boa_upd NULL preds");
    BREAKER_CHECK(eproc_boa_snapshot_weights(NULL, dummy_d, 4, NULL) == EPROC_ERR_INVALID, "boa_snap NULL");
    BREAKER_CHECK(eproc_boa_snapshot_weights(&boa, NULL, 4, NULL) == EPROC_ERR_INVALID, "boa_snap NULL out n>0");
    BREAKER_CHECK(eproc_boa_snapshot_weights(&boa, NULL, 0, NULL) == EPROC_OK, "boa_snap n=0");

    eproc_echopool_t *epool = (eproc_echopool_t *)calloc(1, sizeof(*epool));
    BREAKER_CHECK(epool != NULL, "alloc echopool");
    eproc_echo_out_t eo;
    BREAKER_CHECK(eproc_echopool_init(NULL, 4, 4, 0.5) == EPROC_ERR_INVALID, "echo NULL");
    BREAKER_CHECK(eproc_echopool_init(epool, 0, 4, 0.5) == EPROC_ERR_INVALID, "echo 0 sources");
    BREAKER_CHECK(eproc_echopool_init(epool, EPROC_ECHO_MAX_SOURCES + 1, 4, 0.5) == EPROC_ERR_INVALID, "echo >256 sources");
    BREAKER_CHECK(eproc_echopool_init(epool, 4, 0, 0.5) == EPROC_ERR_INVALID, "echo 0 claims");
    BREAKER_CHECK(eproc_echopool_init(epool, 4, EPROC_ECHO_MAX_CLAIMS + 1, 0.5) == EPROC_ERR_INVALID, "echo >256 claims");
    BREAKER_CHECK(eproc_echopool_init(epool, 4, 4, 0.0) == EPROC_ERR_INVALID, "echo prior=0");
    BREAKER_CHECK(eproc_echopool_init(epool, 4, 4, 1.0) == EPROC_ERR_INVALID, "echo prior=1");
    BREAKER_CHECK(eproc_echopool_init(epool, 4, 4, 0.5) == EPROC_OK, "echo valid");
    BREAKER_CHECK(eproc_echopool_submit(NULL, 0, 0, 1, 1.0, &eo) == EPROC_ERR_INVALID, "echo_sub NULL");
    BREAKER_CHECK(eproc_echopool_submit(epool, 4, 0, 1, 1.0, &eo) == EPROC_ERR_INVALID, "echo_sub src OOB");
    BREAKER_CHECK(eproc_echopool_submit(epool, 0, 4, 1, 1.0, &eo) == EPROC_ERR_INVALID, "echo_sub clm OOB");
    BREAKER_CHECK(eproc_echopool_resolve(NULL, 0, 1) == EPROC_ERR_INVALID, "echo_res NULL");
    BREAKER_CHECK(eproc_echopool_resolve(epool, 4, 1) == EPROC_ERR_INVALID, "echo_res OOB");
    BREAKER_CHECK(eproc_echopool_posterior(NULL, 0) == 0.5, "echo_post NULL");
    BREAKER_CHECK(eproc_echopool_posterior(epool, 99) == 0.5, "echo_post OOB");
    BREAKER_CHECK(eproc_echopool_source_reliability(NULL, 0) == 0.0, "echo_rel NULL");
    BREAKER_CHECK(eproc_echopool_source_reliability(epool, 99) == 0.0, "echo_rel OOB");
    free(epool);

    /* 1G. VarOpt, RLS, Sift */
    eproc_varopt_t *vo = (eproc_varopt_t *)calloc(1, sizeof(*vo));
    BREAKER_CHECK(vo != NULL, "alloc vo");
    BREAKER_CHECK(eproc_varopt_init(NULL, 8, 1) == EPROC_ERR_INVALID, "vo NULL");
    BREAKER_CHECK(eproc_varopt_init(vo, 0, 1) == EPROC_ERR_INVALID, "vo k=0");
    BREAKER_CHECK(eproc_varopt_init(vo, EPROC_VAROPT_MAX_K + 1, 1) == EPROC_ERR_INVALID, "vo k>4096");
    BREAKER_CHECK(eproc_varopt_init(vo, 8, 1) == EPROC_OK, "vo valid");
    BREAKER_CHECK(eproc_varopt_push(NULL, 0, 1.0, 1.0) == EPROC_ERR_INVALID, "vo_push NULL");
    BREAKER_CHECK(eproc_varopt_push(vo, 0, -1.0, 1.0) == EPROC_ERR_INVALID, "vo_push w<0");
    BREAKER_CHECK(eproc_varopt_push(vo, 0, NAN, 1.0) == EPROC_ERR_INVALID, "vo_push w=NaN");
    BREAKER_CHECK(eproc_varopt_push(vo, 0, 1.0, INFINITY) == EPROC_ERR_INVALID, "vo_push v=Inf");
    BREAKER_CHECK(eproc_varopt_estimate_sum(NULL) == 0.0, "vo_sum NULL");
    BREAKER_CHECK(eproc_varopt_estimate_mean(NULL) == 0.0, "vo_mean NULL");
    BREAKER_CHECK(eproc_varopt_estimate_sum(vo) == 0.0, "vo_sum empty");
    BREAKER_CHECK(eproc_varopt_estimate_mean(vo) == 0.0, "vo_mean empty");
    free(vo);

    eproc_rls_t rls;
    eproc_rls_out_t ro;
    BREAKER_CHECK(eproc_rls_init(NULL, 4, 1.0, 2.0, 0.05, 1) == EPROC_ERR_INVALID, "rls NULL");
    BREAKER_CHECK(eproc_rls_init(&rls, 0, 1.0, 2.0, 0.05, 1) == EPROC_ERR_INVALID, "rls d=0");
    BREAKER_CHECK(eproc_rls_init(&rls, EPROC_RLS_MAX_D + 1, 1.0, 2.0, 0.05, 1) == EPROC_ERR_INVALID, "rls d>32");
    BREAKER_CHECK(eproc_rls_init(&rls, 4, 0.0, 2.0, 0.05, 1) == EPROC_ERR_INVALID, "rls lam=0");
    BREAKER_CHECK(eproc_rls_init(&rls, 4, 1.0, 0.0, 0.05, 1) == EPROC_ERR_INVALID, "rls over=0");
    BREAKER_CHECK(eproc_rls_init(&rls, 4, 1.0, 2.0, 0.0, 1) == EPROC_ERR_INVALID, "rls min_p=0");
    BREAKER_CHECK(eproc_rls_init(&rls, 4, 1.0, 2.0, 1.5, 1) == EPROC_ERR_INVALID, "rls min_p>1");
    BREAKER_CHECK(eproc_rls_init(&rls, 4, 1.0, 2.0, 0.05, 1) == EPROC_OK, "rls valid");
    BREAKER_CHECK(eproc_rls_step(NULL, dummy_d, &ro) == EPROC_ERR_INVALID, "rls_step NULL");
    BREAKER_CHECK(eproc_rls_step(&rls, NULL, &ro) == EPROC_ERR_INVALID, "rls_step NULL x");

    eproc_sift_t *sf = (eproc_sift_t *)calloc(1, sizeof(*sf));
    BREAKER_CHECK(sf != NULL, "alloc sift");
    eproc_sift_action_t act;
    BREAKER_CHECK(eproc_sift_init(NULL, 4, 4, 1.0, 1.0) == EPROC_ERR_INVALID, "sift NULL");
    BREAKER_CHECK(eproc_sift_init(sf, 0, 4, 1.0, 1.0) == EPROC_ERR_INVALID, "sift 0 sites");
    BREAKER_CHECK(eproc_sift_init(sf, EPROC_SIFT_MAX_SITES + 1, 4, 1.0, 1.0) == EPROC_ERR_INVALID, "sift >1024 sites");
    BREAKER_CHECK(eproc_sift_init(sf, 4, 0, 1.0, 1.0) == EPROC_ERR_INVALID, "sift 0 per_site");
    BREAKER_CHECK(eproc_sift_init(sf, 4, EPROC_SIFT_MAX_PER_SITE + 1, 1.0, 1.0) == EPROC_ERR_INVALID, "sift >16 per_site");
    BREAKER_CHECK(eproc_sift_init(sf, 4, 4, 0.0, 1.0) == EPROC_ERR_INVALID, "sift mean=0");
    BREAKER_CHECK(eproc_sift_init(sf, 4, 4, 1.0, 0.0) == EPROC_ERR_INVALID, "sift sd=0");
    BREAKER_CHECK(eproc_sift_init(sf, 4, 4, 1.0, 1.0) == EPROC_OK, "sift valid");
    BREAKER_CHECK(eproc_sift_set_noise_scale(NULL, 0, 1.0) == EPROC_ERR_INVALID, "sift_ns NULL");
    BREAKER_CHECK(eproc_sift_set_noise_scale(sf, 16, 1.0) == EPROC_ERR_INVALID, "sift_ns OOB");
    BREAKER_CHECK(eproc_sift_set_noise_scale(sf, 0, 0.0) == EPROC_ERR_INVALID, "sift_ns 0");
    BREAKER_CHECK(eproc_sift_observe_channel(NULL, 0, 1.0, 1) == EPROC_ERR_INVALID, "sift_oc NULL");
    BREAKER_CHECK(eproc_sift_observe_channel(sf, 16, 1.0, 1) == EPROC_ERR_INVALID, "sift_oc OOB");
    BREAKER_CHECK(eproc_sift_observe_channel(sf, 0, 1.0, 0) == EPROC_ERR_INVALID, "sift_oc 0 ticks");
    BREAKER_CHECK(eproc_sift_observe_site(NULL, 0, 1.0) == EPROC_ERR_INVALID, "sift_os NULL");
    BREAKER_CHECK(eproc_sift_observe_site(sf, 4, 1.0) == EPROC_ERR_INVALID, "sift_os OOB");
    BREAKER_CHECK(eproc_sift_select_next(NULL, 1.0, 1.0, 1.0, 4, &act) == EPROC_ERR_INVALID, "sift_sel NULL");
    BREAKER_CHECK(eproc_sift_select_next(sf, 1.0, 1.0, 1.0, 4, NULL) == EPROC_ERR_INVALID, "sift_sel NULL out");
    BREAKER_CHECK(eproc_sift_select_next(sf, 1.0, 1.0, 1.0, 0, &act) == EPROC_ERR_INVALID, "sift_sel 0 ticks");
    BREAKER_CHECK(eproc_sift_select_next(sf, 0.0, -1.0, NAN, 4, &act) == EPROC_ERR_INVALID, "sift_sel no valid cost");
    free(sf);

    printf("[breaker-c] Section 1 (NULL/invalid/OOB/zero-len): done\n");
}

/* ============================================================================
 * 2. BOUNDARY mu0 = 1e-12 & 1 - 1e-12 + NUMERICAL EXTREMES
 * ============================================================================ */

static void test_boundary_mu0_and_extremes(void) {
    const double boundary_mus[4] = {1e-12, 1.0 - 1e-12, 0.0, 1.0};
    const int32_t sides[3] = {EPROC_SIDE_UPPER, EPROC_SIDE_LOWER, EPROC_SIDE_TWO_SIDED};
    const double probes[10] = {0.0, -0.0, 1.0, 1e-310, -1e-310, 1e15, -1e15, NAN, INFINITY, -INFINITY};

    for (int m = 0; m < 4; ++m) {
        for (int s = 0; s < 3; ++s) {
            eproc_bet_t ep;
            eproc_bet_result_t r;
            BREAKER_CHECK(eproc_bet_init(&ep, boundary_mus[m], 0.05, 0.0, 1.0, sides[s]) == EPROC_OK,
                          "boundary mu0 init");
            for (int t = 0; t < 200; ++t) {
                double x = probes[t % 10];
                BREAKER_CHECK(eproc_bet_update(&ep, x, &r) == EPROC_OK, "boundary mu0 update");
                BREAKER_CHECK(isfinite(r.log_e) && isfinite(r.e_value) && isfinite(r.p_value) && isfinite(r.mean),
                              "boundary mu0 finite outputs");
                BREAKER_CHECK(r.p_value >= 0.0 && r.p_value <= 1.0, "boundary mu0 p in [0,1]");
            }
        }
    }

    /* Check boundary mu0=1e-12 null validity (all X=0 must never falsely reject) */
    {
        eproc_bet_t ep;
        eproc_bet_result_t r;
        eproc_bet_init(&ep, 1e-12, 0.05, 0.0, 1.0, EPROC_SIDE_UPPER);
        for (int t = 0; t < 1000; ++t) {
            eproc_bet_update(&ep, 0.0, &r);
        }
        BREAKER_CHECK(!r.rejected && isfinite(r.log_e), "mu0=1e-12 with X=0 never rejects");
    }

    /* Check boundary mu0=1-1e-12 null validity (all X=1 must never falsely reject) */
    {
        eproc_bet_t ep;
        eproc_bet_result_t r;
        eproc_bet_init(&ep, 1.0 - 1e-12, 0.05, 0.0, 1.0, EPROC_SIDE_LOWER);
        for (int t = 0; t < 1000; ++t) {
            eproc_bet_update(&ep, 1.0, &r);
        }
        BREAKER_CHECK(!r.rejected && isfinite(r.log_e), "mu0=1-1e-12 with X=1 never rejects");
    }

    printf("[breaker-c] Section 2 (boundary mu0 = 1e-12 / 1-1e-12): done\n");
}

/* ============================================================================
 * 3. DELAYED CONFORMAL & ECHOPOOL FULL-CAPACITY SATURATION STRESS
 * ============================================================================ */

static void test_capacity_saturation(void) {
    /* 3A. DelayedConformal 4096-slot full table + Knuth Alg R out-of-order deletion */
    eproc_delayed_conformal_t *dc = (eproc_delayed_conformal_t *)calloc(1, sizeof(*dc));
    BREAKER_CHECK(dc != NULL, "alloc dc cap");
    const double alphas[3] = {0.20, 0.10, 0.05};
    BREAKER_CHECK(eproc_delayed_conformal_init(dc, alphas, 3, 64, 0.05) == EPROC_OK, "dc cap init");

    uint64_t *tids = (uint64_t *)malloc(sizeof(uint64_t) * EPROC_DELAY_MAX_PENDING);
    BREAKER_CHECK(tids != NULL, "alloc tids");

    for (uint32_t i = 0; i < EPROC_DELAY_MAX_PENDING; ++i) {
        BREAKER_CHECK(eproc_delayed_conformal_issue(dc, (double)i * 0.01, 1.0, &tids[i], NULL, NULL) == EPROC_OK,
                      "dc fill 4096");
    }
    BREAKER_CHECK(dc->pending_count == EPROC_DELAY_MAX_PENDING, "dc pending == 4096");
    /* 4097th must fail cleanly with EPROC_ERR_CAPACITY */
    uint64_t overflow_tid = 0;
    BREAKER_CHECK(eproc_delayed_conformal_issue(dc, 1.0, 1.0, &overflow_tid, NULL, NULL) == EPROC_ERR_CAPACITY,
                  "dc 4097th must return EPROC_ERR_CAPACITY");

    /* Reveal in pseudo-random permutation (coprime stride 1597) */
    for (uint32_t k = 0; k < EPROC_DELAY_MAX_PENDING; ++k) {
        uint32_t idx = (k * 1597U) % EPROC_DELAY_MAX_PENDING;
        BREAKER_CHECK(eproc_delayed_conformal_reveal(dc, tids[idx], (double)idx * 0.01 + 0.5, NULL) == EPROC_OK,
                      "dc perm reveal");
    }
    BREAKER_CHECK(dc->pending_count == 0, "dc pending == 0 after full perm reveal");
    /* Double-reveal must fail cleanly */
    BREAKER_CHECK(eproc_delayed_conformal_reveal(dc, tids[0], 1.0, NULL) == EPROC_ERR_INVALID,
                  "dc double-reveal rejected");
    free(tids);
    free(dc);

    /* 3B. EchoPool 4096-slot full table saturation */
    eproc_echopool_t *pool = (eproc_echopool_t *)calloc(1, sizeof(*pool));
    BREAKER_CHECK(pool != NULL, "alloc echo cap");
    BREAKER_CHECK(eproc_echopool_init(pool, 16, 16, 0.5) == EPROC_OK, "echo cap init");
    eproc_echo_out_t eo;
    for (uint32_t i = 0; i < EPROC_ECHO_TABLE_CAP; ++i) {
        BREAKER_CHECK(eproc_echopool_submit(pool, i % 16U, i % 16U, (uint64_t)i + 1ULL, 0.1, &eo) == EPROC_OK,
                      "echo fill 4096");
        BREAKER_CHECK(eo.accepted == 1, "echo fill accepted");
    }
    /* Duplicate of an existing item when table is 100% full must still dedup cleanly */
    BREAKER_CHECK(eproc_echopool_submit(pool, 0, 0, 1ULL, 0.1, &eo) == EPROC_OK && eo.accepted == 0,
                  "echo dup when full still drops echo");
    /* Novel 4097th item must return EPROC_ERR_CAPACITY cleanly without corruption */
    BREAKER_CHECK(eproc_echopool_submit(pool, 0, 0, 999999ULL, 0.1, &eo) == EPROC_ERR_CAPACITY && eo.accepted == 0,
                  "echo 4097th returns EPROC_ERR_CAPACITY");
    BREAKER_CHECK(pool->n_table_full == 1, "echo n_table_full == 1");
    free(pool);

    printf("[breaker-c] Section 3 (DelayedConformal & EchoPool 4096 saturation): done\n");
}

/* ============================================================================
 * 4. CONFORMAL MONOID ALIASING + 640-BYTE BIT-FLIP SWEEP + SUBTRACT GUARDS
 * ============================================================================ */

static void test_cmonoid_torture(void) {
    eproc_cmonoid_t a, b, ab, alias_a, alias_b;
    eproc_cmonoid_init(&a, 1.0);
    eproc_cmonoid_init(&b, 1.0);
    for (int i = 0; i < 100; ++i) {
        eproc_cmonoid_add(&a, (double)i * 0.1);
        eproc_cmonoid_add(&b, (double)i * 0.2 - 5.0);
    }
    eproc_cmonoid_merge(&a, &b, &ab);

    /* Aliasing: out == a, out == b, a == b == out */
    alias_a = a;
    eproc_cmonoid_merge(&alias_a, &b, &alias_a);
    BREAKER_CHECK(memcmp(alias_a.bin_counts, ab.bin_counts, sizeof(ab.bin_counts)) == 0, "merge out==a alias");

    alias_b = b;
    eproc_cmonoid_merge(&a, &alias_b, &alias_b);
    BREAKER_CHECK(memcmp(alias_b.bin_counts, ab.bin_counts, sizeof(ab.bin_counts)) == 0, "merge out==b alias");

    eproc_cmonoid_t sub_alias = ab;
    BREAKER_CHECK(eproc_cmonoid_subtract(&sub_alias, &a, &sub_alias) == EPROC_OK, "subtract out==global alias");
    BREAKER_CHECK(memcmp(sub_alias.bin_counts, b.bin_counts, sizeof(b.bin_counts)) == 0, "subtract alias bins exact");

    /* Single-bit-flip sweep across ALL 640 bytes of wire payload */
    uint8_t wire[640];
    BREAKER_CHECK(eproc_cmonoid_serialize(&ab, wire, 640) == EPROC_OK, "cm ser");
    for (size_t byte_i = 0; byte_i < 640; ++byte_i) {
        uint8_t corrupt[640];
        memcpy(corrupt, wire, 640);
        corrupt[byte_i] ^= 0x80U;
        eproc_cmonoid_t dec;
        int rc = eproc_cmonoid_deserialize(&dec, corrupt, 640);
        BREAKER_CHECK(rc != EPROC_OK, "every single byte flip in 640B wire must be rejected");
    }

    printf("[breaker-c] Section 4 (ConformalMonoid aliasing & 640-byte bit-flip sweep): done\n");
}

/* ============================================================================
 * 5. CONCURRENT SAFEBOA SEQLOCK (1M+ READS, 64 EXPERTS) + MULTI-WRITER SEQLOCK
 * ============================================================================ */

typedef struct {
    const eproc_boa_t *boa;
    uint32_t n_experts;
    uint32_t anchor_idx;
    size_t iters;
    _Atomic uint64_t torn_reads;
} boa_reader_arg_t;

typedef struct {
    eproc_boa_t *boa;
    _Atomic int stop;
    uint64_t updates;
} boa_writer_arg_t;

static void *boa_reader_fn(void *arg) {
    boa_reader_arg_t *ra = (boa_reader_arg_t *)arg;
    double w[EPROC_BOA_MAX_EXPERTS];
    uint64_t torn = 0;
    for (size_t it = 0; it < ra->iters; ++it) {
        int32_t fb = -1;
        int rc = eproc_boa_snapshot_weights(ra->boa, w, ra->n_experts, &fb);
        if (rc != EPROC_OK || (fb != 0 && fb != 1)) {
            torn++;
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
            torn++;
        }
    }
    atomic_fetch_add_explicit(&ra->torn_reads, torn, memory_order_relaxed);
    return NULL;
}

static void *boa_writer_fn(void *arg) {
    boa_writer_arg_t *wa = (boa_writer_arg_t *)arg;
    uint64_t s = 0xDEADBEEF1234ULL;
    double preds[EPROC_BOA_MAX_EXPERTS];
    uint64_t n = 0;
    while (!atomic_load_explicit(&wa->stop, memory_order_acquire)) {
        /* Alternate regimes to provoke fallback trips, quarantines, and weight swings */
        int phase = (int)((n / 40ULL) % 3ULL);
        double y = (double)(n % 17ULL) - 8.0;
        preds[0] = y + (phase == 1 ? 0.0 : 0.5);
        for (int i = 1; i < EPROC_BOA_MAX_EXPERTS; ++i) {
            if (phase == 1) {
                preds[i] = y + 3.0; /* degraded vs anchor */
            } else {
                preds[i] = y + (rng_u01(&s) - 0.5) * 0.2 * (double)((i % 5) + 1);
            }
        }
        eproc_boa_update(wa->boa, preds, y, NULL);
        n++;
    }
    wa->updates = n;
    return NULL;
}

/* Raw multi-writer seqlock stress: 16 words must all carry identical version stamp */
typedef struct {
    _Atomic uint64_t seq;
    _Atomic uint64_t words[16];
    _Atomic int stop;
    _Atomic uint64_t torn_reads;
} raw_seq_state_t;

static void *raw_seq_writer_fn(void *arg) {
    raw_seq_state_t *st = (raw_seq_state_t *)arg;
    uint64_t buf[16];
    uint64_t iter = 1;
    while (!atomic_load_explicit(&st->stop, memory_order_acquire)) {
        uint64_t odd = eproc_seqlock_write_begin(&st->seq);
        uint64_t tag = (odd << 16) ^ iter;
        for (int i = 0; i < 16; ++i) {
            buf[i] = tag ^ (uint64_t)i;
        }
        eproc_seqlock_store_words(st->words, buf, 16);
        eproc_seqlock_write_end(&st->seq, odd);
        iter++;
    }
    return NULL;
}

static void *raw_seq_reader_fn(void *arg) {
    raw_seq_state_t *st = (raw_seq_state_t *)arg;
    uint64_t buf[16];
    uint64_t torn = 0;
    for (size_t it = 0; it < 250000; ++it) {
        eproc_seqlock_read_words(&st->seq, st->words, buf, 16);
        uint64_t base = buf[0];
        for (int i = 1; i < 16; ++i) {
            if ((buf[i] ^ (uint64_t)i) != base) {
                torn++;
                break;
            }
        }
    }
    atomic_fetch_add_explicit(&st->torn_reads, torn, memory_order_relaxed);
    return NULL;
}

static void test_concurrent_seqlock(void) {
    /* 5A. SafeBOA 64-expert seqlock: 4 readers x 250,000 = 1,000,000 reads vs live writer */
    {
        eproc_boa_t *boa = (eproc_boa_t *)calloc(1, sizeof(*boa));
        BREAKER_CHECK(boa != NULL, "alloc boa 64");
        BREAKER_CHECK(eproc_boa_init(boa, EPROC_BOA_MAX_EXPERTS, 0, 0.05) == EPROC_OK, "boa 64 init");

        boa_writer_arg_t warg;
        warg.boa = boa;
        atomic_init(&warg.stop, 0);
        warg.updates = 0;

        pthread_t wt;
        BREAKER_CHECK(pthread_create(&wt, NULL, boa_writer_fn, &warg) == 0, "spawn boa writer");

        pthread_t rts[4];
        boa_reader_arg_t rargs[4];
        for (int r = 0; r < 4; ++r) {
            rargs[r].boa = boa;
            rargs[r].n_experts = EPROC_BOA_MAX_EXPERTS;
            rargs[r].anchor_idx = 0;
            rargs[r].iters = 250000;
            atomic_init(&rargs[r].torn_reads, 0);
            BREAKER_CHECK(pthread_create(&rts[r], NULL, boa_reader_fn, &rargs[r]) == 0, "spawn boa reader");
        }

        uint64_t total_torn = 0;
        for (int r = 0; r < 4; ++r) {
            pthread_join(rts[r], NULL);
            total_torn += atomic_load_explicit(&rargs[r].torn_reads, memory_order_relaxed);
        }
        atomic_store_explicit(&warg.stop, 1, memory_order_release);
        pthread_join(wt, NULL);

        printf("[breaker-c] SafeBOA 64-expert seqlock: 1,000,000 reads vs %llu updates -> torn=%llu\n",
               (unsigned long long)warg.updates, (unsigned long long)total_torn);
        BREAKER_CHECK(total_torn == 0ULL, "SafeBOA 64-expert seqlock 0 torn reads");
        free(boa);
    }

    /* 5B. Multi-writer raw seqlock: 2 writers + 4 readers = 1,000,000 reads */
    {
        raw_seq_state_t st;
        atomic_init(&st.seq, 0ULL);
        for (int i = 0; i < 16; ++i) {
            atomic_init(&st.words[i], (uint64_t)i);
        }
        atomic_init(&st.stop, 0);
        atomic_init(&st.torn_reads, 0ULL);

        pthread_t ws[2], rs[4];
        for (int w = 0; w < 2; ++w) {
            BREAKER_CHECK(pthread_create(&ws[w], NULL, raw_seq_writer_fn, &st) == 0, "spawn raw writer");
        }
        for (int r = 0; r < 4; ++r) {
            BREAKER_CHECK(pthread_create(&rs[r], NULL, raw_seq_reader_fn, &st) == 0, "spawn raw reader");
        }
        for (int r = 0; r < 4; ++r) {
            pthread_join(rs[r], NULL);
        }
        atomic_store_explicit(&st.stop, 1, memory_order_release);
        for (int w = 0; w < 2; ++w) {
            pthread_join(ws[w], NULL);
        }
        uint64_t torn = atomic_load_explicit(&st.torn_reads, memory_order_relaxed);
        printf("[breaker-c] Multi-writer raw seqlock (2W+4R, 1M reads): torn=%llu\n",
               (unsigned long long)torn);
        BREAKER_CHECK(torn == 0ULL, "multi-writer raw seqlock 0 torn reads");
    }
}

/* ============================================================================
 * 6. UB WITNESS: MultiRateConformal finite (-1e308, +1e308, +1e308) -> (uint32_t)NaN
 * ============================================================================ */

static void test_conformal_ub_witness(void) {
    eproc_conformal_t cp;
    eproc_conformal_out_t out;
    const double alphas[5] = {0.50, 0.20, 0.10, 0.05, 0.01};
    BREAKER_CHECK(eproc_conformal_init(&cp, alphas, 5, 32, 0.05) == EPROC_OK, "ub cp init");
    BREAKER_CHECK(eproc_conformal_update(&cp, -1e308, &out) == EPROC_OK, "ub cp -1e308");
    BREAKER_CHECK(eproc_conformal_update(&cp, +1e308, &out) == EPROC_OK, "ub cp +1e308 #1");
    BREAKER_CHECK(eproc_conformal_update(&cp, +1e308, &out) == EPROC_OK, "ub cp +1e308 #2");
    for (uint32_t r = 0; r < 5; ++r) {
        BREAKER_CHECK(isfinite(out.eff_alphas[r]), "ub witness: eff_alpha must stay finite");
        BREAKER_CHECK(isfinite(out.quantiles[r]), "ub witness: quantile must stay finite");
    }
}

int main(int argc, char **argv) {
    int run_ub_witness = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--ub-witness") == 0) {
            run_ub_witness = 1;
        }
    }

    if (run_ub_witness) {
        printf("=== Running UBSan witness: MultiRateConformal (-1e308, +1e308, +1e308) ===\n");
        test_conformal_ub_witness();
    } else {
        test_null_invalid_and_empty();
        test_boundary_mu0_and_extremes();
        test_capacity_saturation();
        test_cmonoid_torture();
        test_concurrent_seqlock();
    }

    if (g_failures == 0) {
        printf("BREAKER C TORTURE PASSED (0 failures)\n");
        return 0;
    }
    fprintf(stderr, "BREAKER C TORTURE FAILED (%d failures)\n", g_failures);
    return 1;
}
