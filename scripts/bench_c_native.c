/*
 * Native C11 bare-metal microbenchmark for eproc.
 * Measures median latency (ns/op) and throughput (M ops/s) across 5 paired
 * runs with zero Python FFI overhead.
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../csrc/eproc.h"

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median5(double v[5]) {
    qsort(v, 5, sizeof(double), cmp_double);
    return v[2];
}

static uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double rand_u01(uint64_t *s) {
    return (double)(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
}

int main(void) {
    const size_t N = 200000;
    double *xs = (double *)malloc(N * sizeof(double));
    double *scores = (double *)malloc(N * sizeof(double));
    uint32_t *arms = (uint32_t *)malloc(N * sizeof(uint32_t));
    if (!xs || !scores || !arms) {
        fprintf(stderr, "OOM\n");
        return 1;
    }

    uint64_t rng = 0x20261122ULL;
    for (size_t i = 0; i < N; ++i) {
        xs[i] = rand_u01(&rng);
        scores[i] = fabs(rand_u01(&rng) * 3.0 - 1.5);
        arms[i] = (uint32_t)(i & 7U);
    }

    double ns_bet[5], ns_cs_eb[5], ns_cs_nm[5], ns_race[5];
    double ns_conf[5], ns_monoid_add[5], ns_monoid_merge[5];
    double ns_boa[5], ns_echo[5], ns_varopt[5], ns_rls[5], ns_sift[5];
    volatile double sink = 0.0;

    /* 1. ONS Betting E-Process */
    for (int r = 0; r < 5; ++r) {
        eproc_bet_t ep;
        eproc_bet_result_t out;
        eproc_bet_init(&ep, 0.48, 0.05, 0.0, 1.0, EPROC_SIDE_UPPER);
        uint64_t t0 = now_ns();
        eproc_bet_update_batch(&ep, xs, N, &out);
        uint64_t t1 = now_ns();
        ns_bet[r] = (double)(t1 - t0) / (double)N;
        sink += out.log_e;
    }

    /* 2. Empirical-Bernstein Confidence Sequence */
    for (int r = 0; r < 5; ++r) {
        eproc_cs_t cs;
        eproc_cs_result_t out;
        eproc_cs_init_eb(&cs, 0.05, 0.0, 1.0);
        uint64_t t0 = now_ns();
        eproc_cs_update_batch(&cs, xs, N, &out);
        uint64_t t1 = now_ns();
        ns_cs_eb[r] = (double)(t1 - t0) / (double)N;
        sink += out.width;
    }

    /* 3. Normal-Mixture Confidence Sequence */
    for (int r = 0; r < 5; ++r) {
        eproc_cs_t cs;
        eproc_cs_result_t out;
        eproc_cs_init_normal(&cs, 0.05, 0.5, 100.0);
        uint64_t t0 = now_ns();
        eproc_cs_update_batch(&cs, xs, N, &out);
        uint64_t t1 = now_ns();
        ns_cs_nm[r] = (double)(t1 - t0) / (double)N;
        sink += out.width;
    }

    /* 4. 8-Arm AgentRace */
    for (int r = 0; r < 5; ++r) {
        eproc_race_t race;
        eproc_race_status_t out;
        eproc_race_init(&race, 8, 0.05, 0.0, 1.0, 0.0);
        uint64_t t0 = now_ns();
        eproc_race_update_batch(&race, arms, xs, N, &out);
        uint64_t t1 = now_ns();
        ns_race[r] = (double)(t1 - t0) / (double)N;
        sink += out.leader_low;
    }

    /* 5. Multi-Rate AgACI + PID Conformal (5 rates x 4 experts, W=128) */
    const double alphas[5] = {0.50, 0.20, 0.10, 0.05, 0.01};
    const size_t N_CONF = 50000;
    for (int r = 0; r < 5; ++r) {
        eproc_conformal_t cp;
        eproc_conformal_out_t out;
        eproc_conformal_init(&cp, alphas, 5, 128, 0.05);
        uint64_t t0 = now_ns();
        eproc_conformal_update_batch(&cp, scores, N_CONF, &out);
        uint64_t t1 = now_ns();
        ns_conf[r] = (double)(t1 - t0) / (double)N_CONF;
        sink += out.quantiles[2];
    }

    /* 6. Distributed ConformalMonoid add + merge */
    eproc_cmonoid_t m_a, m_b, m_out;
    eproc_cmonoid_init(&m_a, 1.0);
    eproc_cmonoid_init(&m_b, 1.0);
    eproc_cmonoid_add_batch(&m_b, scores, 1000);
    for (int r = 0; r < 5; ++r) {
        eproc_cmonoid_init(&m_a, 1.0);
        uint64_t t0 = now_ns();
        eproc_cmonoid_add_batch(&m_a, scores, N);
        uint64_t t1 = now_ns();
        ns_monoid_add[r] = (double)(t1 - t0) / (double)N;
        sink += (double)m_a.count;
    }
    const size_t N_MERGES = 500000;
    for (int r = 0; r < 5; ++r) {
        uint64_t t0 = now_ns();
        for (size_t i = 0; i < N_MERGES; ++i) {
            eproc_cmonoid_merge(&m_a, &m_b, &m_out);
        }
        uint64_t t1 = now_ns();
        ns_monoid_merge[r] = (double)(t1 - t0) / (double)N_MERGES;
        sink += (double)m_out.count;
    }

    /* 7. SafeBOA (8 experts + 3 Ville gates + Boehm-2012 seqlock) */
    const size_t N_BOA = 50000;
    double epreds[8] = {0.48, 0.50, 0.52, 0.45, 0.55, 0.49, 0.51, 0.60};
    for (int r = 0; r < 5; ++r) {
        eproc_boa_t boa;
        eproc_boa_step_t out;
        eproc_boa_init(&boa, 8, 0, 0.05);
        uint64_t t0 = now_ns();
        for (size_t i = 0; i < N_BOA; ++i) {
            eproc_boa_update(&boa, epreds, xs[i], &out);
        }
        uint64_t t1 = now_ns();
        ns_boa[r] = (double)(t1 - t0) / (double)N_BOA;
        sink += out.fused_pred;
    }

    /* 8. EchoPool (provenance dedup + reliability weighted log-odds) */
    const size_t N_ECHO = 100000;
    for (int r = 0; r < 5; ++r) {
        eproc_echopool_t pool;
        eproc_echo_out_t out;
        eproc_echopool_init(&pool, 16, 64, 0.5);
        uint64_t t0 = now_ns();
        for (size_t i = 0; i < N_ECHO; ++i) {
            uint32_t src = (uint32_t)(i & 15U);
            uint32_t clm = (uint32_t)(i & 63U);
            uint64_t eid = (uint64_t)(i % 1024U);
            eproc_echopool_submit(&pool, src, clm, eid, 0.35, &out);
        }
        uint64_t t1 = now_ns();
        ns_echo[r] = (double)(t1 - t0) / (double)N_ECHO;
        sink += out.posterior_prob;
    }

    /* 9. VarOpt Horvitz-Thompson Reservoir (k=256) */
    const size_t N_VO = 100000;
    for (int r = 0; r < 5; ++r) {
        eproc_varopt_t vo;
        eproc_varopt_init(&vo, 256, 42ULL);
        uint64_t t0 = now_ns();
        for (size_t i = 0; i < N_VO; ++i) {
            eproc_varopt_push(&vo, (uint64_t)i, 0.1 + xs[i], xs[i]);
        }
        uint64_t t1 = now_ns();
        ns_varopt[r] = (double)(t1 - t0) / (double)N_VO;
        sink += eproc_varopt_estimate_sum(&vo);
    }

    /* 10. Online Ridge Leverage Score (d=16) */
    const size_t N_RLS = 50000;
    double xvec[16];
    for (int j = 0; j < 16; ++j) xvec[j] = 0.1 * (double)(j + 1);
    for (int r = 0; r < 5; ++r) {
        eproc_rls_t rls;
        eproc_rls_out_t out;
        eproc_rls_init(&rls, 16, 1.0, 2.0, 0.02, 42ULL);
        uint64_t t0 = now_ns();
        for (size_t i = 0; i < N_RLS; ++i) {
            xvec[0] = xs[i];
            eproc_rls_step(&rls, xvec, &out);
        }
        uint64_t t1 = now_ns();
        ns_rls[r] = (double)(t1 - t0) / (double)N_RLS;
        sink += out.leverage;
    }

    /* 11. SiftVOI selection across 2,048 channels (256 sites x 8) */
    const size_t N_SIFT = 2000;
    for (int r = 0; r < 5; ++r) {
        eproc_sift_t sf;
        eproc_sift_action_t act;
        eproc_sift_init(&sf, 256, 8, 1.5, 1.0);
        uint64_t t0 = now_ns();
        for (size_t i = 0; i < N_SIFT; ++i) {
            eproc_sift_select_next(&sf, 1.0, 1.0, 8.0, 8, &act);
            eproc_sift_observe_channel(&sf, (uint32_t)(i % 2048U), xs[i], 1);
        }
        uint64_t t1 = now_ns();
        ns_sift[r] = (double)(t1 - t0) / (double)N_SIFT;
        sink += act.evoi_per_cost;
    }

    printf("{\n");
    printf("  \"abi_version\": %u,\n", eproc_abi_version());
    printf("  \"simd_flags\": %u,\n", eproc_simd_flags());
    printf("  \"cmonoid_wire_bytes\": %zu,\n", eproc_cmonoid_wire_size());
    printf("  \"kernels_ns_per_op\": {\n");
    printf("    \"bet_eprocess\": %.2f,\n", median5(ns_bet));
    printf("    \"cs_empirical_bernstein\": %.2f,\n", median5(ns_cs_eb));
    printf("    \"cs_normal_mixture\": %.2f,\n", median5(ns_cs_nm));
    printf("    \"agent_race_8arms\": %.2f,\n", median5(ns_race));
    printf("    \"conformal_5rate_agaci_w128\": %.2f,\n", median5(ns_conf));
    printf("    \"cmonoid_add\": %.2f,\n", median5(ns_monoid_add));
    printf("    \"cmonoid_merge_640B\": %.2f,\n", median5(ns_monoid_merge));
    printf("    \"safe_boa_8experts_seqlock\": %.2f,\n", median5(ns_boa));
    printf("    \"echopool_submit\": %.2f,\n", median5(ns_echo));
    printf("    \"varopt_k256\": %.2f,\n", median5(ns_varopt));
    printf("    \"online_rls_d16\": %.2f,\n", median5(ns_rls));
    printf("    \"sift_voi_2048ch_select_and_obs\": %.2f\n", median5(ns_sift));
    printf("  },\n");
    printf("  \"_sink\": %.6f\n", (double)sink);
    printf("}\n");

    free(xs);
    free(scores);
    free(arms);
    return 0;
}
