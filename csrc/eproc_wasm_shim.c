#if !defined(__GNUC__) || defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

/*
 * Freestanding WebAssembly shim for eproc (zero libc, zero WASI, zero Emscripten).
 *
 * Provides:
 *   1. memcpy / memmove / memset / memcmp (guarded with no_builtin against self-recursion)
 *   2. Double-precision erfc(double) (< 1e-15 max relative error vs glibc libm)
 *   3. 16-byte aligned coalescing free-list allocator over WASM linear memory
 *      backed by memory.grow (starting at linker symbol __heap_base)
 *   4. Struct sizeof helpers and compile-time _Static_assert layout proofs for JS FFI
 *
 * License: Apache-2.0
 */

#include "eproc.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ============================================================================
 * 1. FREESTANDING MEMORY ROUTINES
 * ============================================================================ */

#if defined(__clang__)
#define EPROC_NO_BUILTIN __attribute__((no_builtin("memcpy", "memmove", "memset", "memcmp")))
#else
#define EPROC_NO_BUILTIN
#endif

EPROC_NO_BUILTIN
void *memcpy(void *restrict dst, const void *restrict src, size_t n) {
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    size_t i = 0;
    for (; i + 8U <= n; i += 8U) {
        uint64_t w;
        const uint8_t *sp = s + i;
        w = ((uint64_t)sp[0])
          | ((uint64_t)sp[1] << 8)
          | ((uint64_t)sp[2] << 16)
          | ((uint64_t)sp[3] << 24)
          | ((uint64_t)sp[4] << 32)
          | ((uint64_t)sp[5] << 40)
          | ((uint64_t)sp[6] << 48)
          | ((uint64_t)sp[7] << 56);
        uint8_t *dp = d + i;
        dp[0] = (uint8_t)(w & 0xFFU);
        dp[1] = (uint8_t)((w >> 8) & 0xFFU);
        dp[2] = (uint8_t)((w >> 16) & 0xFFU);
        dp[3] = (uint8_t)((w >> 24) & 0xFFU);
        dp[4] = (uint8_t)((w >> 32) & 0xFFU);
        dp[5] = (uint8_t)((w >> 40) & 0xFFU);
        dp[6] = (uint8_t)((w >> 48) & 0xFFU);
        dp[7] = (uint8_t)((w >> 56) & 0xFFU);
    }
    for (; i < n; ++i) {
        d[i] = s[i];
    }
    return dst;
}

EPROC_NO_BUILTIN
void *memmove(void *dst, const void *src, size_t n) {
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    if (d == s || n == 0U) {
        return dst;
    }
    if ((uintptr_t)d < (uintptr_t)s) {
        for (size_t i = 0; i < n; ++i) {
            d[i] = s[i];
        }
    } else {
        for (size_t i = n; i > 0U; --i) {
            d[i - 1U] = s[i - 1U];
        }
    }
    return dst;
}

EPROC_NO_BUILTIN
void *memset(void *dst, int c, size_t n) {
    uint8_t *d = (uint8_t *)dst;
    const uint8_t v = (uint8_t)c;
    for (size_t i = 0; i < n; ++i) {
        d[i] = v;
    }
    return dst;
}

EPROC_NO_BUILTIN
int memcmp(const void *a, const void *b, size_t n) {
    const uint8_t *pa = (const uint8_t *)a;
    const uint8_t *pb = (const uint8_t *)b;
    for (size_t i = 0; i < n; ++i) {
        if (pa[i] != pb[i]) {
            return (int)pa[i] - (int)pb[i];
        }
    }
    return 0;
}

/* ============================================================================
 * 2. FULL DOUBLE-PRECISION PURE-C erfc(double) (< 1e-15 MAX RELATIVE ERROR)
 * ============================================================================
 *
 * - |x| < 0.5: alternating Maclaurin series for erf(x), 1 - erf(x).
 * - |x| >= 0.5: 24-term Chebyshev polynomial approximation of
 *       g(x) = (1 + sqrt(pi)*x) * exp(x^2) * erfc(x)
 *   over the rational map t = ((x - 0.5) - 3.5) / ((x - 0.5) + 3.5) in [-1, 1],
 *   evaluated via Clenshaw recurrence, paired with exact 1/16384-grid splitting
 *   of x^2 so exp(-x_hi^2) * exp(-(x - x_hi)*(x + x_hi)) loses zero bits.
 * Verified max relative error vs glibc libm erfc: 6.76e-16 across [-30, +26.5].
 */
double erfc(double x) {
    static const double SQRT_PI = 1.77245385090551602729816748334114518;
    static const double TWO_OVER_SQRT_PI = 1.12837916709551257389615890312154517;
    static const double CHEB_P = 3.5;
    static const double CHEB_C[24] = {
         1.09994050957608125e+00,
        -9.66164286521524335e-02,
        -1.36139706854263281e-02,
         1.46281453926167156e-02,
        -5.45921896987202216e-03,
         1.30889350112462291e-03,
        -2.03278152873097388e-04,
         1.40496169413215164e-05,
         1.79810168387531798e-06,
        -5.23803382990878363e-07,
         1.37411520354750808e-08,
         1.17220012921283353e-08,
        -1.17529971009473908e-09,
        -2.62797400160663663e-10,
         4.50430718988804560e-11,
         6.87944149878433094e-12,
        -1.55356501823975009e-12,
        -2.25584176328328215e-13,
         5.25549214882456015e-14,
         9.07516591462458687e-15,
        -1.71956697718412807e-15,
        -4.12386789141378269e-16,
         4.95255595398811850e-17,
         1.97390801465836811e-17
    };

    if (isnan(x)) {
        return x;
    }
    if (x == 0.0) {
        return 1.0;
    }
    const int neg = (x < 0.0);
    const double ax = neg ? -x : x;
    if (ax >= 27.3) {
        return neg ? 2.0 : 0.0;
    }
    if (ax < 0.5) {
        const double x2 = ax * ax;
        double term = ax;
        double sum = ax;
        for (int n = 1; n < 20; ++n) {
            term *= -x2 / (double)n;
            const double add = term / (double)(2 * n + 1);
            sum += add;
            if (fabs(add) <= fabs(sum) * 1e-17) {
                break;
            }
        }
        const double erf_val = TWO_OVER_SQRT_PI * sum;
        return neg ? (1.0 + erf_val) : (1.0 - erf_val);
    }

    const double u = ax - 0.5;
    const double t = (u - CHEB_P) / (u + CHEB_P);
    const double two_t = 2.0 * t;
    double b1 = 0.0;
    double b2 = 0.0;
    for (int k = 23; k >= 1; --k) {
        const double b0 = two_t * b1 - b2 + CHEB_C[k];
        b2 = b1;
        b1 = b0;
    }
    const double g = t * b1 - b2 + CHEB_C[0];
    const double erfcx = g / (1.0 + SQRT_PI * ax);

    const double xh = floor(ax * 16384.0) * (1.0 / 16384.0);
    const double xl = ax - xh;
    double res;
    if (ax < 26.0) {
        res = erfcx * exp(-(xh * xh)) * exp(-xl * (ax + xh));
    } else {
        /* Rescale to prevent intermediate subnormal precision loss for x in [26, 27.3) */
        res = (erfcx * exp(-(xh * xh - 600.0)) * exp(-xl * (ax + xh))) * exp(-600.0);
    }
    return neg ? (2.0 - res) : res;
}

double eproc_wasm_erfc(double x) {
    return erfc(x);
}

/* ============================================================================
 * 3. COALESCING FREE-LIST ALLOCATOR OVER WASM LINEAR MEMORY
 * ============================================================================ */

extern unsigned char __heap_base;

#define EPROC_WASM_ALLOC_MAGIC 0x45505241U /* "EPRA" */
#define EPROC_WASM_FREE_MAGIC  0x45505246U /* "EPRF" */
#define EPROC_WASM_PAGE_SIZE   65536U

typedef struct eproc_wasm_blk {
    size_t size;
    struct eproc_wasm_blk *next;
    uint32_t magic;
    uint32_t _pad;
} eproc_wasm_blk_t;

_Static_assert(sizeof(eproc_wasm_blk_t) == 16U, "allocator header must be 16-byte aligned");

static uintptr_t g_heap_cur = 0U;
static eproc_wasm_blk_t *g_free_head = NULL;
static size_t g_heap_used = 0U;

static inline uintptr_t eproc_align16(uintptr_t v) {
    return (v + 15U) & ~(uintptr_t)15U;
}

void eproc_wasm_reset(void) {
    g_heap_cur = eproc_align16((uintptr_t)&__heap_base);
    g_free_head = NULL;
    g_heap_used = 0U;
}

size_t eproc_wasm_heap_used(void) {
    return g_heap_used;
}

void *eproc_wasm_alloc(size_t size) {
    if (size == 0U) {
        size = 16U;
    }
    if (size > 0x7FFFFFF0U) {
        return NULL;
    }
    size = (size + 15U) & ~(size_t)15U;

    /* First-fit search in address-ordered free list */
    eproc_wasm_blk_t *prev = NULL;
    eproc_wasm_blk_t *cur = g_free_head;
    while (cur != NULL) {
        if (cur->size >= size) {
            if (cur->size >= size + sizeof(eproc_wasm_blk_t) + 16U) {
                eproc_wasm_blk_t *rem =
                    (eproc_wasm_blk_t *)((uint8_t *)(cur + 1) + size);
                rem->size = cur->size - size - sizeof(eproc_wasm_blk_t);
                rem->next = cur->next;
                rem->magic = EPROC_WASM_FREE_MAGIC;
                rem->_pad = 0U;
                cur->size = size;
                if (prev != NULL) {
                    prev->next = rem;
                } else {
                    g_free_head = rem;
                }
            } else {
                if (prev != NULL) {
                    prev->next = cur->next;
                } else {
                    g_free_head = cur->next;
                }
            }
            cur->next = NULL;
            cur->magic = EPROC_WASM_ALLOC_MAGIC;
            g_heap_used += cur->size;
            return (void *)(cur + 1);
        }
        prev = cur;
        cur = cur->next;
    }

    /* Bump allocate from top of heap, growing WASM linear memory on demand */
    if (g_heap_cur == 0U) {
        g_heap_cur = eproc_align16((uintptr_t)&__heap_base);
    }
    const size_t total = sizeof(eproc_wasm_blk_t) + size;
    if (total < size || g_heap_cur > (uintptr_t)SIZE_MAX - total) {
        return NULL;
    }
    const uintptr_t needed_end = g_heap_cur + total;
    const size_t cur_pages = __builtin_wasm_memory_size(0);
    const size_t cur_bytes = cur_pages * EPROC_WASM_PAGE_SIZE;
    if (needed_end > cur_bytes) {
        const size_t delta_bytes = (size_t)needed_end - cur_bytes;
        const size_t delta_pages = (delta_bytes + (EPROC_WASM_PAGE_SIZE - 1U)) / EPROC_WASM_PAGE_SIZE;
        const size_t old_pages = __builtin_wasm_memory_grow(0, delta_pages);
        if (old_pages == (size_t)-1) {
            return NULL;
        }
    }

    eproc_wasm_blk_t *blk = (eproc_wasm_blk_t *)g_heap_cur;
    g_heap_cur = needed_end;
    blk->size = size;
    blk->next = NULL;
    blk->magic = EPROC_WASM_ALLOC_MAGIC;
    blk->_pad = 0U;
    g_heap_used += size;
    return (void *)(blk + 1);
}

void eproc_wasm_free(void *ptr) {
    if (ptr == NULL) {
        return;
    }
    eproc_wasm_blk_t *blk = ((eproc_wasm_blk_t *)ptr) - 1;
    if (blk->magic != EPROC_WASM_ALLOC_MAGIC) {
        return;
    }
    blk->magic = EPROC_WASM_FREE_MAGIC;
    g_heap_used -= blk->size;

    /* Insert into address-ordered free list and coalesce immediate neighbors */
    eproc_wasm_blk_t *prev = NULL;
    eproc_wasm_blk_t *cur = g_free_head;
    while (cur != NULL && (uintptr_t)cur < (uintptr_t)blk) {
        prev = cur;
        cur = cur->next;
    }

    blk->next = cur;
    if (prev != NULL) {
        prev->next = blk;
    } else {
        g_free_head = blk;
    }

    if (cur != NULL && ((uint8_t *)(blk + 1) + blk->size) == (uint8_t *)cur) {
        blk->size += sizeof(eproc_wasm_blk_t) + cur->size;
        blk->next = cur->next;
    }
    if (prev != NULL && ((uint8_t *)(prev + 1) + prev->size) == (uint8_t *)blk) {
        prev->size += sizeof(eproc_wasm_blk_t) + blk->size;
        prev->next = blk->next;
    }
}

/* ============================================================================
 * 4. STRUCT SIZE HELPERS FOR JS LINEAR-MEMORY ALLOCATION
 * ============================================================================ */

size_t eproc_wasm_sizeof_bet(void)               { return sizeof(eproc_bet_t); }
size_t eproc_wasm_sizeof_bet_result(void)        { return sizeof(eproc_bet_result_t); }
size_t eproc_wasm_sizeof_cs(void)                { return sizeof(eproc_cs_t); }
size_t eproc_wasm_sizeof_cs_result(void)         { return sizeof(eproc_cs_result_t); }
size_t eproc_wasm_sizeof_edetect(void)           { return sizeof(eproc_edetect_t); }
size_t eproc_wasm_sizeof_edetect_result(void)    { return sizeof(eproc_edetect_result_t); }
size_t eproc_wasm_sizeof_race(void)              { return sizeof(eproc_race_t); }
size_t eproc_wasm_sizeof_race_status(void)       { return sizeof(eproc_race_status_t); }
size_t eproc_wasm_sizeof_conformal(void)         { return sizeof(eproc_conformal_t); }
size_t eproc_wasm_sizeof_conformal_out(void)     { return sizeof(eproc_conformal_out_t); }
size_t eproc_wasm_sizeof_delayed_conformal(void) { return sizeof(eproc_delayed_conformal_t); }
size_t eproc_wasm_sizeof_budget_conformal(void)  { return sizeof(eproc_budget_conformal_t); }
size_t eproc_wasm_sizeof_cmonoid(void)           { return sizeof(eproc_cmonoid_t); }
size_t eproc_wasm_sizeof_boa(void)               { return sizeof(eproc_boa_t); }
size_t eproc_wasm_sizeof_boa_step(void)          { return sizeof(eproc_boa_step_t); }
size_t eproc_wasm_sizeof_echopool(void)          { return sizeof(eproc_echopool_t); }
size_t eproc_wasm_sizeof_echo_out(void)          { return sizeof(eproc_echo_out_t); }
size_t eproc_wasm_sizeof_varopt(void)            { return sizeof(eproc_varopt_t); }
size_t eproc_wasm_sizeof_rls(void)               { return sizeof(eproc_rls_t); }
size_t eproc_wasm_sizeof_rls_out(void)           { return sizeof(eproc_rls_out_t); }
size_t eproc_wasm_sizeof_sift(void)              { return sizeof(eproc_sift_t); }
size_t eproc_wasm_sizeof_sift_action(void)       { return sizeof(eproc_sift_action_t); }

/* ============================================================================
 * 5. COMPILE-TIME ABI LAYOUT PROOFS FOR JS DATAVIEW READERS
 * ============================================================================ */

/* eproc_bet_result_t */
_Static_assert(sizeof(eproc_bet_result_t) == 56U, "eproc_bet_result_t size");
_Static_assert(offsetof(eproc_bet_result_t, n) == 0U, "bet_res.n");
_Static_assert(offsetof(eproc_bet_result_t, e_value) == 8U, "bet_res.e_value");
_Static_assert(offsetof(eproc_bet_result_t, log_e) == 16U, "bet_res.log_e");
_Static_assert(offsetof(eproc_bet_result_t, max_log_e) == 24U, "bet_res.max_log_e");
_Static_assert(offsetof(eproc_bet_result_t, p_value) == 32U, "bet_res.p_value");
_Static_assert(offsetof(eproc_bet_result_t, mean) == 40U, "bet_res.mean");
_Static_assert(offsetof(eproc_bet_result_t, rejected) == 48U, "bet_res.rejected");

/* eproc_bet_t */
_Static_assert(offsetof(eproc_bet_t, n) == 152U, "bet.n");
_Static_assert(offsetof(eproc_bet_t, rejected_at) == 160U, "bet.rejected_at");
_Static_assert(offsetof(eproc_bet_t, n_invalid) == 168U, "bet.n_invalid");

/* eproc_cs_result_t */
_Static_assert(sizeof(eproc_cs_result_t) == 40U, "eproc_cs_result_t size");
_Static_assert(offsetof(eproc_cs_result_t, n) == 0U, "cs_res.n");
_Static_assert(offsetof(eproc_cs_result_t, mean) == 8U, "cs_res.mean");
_Static_assert(offsetof(eproc_cs_result_t, low) == 16U, "cs_res.low");
_Static_assert(offsetof(eproc_cs_result_t, high) == 24U, "cs_res.high");
_Static_assert(offsetof(eproc_cs_result_t, width) == 32U, "cs_res.width");

/* eproc_edetect_result_t & eproc_edetect_t */
_Static_assert(sizeof(eproc_edetect_result_t) == 32U, "eproc_edetect_result_t size");
_Static_assert(offsetof(eproc_edetect_result_t, n) == 0U, "ed_res.n");
_Static_assert(offsetof(eproc_edetect_result_t, log_stat) == 8U, "ed_res.log_stat");
_Static_assert(offsetof(eproc_edetect_result_t, stat) == 16U, "ed_res.stat");
_Static_assert(offsetof(eproc_edetect_result_t, alarm) == 24U, "ed_res.alarm");
_Static_assert(offsetof(eproc_edetect_t, log_cusum) == 176U, "ed.log_cusum");
_Static_assert(offsetof(eproc_edetect_t, log_thresh) == 192U, "ed.log_thresh");
_Static_assert(offsetof(eproc_edetect_t, alarm_at) == 200U, "ed.alarm_at");
_Static_assert(offsetof(eproc_edetect_t, n_alarms) == 208U, "ed.n_alarms");
_Static_assert(offsetof(eproc_edetect_t, n) == 216U, "ed.n");

/* eproc_race_status_t & eproc_race_t */
_Static_assert(sizeof(eproc_race_status_t) == 48U, "eproc_race_status_t size");
_Static_assert(offsetof(eproc_race_status_t, n_arms) == 0U, "race_st.n_arms");
_Static_assert(offsetof(eproc_race_status_t, n_active) == 4U, "race_st.n_active");
_Static_assert(offsetof(eproc_race_status_t, winner) == 8U, "race_st.winner");
_Static_assert(offsetof(eproc_race_status_t, leader) == 12U, "race_st.leader");
_Static_assert(offsetof(eproc_race_status_t, total_obs) == 16U, "race_st.total_obs");
_Static_assert(offsetof(eproc_race_status_t, leader_low) == 24U, "race_st.leader_low");
_Static_assert(offsetof(eproc_race_status_t, leader_high) == 32U, "race_st.leader_high");
_Static_assert(offsetof(eproc_race_status_t, runner_up_high) == 40U, "race_st.runner_up_high");
_Static_assert(offsetof(eproc_race_t, winner) == 40U, "race.winner");
_Static_assert(offsetof(eproc_race_t, active) == 56U, "race.active");
_Static_assert(offsetof(eproc_race_t, eliminated_at) == 120U, "race.eliminated_at");

/* eproc_conformal_out_t & eproc_conformal_t */
_Static_assert(sizeof(eproc_conformal_out_t) == 128U, "eproc_conformal_out_t size");
_Static_assert(offsetof(eproc_conformal_out_t, n_rates) == 0U, "cp_out.n_rates");
_Static_assert(offsetof(eproc_conformal_out_t, quantiles) == 8U, "cp_out.quantiles");
_Static_assert(offsetof(eproc_conformal_out_t, eff_alphas) == 48U, "cp_out.eff_alphas");
_Static_assert(offsetof(eproc_conformal_out_t, emp_coverage) == 88U, "cp_out.emp_coverage");
_Static_assert(offsetof(eproc_conformal_t, n_updates) == 32U, "cp.n_updates");
_Static_assert(offsetof(eproc_conformal_t, target_alphas) == 40U, "cp.target_alphas");
_Static_assert(offsetof(eproc_conformal_t, current_q) == 672U, "cp.current_q");

/* eproc_delayed_conformal_t & eproc_budget_conformal_t */
_Static_assert(offsetof(eproc_delayed_conformal_t, core) == 0U, "dc.core");
_Static_assert(offsetof(eproc_delayed_conformal_t, pending_count) == sizeof(eproc_conformal_t), "dc.pending_count");
_Static_assert(offsetof(eproc_budget_conformal_t, core) == 0U, "bc.core");
_Static_assert(offsetof(eproc_budget_conformal_t, n_completed) == sizeof(eproc_conformal_t) + 8U, "bc.n_completed");
_Static_assert(offsetof(eproc_budget_conformal_t, n_censored) == sizeof(eproc_conformal_t) + 16U, "bc.n_censored");

/* eproc_cmonoid_t */
_Static_assert(sizeof(eproc_cmonoid_t) == 640U, "eproc_cmonoid_t size");
_Static_assert(offsetof(eproc_cmonoid_t, count) == 0U, "cm.count");
_Static_assert(offsetof(eproc_cmonoid_t, bin_counts) == 8U, "cm.bin_counts");
_Static_assert(offsetof(eproc_cmonoid_t, sum) == 520U, "cm.sum");
_Static_assert(offsetof(eproc_cmonoid_t, sum_comp) == 528U, "cm.sum_comp");
_Static_assert(offsetof(eproc_cmonoid_t, min_val) == 552U, "cm.min_val");
_Static_assert(offsetof(eproc_cmonoid_t, max_val) == 560U, "cm.max_val");
_Static_assert(offsetof(eproc_cmonoid_t, scale_hint) == 576U, "cm.scale_hint");

/* eproc_boa_step_t & eproc_boa_t */
_Static_assert(sizeof(eproc_boa_step_t) == 48U, "eproc_boa_step_t size");
_Static_assert(offsetof(eproc_boa_step_t, fused_pred) == 0U, "boa_step.fused_pred");
_Static_assert(offsetof(eproc_boa_step_t, anchor_pred) == 8U, "boa_step.anchor_pred");
_Static_assert(offsetof(eproc_boa_step_t, fallback_active) == 16U, "boa_step.fallback_active");
_Static_assert(offsetof(eproc_boa_step_t, n_quarantined) == 20U, "boa_step.n_quarantined");
_Static_assert(offsetof(eproc_boa_step_t, sup_log_e) == 24U, "boa_step.sup_log_e");
_Static_assert(offsetof(eproc_boa_step_t, deg_log_e) == 32U, "boa_step.deg_log_e");
_Static_assert(offsetof(eproc_boa_step_t, n_updates) == 40U, "boa_step.n_updates");
_Static_assert(offsetof(eproc_boa_t, n_updates) == 32U, "boa.n_updates");
_Static_assert(offsetof(eproc_boa_t, n_poison_clipped) == 40U, "boa.n_poison_clipped");
_Static_assert(offsetof(eproc_boa_t, n_fallback_trips) == 48U, "boa.n_fallback_trips");
_Static_assert(offsetof(eproc_boa_t, quarantined) == 2104U, "boa.quarantined");

/* eproc_echo_out_t & eproc_echopool_t */
_Static_assert(sizeof(eproc_echo_out_t) == 40U, "eproc_echo_out_t size");
_Static_assert(offsetof(eproc_echo_out_t, accepted) == 0U, "echo_out.accepted");
_Static_assert(offsetof(eproc_echo_out_t, posterior_prob) == 8U, "echo_out.posterior_prob");
_Static_assert(offsetof(eproc_echo_out_t, log_odds) == 16U, "echo_out.log_odds");
_Static_assert(offsetof(eproc_echo_out_t, source_reliability) == 24U, "echo_out.source_reliability");
_Static_assert(offsetof(eproc_echo_out_t, unique_count) == 32U, "echo_out.unique_count");
_Static_assert(offsetof(eproc_echopool_t, n_accepted) == 8U, "echo.n_accepted");
_Static_assert(offsetof(eproc_echopool_t, n_echo_dropped) == 16U, "echo.n_echo_dropped");

/* eproc_varopt_t & eproc_varopt_item_t */
_Static_assert(sizeof(eproc_varopt_item_t) == 40U, "eproc_varopt_item_t size");
_Static_assert(offsetof(eproc_varopt_t, count) == 4U, "vo.count");
_Static_assert(offsetof(eproc_varopt_t, n_heavy) == 8U, "vo.n_heavy");
_Static_assert(offsetof(eproc_varopt_t, tau) == 16U, "vo.tau");
_Static_assert(offsetof(eproc_varopt_t, total_weight) == 24U, "vo.total_weight");
_Static_assert(offsetof(eproc_varopt_t, total_weight_comp) == 32U, "vo.total_weight_comp");
_Static_assert(offsetof(eproc_varopt_t, n_seen) == 40U, "vo.n_seen");
_Static_assert(offsetof(eproc_varopt_t, items) == 64U, "vo.items");

/* eproc_rls_out_t & eproc_rls_t */
_Static_assert(sizeof(eproc_rls_out_t) == 32U, "eproc_rls_out_t size");
_Static_assert(offsetof(eproc_rls_out_t, keep) == 0U, "rls_out.keep");
_Static_assert(offsetof(eproc_rls_out_t, leverage) == 8U, "rls_out.leverage");
_Static_assert(offsetof(eproc_rls_out_t, prob) == 16U, "rls_out.prob");
_Static_assert(offsetof(eproc_rls_out_t, ht_weight) == 24U, "rls_out.ht_weight");
_Static_assert(offsetof(eproc_rls_t, n_seen) == 32U, "rls.n_seen");
_Static_assert(offsetof(eproc_rls_t, n_kept) == 40U, "rls.n_kept");

/* eproc_sift_action_t & eproc_sift_t */
_Static_assert(sizeof(eproc_sift_action_t) == 16U, "eproc_sift_action_t size");
_Static_assert(offsetof(eproc_sift_action_t, action_type) == 0U, "sift_act.action_type");
_Static_assert(offsetof(eproc_sift_action_t, target_idx) == 4U, "sift_act.target_idx");
_Static_assert(offsetof(eproc_sift_action_t, evoi_per_cost) == 8U, "sift_act.evoi_per_cost");
_Static_assert(offsetof(eproc_sift_t, site_log_prob) == 48U, "sift.site_log_prob");
_Static_assert(offsetof(eproc_sift_t, chan_prob) == 8240U, "sift.chan_prob");
