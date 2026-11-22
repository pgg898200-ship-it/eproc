#ifndef EPROC_INTERNAL_H
#define EPROC_INTERNAL_H

/*
 * Internal numerical primitives, compensated summation, log-domain math,
 * and deterministic PRNG shared across eproc C11/C++17 translation units.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
#define EPROC_STATIC_ASSERT(cond, msg) static_assert((cond), msg)
#else
#define EPROC_STATIC_ASSERT(cond, msg) _Static_assert((cond), msg)
#endif

#define EPROC_LN2       0.693147180559945309417232121458176568
#define EPROC_LN3       1.098612288668109691395245236922525705
#define EPROC_INV_SQRT2 0.707106781186547524400844362104849039
/* ONS step size: 2 / (2 - log(3)) */
#define EPROC_ONS_ETA   (2.0 / (2.0 - EPROC_LN3))

/* NaN-safe clamp: NaN maps deterministically to lo */
static inline double eproc_clamp(double x, double lo, double hi) {
    if (isnan(x)) return lo;
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline uint64_t eproc_f64_to_u64(double x) {
    uint64_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

static inline double eproc_u64_to_f64(uint64_t u) {
    double x;
    memcpy(&x, &u, sizeof(x));
    return x;
}

/* Overflow-safe Neumaier improved Kahan-Babuska compensated summation.
 * Exact even when |x| > |sum|, and never generates (+Inf) + (-Inf) = NaN on overflow. */
static inline void eproc_kahan_add(double *sum, double *comp, double x) {
    if (!isfinite(x)) {
        return;
    }
    const double s = *sum;
    const double t = s + x;
    if (!isfinite(t)) {
        *sum = t;
        *comp = 0.0;
        return;
    }
    double c;
    if (fabs(s) >= fabs(x)) {
        c = (s - t) + x;
    } else {
        c = (x - t) + s;
    }
    if (isfinite(c)) {
        const double nc = *comp + c;
        *comp = isfinite(nc) ? nc : 0.0;
    }
    *sum = t;
}

static inline double eproc_kahan_total(double sum, double comp) {
    if (!isfinite(sum)) return sum;
    if (!isfinite(comp)) return sum;
    const double tot = sum + comp;
    return isnan(tot) ? sum : tot;
}

/* Total order on IEEE-754 doubles for bit-exact canonical operand ordering */
static inline int eproc_f64_less_total(double a, double b) {
    if (a < b) return 1;
    if (a > b) return 0;
    return eproc_f64_to_u64(a) < eproc_f64_to_u64(b);
}

/* Bit-exact commutative compensated merge of two (sum, comp) pairs */
static inline void eproc_kahan_merge_sym(double s1, double c1,
                                         double s2, double c2,
                                         double *s_out, double *c_out) {
    if (eproc_f64_less_total(s2, s1) ||
        (eproc_f64_to_u64(s1) == eproc_f64_to_u64(s2) && eproc_f64_less_total(c2, c1))) {
        double ts = s1; s1 = s2; s2 = ts;
        double tc = c1; c1 = c2; c2 = tc;
    }
    double s = s1;
    double c = c1;
    if (!isfinite(s) || !isfinite(s2)) {
        double tot = s + s2;
        if (isnan(tot)) {
            tot = 0.0;
        }
        *s_out = (tot == 0.0) ? 0.0 : tot;
        *c_out = 0.0;
        return;
    }
    eproc_kahan_add(&s, &c, s2);
    if (isfinite(c2)) {
        eproc_kahan_add(&s, &c, c2);
    }
    /* Canonicalize -0.0 -> +0.0 for deterministic wire bytes */
    if (s == 0.0) s = 0.0;
    if (c == 0.0) c = 0.0;
    *s_out = s;
    *c_out = c;
}

/* Overflow/underflow-safe exp capped at exp(700) to avoid +Inf */
static inline double eproc_safe_exp(double x) {
    if (isnan(x)) return 1.0;
    if (x > 700.0) return exp(700.0);
    if (x < -745.0) return 0.0;
    return exp(x);
}

/* Numerically stable log(exp(a) + exp(b)) */
static inline double eproc_logaddexp(double a, double b) {
    if (isnan(a) || isnan(b)) {
        if (!isnan(a)) return a;
        if (!isnan(b)) return b;
        return 0.0;
    }
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    if (isinf(a) || isinf(b)) {
        return (a > b) ? a : b;
    }
    double hi = (a >= b) ? a : b;
    double lo = (a >= b) ? b : a;
    double diff = lo - hi;
    if (diff < -745.0) return hi;
    return hi + log1p(exp(diff));
}

/* Overflow-safe logistic sigmoid */
static inline double eproc_sigmoid(double x) {
    if (isnan(x)) return 0.5;
    if (x >= 0.0) {
        if (x > 700.0) return 1.0;
        double z = exp(-x);
        return 1.0 / (1.0 + z);
    } else {
        if (x < -700.0) return 0.0;
        double z = exp(x);
        return z / (1.0 + z);
    }
}

/* Safe logit for p in (0, 1) */
static inline double eproc_logit(double p) {
    double pc = eproc_clamp(p, 1e-12, 1.0 - 1e-12);
    return log(pc / (1.0 - pc));
}

/* Standard normal CDF Phi(z) = 0.5 * erfc(-z / sqrt(2)) */
static inline double eproc_norm_cdf(double z) {
    if (isnan(z)) return 0.5;
    if (z > 38.0) return 1.0;
    if (z < -38.0) return 0.0;
    return 0.5 * erfc(-z * EPROC_INV_SQRT2);
}

/* Deterministic splitmix64 + xoroshiro128++ PRNG */
static inline uint64_t eproc_splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t eproc_rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline void eproc_rng_seed(uint64_t s[2], uint64_t seed) {
    uint64_t sm = seed;
    s[0] = eproc_splitmix64(&sm);
    s[1] = eproc_splitmix64(&sm);
    if (s[0] == 0ULL && s[1] == 0ULL) {
        s[0] = 0x9e3779b97f4a7c15ULL;
        s[1] = 0xbf58476d1ce4e5b9ULL;
    }
}

static inline uint64_t eproc_rng_next_u64(uint64_t s[2]) {
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = eproc_rotl64(s0 + s1, 17) + s0;
    s1 ^= s0;
    s[0] = eproc_rotl64(s0, 49) ^ s1 ^ (s1 << 21);
    s[1] = eproc_rotl64(s1, 28);
    return result;
}

static inline double eproc_rng_uniform01(uint64_t s[2]) {
    return (double)(eproc_rng_next_u64(s) >> 11) * (1.0 / 9007199254740992.0);
}

/* Endian-safe little-endian wire encoding helpers */
static inline void eproc_put_le64(uint8_t *p, uint64_t v) {
    p[0] = (uint8_t)(v & 0xFFU);
    p[1] = (uint8_t)((v >> 8) & 0xFFU);
    p[2] = (uint8_t)((v >> 16) & 0xFFU);
    p[3] = (uint8_t)((v >> 24) & 0xFFU);
    p[4] = (uint8_t)((v >> 32) & 0xFFU);
    p[5] = (uint8_t)((v >> 40) & 0xFFU);
    p[6] = (uint8_t)((v >> 48) & 0xFFU);
    p[7] = (uint8_t)((v >> 56) & 0xFFU);
}

static inline uint64_t eproc_get_le64(const uint8_t *p) {
    return ((uint64_t)p[0])
         | ((uint64_t)p[1] << 8)
         | ((uint64_t)p[2] << 16)
         | ((uint64_t)p[3] << 24)
         | ((uint64_t)p[4] << 32)
         | ((uint64_t)p[5] << 40)
         | ((uint64_t)p[6] << 48)
         | ((uint64_t)p[7] << 56);
}

#endif /* EPROC_INTERNAL_H */
