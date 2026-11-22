#ifndef EPROC_WASM_MATH_H
#define EPROC_WASM_MATH_H

/*
 * Freestanding math.h for wasm32-unknown-unknown builds of eproc (zero libc/libm).
 *
 * - Native WASM f64 instructions: fabs, sqrt, floor, ceil, copysign, fmin, fmax,
 *   isnan, isinf, isfinite (zero JS boundary crossing).
 * - Host JS Math imports (env.log, env.log1p, env.exp, env.tanh).
 * - Pure-C double-precision erfc(double) (< 1e-15 max relative error vs libm).
 */

#define INFINITY (__builtin_inff())
#define NAN      (__builtin_nanf(""))
#define HUGE_VAL (__builtin_huge_val())

#define isnan(x)    __builtin_isnan(x)
#define isinf(x)    __builtin_isinf(x)
#define isfinite(x) __builtin_isfinite(x)

static inline double fabs(double x) {
    return __builtin_fabs(x);
}

static inline double sqrt(double x) {
    return __builtin_sqrt(x);
}

static inline double floor(double x) {
    return __builtin_floor(x);
}

static inline double ceil(double x) {
    return __builtin_ceil(x);
}

static inline double copysign(double x, double y) {
    return __builtin_copysign(x, y);
}

/* IEEE-754 fmin/fmax: return the non-NaN argument when one operand is NaN. */
static inline double fmin(double x, double y) {
    if (__builtin_isnan(x)) {
        return y;
    }
    if (__builtin_isnan(y)) {
        return x;
    }
    if (x < y) {
        return x;
    }
    if (y < x) {
        return y;
    }
    return (__builtin_copysign(1.0, x) < 0.0) ? x : y;
}

static inline double fmax(double x, double y) {
    if (__builtin_isnan(x)) {
        return y;
    }
    if (__builtin_isnan(y)) {
        return x;
    }
    if (x > y) {
        return x;
    }
    if (y > x) {
        return y;
    }
    return (__builtin_copysign(1.0, x) > 0.0) ? x : y;
}

/* Transcendental functions: imported from JS Math via env.* (defined/imported in shim). */
__attribute__((import_module("env"), import_name("log")))
double log(double x);

__attribute__((import_module("env"), import_name("log1p")))
double log1p(double x);

__attribute__((import_module("env"), import_name("exp")))
double exp(double x);

__attribute__((import_module("env"), import_name("tanh")))
double tanh(double x);

/* Accurate pure-C erfc(double) implemented in csrc/eproc_wasm_shim.c (< 1e-15 rel err). */
double erfc(double x);

#endif /* EPROC_WASM_MATH_H */
