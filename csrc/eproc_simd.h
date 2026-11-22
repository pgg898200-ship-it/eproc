#ifndef EPROC_SIMD_H
#define EPROC_SIMD_H

/*
 * Portable compile-time + runtime CPUID AVX2 dispatch for eproc.
 *
 * - Native builds (-mavx2 / -march=native): zero-branch static AVX2.
 * - Portable x86_64 builds (PyPI wheels without -mavx2): one-time atomic-cached
 *   __builtin_cpu_supports("avx2") check dispatching into target("avx2") kernels
 *   (100% SIGILL-safe on SSE2-only CPUs, 0-ULP bit-identical to pure scalar).
 * - ARM64 / WASM / -DEPROC_FORCE_SCALAR: pure portable scalar path.
 */

#include <stdint.h>

#if defined(__AVX2__) && !defined(EPROC_FORCE_SCALAR) && !defined(__wasm__)
#  define EPROC_BUILD_AVX2   1
#  define EPROC_STATIC_AVX2  1
#  define EPROC_RUNTIME_AVX2 0
#  define EPROC_TARGET_AVX2
#  if defined(__GNUC__) || defined(__clang__)
#    define EPROC_AVX2_INLINE static inline __attribute__((always_inline))
#  else
#    define EPROC_AVX2_INLINE static inline
#  endif
#elif (defined(__GNUC__) || defined(__clang__)) \
      && (defined(__x86_64__) || defined(_M_X64)) \
      && !defined(__wasm__) && !defined(EPROC_FORCE_SCALAR)
#  define EPROC_BUILD_AVX2   1
#  define EPROC_STATIC_AVX2  0
#  define EPROC_RUNTIME_AVX2 1
#  define EPROC_TARGET_AVX2  __attribute__((target("avx2")))
#  define EPROC_AVX2_INLINE  static inline __attribute__((always_inline, target("avx2")))
#else
#  define EPROC_BUILD_AVX2   0
#  define EPROC_STATIC_AVX2  0
#  define EPROC_RUNTIME_AVX2 0
#  define EPROC_TARGET_AVX2
#  define EPROC_AVX2_INLINE  static inline
#endif

#if EPROC_BUILD_AVX2
#  include <immintrin.h>
#endif

static inline int eproc_use_avx2(void) {
#if EPROC_STATIC_AVX2
    return 1;
#elif EPROC_RUNTIME_AVX2
    static int g_eproc_avx2_state = -1;
    int s = __atomic_load_n(&g_eproc_avx2_state, __ATOMIC_RELAXED);
    if (__builtin_expect(s < 0, 0)) {
        __builtin_cpu_init();
        s = __builtin_cpu_supports("avx2") ? 1 : 0;
        __atomic_store_n(&g_eproc_avx2_state, s, __ATOMIC_RELAXED);
    }
    return s;
#else
    return 0;
#endif
}

#endif /* EPROC_SIMD_H */
