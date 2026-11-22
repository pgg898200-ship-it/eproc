#ifndef EPROC_SEQLOCK_H
#define EPROC_SEQLOCK_H

/*
 * Lock-free Boehm-2012 atomic seqlock for zero-torn-read concurrent
 * snapshots of online inference state (portable C11 & C++17).
 *
 * Writer protocol (single-writer or CAS-serialized multi-writer):
 *   1. CAS seq even -> odd (acq_rel)
 *   2. release fence
 *   3. Write payload words as relaxed atomic uint64_t stores
 *   4. Store seq = odd + 1 (release)
 *
 * Reader protocol (wait-free fast path, zero locks):
 *   1. Load s1 = seq (acquire); if odd, spin with CPU pause
 *   2. Read payload words as relaxed atomic uint64_t loads
 *   3. acquire fence
 *   4. Load s2 = seq (relaxed); retry if s1 != s2
 *
 * Verified under TSan with atomic payload words (zero C11/C++17 data-race UB).
 */

#ifdef __cplusplus
#include <atomic>
#else
#include <stdatomic.h>
#endif

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#define EPROC_CPU_PAUSE() __asm__ __volatile__("pause" ::: "memory")
#elif defined(__aarch64__) || defined(__arm__)
#define EPROC_CPU_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#else
#define EPROC_CPU_PAUSE() ((void)0)
#endif

/* ============================================================================
 * 1. Typed atomic seqlock API (C11 _Atomic uint64_t / C++17 std::atomic<uint64_t>)
 * ============================================================================ */
#ifdef __cplusplus

static inline uint64_t eproc_seqlock_write_begin(std::atomic<uint64_t> *seq) {
    uint64_t s = seq->load(std::memory_order_relaxed);
    for (;;) {
        if ((s & 1ULL) != 0ULL) {
            EPROC_CPU_PAUSE();
            s = seq->load(std::memory_order_relaxed);
            continue;
        }
        if (seq->compare_exchange_weak(
                s, s + 1ULL, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            std::atomic_thread_fence(std::memory_order_release);
            return s + 1ULL;
        }
        EPROC_CPU_PAUSE();
    }
}

static inline void eproc_seqlock_write_end(std::atomic<uint64_t> *seq, uint64_t odd_seq) {
    seq->store(odd_seq + 1ULL, std::memory_order_release);
}

static inline void eproc_seqlock_store_words(std::atomic<uint64_t> *dst,
                                             const void *src,
                                             size_t n_words) {
    const uint8_t *p = (const uint8_t *)src;
    for (size_t i = 0; i < n_words; ++i) {
        uint64_t w;
        memcpy(&w, p + i * sizeof(uint64_t), sizeof(uint64_t));
        dst[i].store(w, std::memory_order_relaxed);
    }
}

static inline uint64_t eproc_seqlock_read_words(const std::atomic<uint64_t> *seq,
                                                const std::atomic<uint64_t> *src,
                                                void *dst,
                                                size_t n_words) {
    uint8_t *p = (uint8_t *)dst;
    for (;;) {
        uint64_t s1 = seq->load(std::memory_order_acquire);
        if ((s1 & 1ULL) != 0ULL) {
            EPROC_CPU_PAUSE();
            continue;
        }
        for (size_t i = 0; i < n_words; ++i) {
            uint64_t w = src[i].load(std::memory_order_relaxed);
            memcpy(p + i * sizeof(uint64_t), &w, sizeof(uint64_t));
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        uint64_t s2 = seq->load(std::memory_order_relaxed);
        if (s1 == s2) {
            return s1;
        }
        EPROC_CPU_PAUSE();
    }
}

#else /* C11 */

static inline uint64_t eproc_seqlock_write_begin(_Atomic uint64_t *seq) {
    uint64_t s = atomic_load_explicit(seq, memory_order_relaxed);
    for (;;) {
        if ((s & 1ULL) != 0ULL) {
            EPROC_CPU_PAUSE();
            s = atomic_load_explicit(seq, memory_order_relaxed);
            continue;
        }
        if (atomic_compare_exchange_weak_explicit(
                seq, &s, s + 1ULL, memory_order_acq_rel, memory_order_relaxed)) {
            atomic_thread_fence(memory_order_release);
            return s + 1ULL;
        }
        EPROC_CPU_PAUSE();
    }
}

static inline void eproc_seqlock_write_end(_Atomic uint64_t *seq, uint64_t odd_seq) {
    atomic_store_explicit(seq, odd_seq + 1ULL, memory_order_release);
}

static inline void eproc_seqlock_store_words(_Atomic uint64_t *dst,
                                             const void *src,
                                             size_t n_words) {
    const uint8_t *p = (const uint8_t *)src;
    for (size_t i = 0; i < n_words; ++i) {
        uint64_t w;
        memcpy(&w, p + i * sizeof(uint64_t), sizeof(uint64_t));
        atomic_store_explicit(&dst[i], w, memory_order_relaxed);
    }
}

static inline uint64_t eproc_seqlock_read_words(const _Atomic uint64_t *seq,
                                                const _Atomic uint64_t *src,
                                                void *dst,
                                                size_t n_words) {
    uint8_t *p = (uint8_t *)dst;
    for (;;) {
        uint64_t s1 = atomic_load_explicit(seq, memory_order_acquire);
        if ((s1 & 1ULL) != 0ULL) {
            EPROC_CPU_PAUSE();
            continue;
        }
        for (size_t i = 0; i < n_words; ++i) {
            uint64_t w = atomic_load_explicit(&src[i], memory_order_relaxed);
            memcpy(p + i * sizeof(uint64_t), &w, sizeof(uint64_t));
        }
        atomic_thread_fence(memory_order_acquire);
        uint64_t s2 = atomic_load_explicit(seq, memory_order_relaxed);
        if (s1 == s2) {
            return s1;
        }
        EPROC_CPU_PAUSE();
    }
}

#endif /* __cplusplus */

/* ============================================================================
 * 2. Zero-UB plain uint64_t ABI seqlock API (for C-ABI structs like eproc_boa_t
 *    that store uint64_t fields in public headers for C/C++/FFI compatibility).
 *    Uses GCC/Clang __atomic_* builtins directly on uint64_t* (zero type-punning
 *    UB, full TSan visibility, works identically in C11 and C++17).
 * ============================================================================ */
#if defined(__GNUC__) || defined(__clang__)

static inline void eproc_seqlock_init_u64(uint64_t *seq, uint64_t *words, size_t n_words) {
    __atomic_store_n(seq, 0ULL, __ATOMIC_RELAXED);
    for (size_t i = 0; i < n_words; ++i) {
        __atomic_store_n(&words[i], 0ULL, __ATOMIC_RELAXED);
    }
}

static inline uint64_t eproc_seqlock_write_begin_u64(uint64_t *seq) {
    uint64_t s = __atomic_load_n(seq, __ATOMIC_RELAXED);
    for (;;) {
        if ((s & 1ULL) != 0ULL) {
            EPROC_CPU_PAUSE();
            s = __atomic_load_n(seq, __ATOMIC_RELAXED);
            continue;
        }
        if (__atomic_compare_exchange_n(
                seq, &s, s + 1ULL, 1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            __atomic_thread_fence(__ATOMIC_RELEASE);
            return s + 1ULL;
        }
        EPROC_CPU_PAUSE();
    }
}

static inline void eproc_seqlock_write_end_u64(uint64_t *seq, uint64_t odd_seq) {
    __atomic_store_n(seq, odd_seq + 1ULL, __ATOMIC_RELEASE);
}

static inline void eproc_seqlock_store_words_u64(uint64_t *dst,
                                                 const void *src,
                                                 size_t n_words) {
    const uint8_t *p = (const uint8_t *)src;
    for (size_t i = 0; i < n_words; ++i) {
        uint64_t w;
        memcpy(&w, p + i * sizeof(uint64_t), sizeof(uint64_t));
        __atomic_store_n(&dst[i], w, __ATOMIC_RELAXED);
    }
}

static inline uint64_t eproc_seqlock_read_words_u64(const uint64_t *seq,
                                                    const uint64_t *src,
                                                    void *dst,
                                                    size_t n_words) {
    uint8_t *p = (uint8_t *)dst;
    for (;;) {
        uint64_t s1 = __atomic_load_n(seq, __ATOMIC_ACQUIRE);
        if ((s1 & 1ULL) != 0ULL) {
            EPROC_CPU_PAUSE();
            continue;
        }
        for (size_t i = 0; i < n_words; ++i) {
            uint64_t w = __atomic_load_n(&src[i], __ATOMIC_RELAXED);
            memcpy(p + i * sizeof(uint64_t), &w, sizeof(uint64_t));
        }
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        uint64_t s2 = __atomic_load_n(seq, __ATOMIC_RELAXED);
        if (s1 == s2) {
            return s1;
        }
        EPROC_CPU_PAUSE();
    }
}

#else

/* Fallback for non-GCC/Clang toolchains */
#ifdef __cplusplus
typedef std::atomic<uint64_t> eproc_seqlock_u64_atom_t;
#else
typedef _Atomic uint64_t eproc_seqlock_u64_atom_t;
#endif

static inline void eproc_seqlock_init_u64(uint64_t *seq, uint64_t *words, size_t n_words) {
    *seq = 0ULL;
    for (size_t i = 0; i < n_words; ++i) {
        words[i] = 0ULL;
    }
}

static inline uint64_t eproc_seqlock_write_begin_u64(uint64_t *seq) {
    return eproc_seqlock_write_begin((eproc_seqlock_u64_atom_t *)(void *)seq);
}

static inline void eproc_seqlock_write_end_u64(uint64_t *seq, uint64_t odd_seq) {
    eproc_seqlock_write_end((eproc_seqlock_u64_atom_t *)(void *)seq, odd_seq);
}

static inline void eproc_seqlock_store_words_u64(uint64_t *dst,
                                                 const void *src,
                                                 size_t n_words) {
    eproc_seqlock_store_words((eproc_seqlock_u64_atom_t *)(void *)dst, src, n_words);
}

static inline uint64_t eproc_seqlock_read_words_u64(const uint64_t *seq,
                                                    const uint64_t *src,
                                                    void *dst,
                                                    size_t n_words) {
    return eproc_seqlock_read_words((const eproc_seqlock_u64_atom_t *)(const void *)seq,
                                    (const eproc_seqlock_u64_atom_t *)(const void *)src,
                                    dst, n_words);
}

#endif /* __GNUC__ || __clang__ */

#endif /* EPROC_SEQLOCK_H */
