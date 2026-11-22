#ifndef EPROC_WASM_STDATOMIC_H
#define EPROC_WASM_STDATOMIC_H

/* Freestanding stdatomic.h for wasm32-unknown-unknown builds of eproc. */

#include <stddef.h>
#include <stdint.h>

typedef enum memory_order {
    memory_order_relaxed = __ATOMIC_RELAXED,
    memory_order_consume = __ATOMIC_CONSUME,
    memory_order_acquire = __ATOMIC_ACQUIRE,
    memory_order_release = __ATOMIC_RELEASE,
    memory_order_acq_rel = __ATOMIC_ACQ_REL,
    memory_order_seq_cst = __ATOMIC_SEQ_CST
} memory_order;

#define atomic_thread_fence(order) __c11_atomic_thread_fence(order)
#define atomic_signal_fence(order) __c11_atomic_signal_fence(order)

#define atomic_load(obj)                 __c11_atomic_load((obj), __ATOMIC_SEQ_CST)
#define atomic_load_explicit(obj, ord)   __c11_atomic_load((obj), (ord))

#define atomic_store(obj, val)               __c11_atomic_store((obj), (val), __ATOMIC_SEQ_CST)
#define atomic_store_explicit(obj, val, ord) __c11_atomic_store((obj), (val), (ord))

#define atomic_compare_exchange_weak_explicit(obj, expected, desired, succ, fail) \
    __c11_atomic_compare_exchange_weak((obj), (expected), (desired), (succ), (fail))

#define atomic_compare_exchange_strong_explicit(obj, expected, desired, succ, fail) \
    __c11_atomic_compare_exchange_strong((obj), (expected), (desired), (succ), (fail))

#endif /* EPROC_WASM_STDATOMIC_H */
