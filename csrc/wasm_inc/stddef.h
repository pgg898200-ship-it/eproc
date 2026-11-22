#ifndef EPROC_WASM_STDDEF_H
#define EPROC_WASM_STDDEF_H

/* Freestanding stddef.h for wasm32-unknown-unknown builds of eproc (zero libc). */

typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;

#ifndef NULL
#define NULL ((void *)0)
#endif

#define offsetof(type, member) __builtin_offsetof(type, member)

#endif /* EPROC_WASM_STDDEF_H */
