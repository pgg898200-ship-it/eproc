#ifndef EPROC_WASM_STRING_H
#define EPROC_WASM_STRING_H

/* Freestanding string.h for wasm32-unknown-unknown builds of eproc (zero libc). */

#include <stddef.h>

void *memcpy(void *restrict dst, const void *restrict src, size_t n);
void *memmove(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
int memcmp(const void *a, const void *b, size_t n);

#endif /* EPROC_WASM_STRING_H */
