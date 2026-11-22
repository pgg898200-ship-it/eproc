CC ?= gcc
CFLAGS_COMMON = -std=c11 -ffp-contract=off -Wall -Wextra -Werror -Icsrc
CFLAGS_OPT    = $(CFLAGS_COMMON) -O3
CFLAGS_NATIVE = $(CFLAGS_OPT) -march=native
CFLAGS_PORT   = $(CFLAGS_OPT)
LDFLAGS       = -pthread -lm

CORE_SRCS = csrc/eproc_eprocess.c \
            csrc/eproc_conformal.c \
            csrc/eproc_fusion.c \
            csrc/eproc_subsample.c

HEADERS   = csrc/eproc.h \
            csrc/eproc_seqlock.h \
            csrc/eproc_internal.h \
            csrc/eproc_simd.h

.PHONY: all test test-portable test-asan test-tsan check-no-alloc clean

all: libeproc.so libeproc_portable.so

libeproc.so: $(CORE_SRCS) $(HEADERS)
	$(CC) $(CFLAGS_NATIVE) -fPIC -shared $(CORE_SRCS) -o $@ $(LDFLAGS)

libeproc_portable.so: $(CORE_SRCS) $(HEADERS)
	$(CC) $(CFLAGS_PORT) -fPIC -shared $(CORE_SRCS) -o $@ $(LDFLAGS)

libeproc_scalar.so: $(CORE_SRCS) $(HEADERS)
	$(CC) $(CFLAGS_PORT) -DEPROC_FORCE_SCALAR -fPIC -shared $(CORE_SRCS) -o $@ $(LDFLAGS)

test_c_core: csrc/test_c_core.c libeproc.so $(HEADERS)
	$(CC) $(CFLAGS_NATIVE) csrc/test_c_core.c -L. -Wl,-rpath,'$$ORIGIN' -leproc -o $@ $(LDFLAGS)

test_c_core_portable: csrc/test_c_core.c libeproc_portable.so $(HEADERS)
	$(CC) $(CFLAGS_PORT) csrc/test_c_core.c -L. -Wl,-rpath,'$$ORIGIN' -leproc_portable -o $@ $(LDFLAGS)

test_c_core_scalar: csrc/test_c_core.c libeproc_scalar.so $(HEADERS)
	$(CC) $(CFLAGS_PORT) -DEPROC_FORCE_SCALAR csrc/test_c_core.c -L. -Wl,-rpath,'$$ORIGIN' -leproc_scalar -o $@ $(LDFLAGS)

test_c_core_asan: csrc/test_c_core.c $(CORE_SRCS) $(HEADERS)
	$(CC) $(CFLAGS_COMMON) -O2 -g -march=native -fno-omit-frame-pointer -fsanitize=address,undefined,float-cast-overflow \
		csrc/test_c_core.c $(CORE_SRCS) -o $@ $(LDFLAGS)

test_c_core_tsan: csrc/test_c_core.c $(CORE_SRCS) $(HEADERS)
	$(CC) $(CFLAGS_COMMON) -O2 -g -march=native -fno-omit-frame-pointer -fsanitize=thread -Wno-tsan \
		csrc/test_c_core.c $(CORE_SRCS) -o $@ $(LDFLAGS)

check-no-alloc: libeproc.so libeproc_portable.so libeproc_scalar.so
	@if nm -u libeproc.so | grep -Eq '\b(malloc|calloc|realloc|free)\b'; then \
		echo "ERROR: libeproc.so references heap allocator!"; exit 1; \
	fi
	@if nm -u libeproc_portable.so | grep -Eq '\b(malloc|calloc|realloc|free)\b'; then \
		echo "ERROR: libeproc_portable.so references heap allocator!"; exit 1; \
	fi
	@if nm -u libeproc_scalar.so | grep -Eq '\b(malloc|calloc|realloc|free)\b'; then \
		echo "ERROR: libeproc_scalar.so references heap allocator!"; exit 1; \
	fi
	@echo "Verified: 0 heap allocator symbols in libeproc.so, libeproc_portable.so, and libeproc_scalar.so"

test: test_c_core test_c_core_portable test_c_core_scalar check-no-alloc
	@set -e; \
	OUT_NAT=$$(./test_c_core); printf "%s\n" "$$OUT_NAT"; \
	OUT_PORT=$$(./test_c_core_portable); printf "%s\n" "$$OUT_PORT"; \
	OUT_SCAL=$$(./test_c_core_scalar); printf "%s\n" "$$OUT_SCAL"; \
	FP_NAT=$$(printf "%s\n" "$$OUT_NAT" | grep '^\[fingerprint\]'); \
	FP_PORT=$$(printf "%s\n" "$$OUT_PORT" | grep '^\[fingerprint\]'); \
	FP_SCAL=$$(printf "%s\n" "$$OUT_SCAL" | grep '^\[fingerprint\]'); \
	if [ -z "$$FP_NAT" ] || [ "$$FP_NAT" != "$$FP_PORT" ] || [ "$$FP_NAT" != "$$FP_SCAL" ]; then \
		echo "ERROR: 3-way fingerprint mismatch: $$FP_NAT vs $$FP_PORT vs $$FP_SCAL"; exit 1; \
	fi; \
	echo "Verified: native AVX2 == portable runtime-AVX2 == pure scalar 0-ULP BIT-IDENTICAL ($$FP_NAT)"

test-asan: test_c_core_asan
	ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ./test_c_core_asan

test-tsan: test_c_core_tsan
	TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1 ./test_c_core_tsan

clean:
	rm -f libeproc.so libeproc_portable.so libeproc_scalar.so test_c_core test_c_core_portable test_c_core_scalar test_c_core_asan test_c_core_tsan
