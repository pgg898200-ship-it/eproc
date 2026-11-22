/*
 * CGo unity build bridge compiling canonical eproc C11 core sources directly
 * with zero prebuilt shared libraries and zero external dependencies.
 *
 * FP contraction is disabled in-file so builds work cleanly under Go's default
 * CGo compiler flag allowlist without requiring CGO_CFLAGS_ALLOW.
 */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#else
#pragma STDC FP_CONTRACT OFF
#endif

#include "../../csrc/eproc_eprocess.c"
#include "../../csrc/eproc_conformal.c"
#include "../../csrc/eproc_fusion.c"
#include "../../csrc/eproc_subsample.c"
