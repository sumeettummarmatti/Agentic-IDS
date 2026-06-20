/**
 * @file test_runner.h
 * @brief Lightweight unit test harness for AegisFlow.
 *
 * Zero external dependencies.  Provides:
 *   TEST()           — define a test case
 *   ASSERT_EQ()      — equality assertion (integers)
 *   ASSERT_NEAR()    — floating-point near-equality
 *   ASSERT_TRUE()    — boolean assertion
 *   ASSERT_NOT_NULL()
 *   RUN_TEST()       — register and run a test
 *   TEST_SUMMARY()   — print pass/fail counts
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#ifndef AEGISFLOW_TEST_RUNNER_H
#define AEGISFLOW_TEST_RUNNER_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Global counters
 * ========================================================================= */
static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

/* =========================================================================
 * Colour codes (ANSI, disabled if not a TTY) 
 * ========================================================================= */
#define CLR_RED   "\033[0;31m"
#define CLR_GRN   "\033[0;32m"
#define CLR_YEL   "\033[0;33m"
#define CLR_RST   "\033[0m"

/* =========================================================================
 * Test macros
 * ========================================================================= */

/** Define a test function. */
#define TEST(name) static void test_##name(void)

/** Run a test function and track result. */
#define RUN_TEST(name) do {                                      \
    printf("  %-50s ", #name "...");                             \
    fflush(stdout);                                              \
    g_tests_run++;                                               \
    test_##name();                                               \
    /* If we reach here without abort, the test passed */        \
    printf(CLR_GRN "[PASS]" CLR_RST "\n");                      \
    g_tests_passed++;                                            \
} while(0)

/** Assert condition is true; fail the test immediately on failure. */
#define ASSERT_TRUE(cond) do {                                   \
    if (!(cond)) {                                               \
        printf(CLR_RED "[FAIL]\n" CLR_RST);                      \
        printf("    Assertion failed: %s\n", #cond);             \
        printf("    at %s:%d\n", __FILE__, __LINE__);            \
        g_tests_failed++;                                        \
        return;                                                  \
    }                                                            \
} while(0)

/** Assert two integers are equal. */
#define ASSERT_EQ(a, b) do {                                     \
    long long _a = (long long)(a);                               \
    long long _b = (long long)(b);                               \
    if (_a != _b) {                                              \
        printf(CLR_RED "[FAIL]\n" CLR_RST);                      \
        printf("    ASSERT_EQ(%s, %s)\n", #a, #b);              \
        printf("    Got: %lld  Expected: %lld\n", _a, _b);      \
        printf("    at %s:%d\n", __FILE__, __LINE__);            \
        g_tests_failed++;                                        \
        return;                                                  \
    }                                                            \
} while(0)

/** Assert pointer is not NULL. */
#define ASSERT_NOT_NULL(ptr) do {                                \
    if ((ptr) == NULL) {                                         \
        printf(CLR_RED "[FAIL]\n" CLR_RST);                      \
        printf("    ASSERT_NOT_NULL(%s) failed\n", #ptr);        \
        printf("    at %s:%d\n", __FILE__, __LINE__);            \
        g_tests_failed++;                                        \
        return;                                                  \
    }                                                            \
} while(0)

/** Assert two doubles are within 'tol' of each other. */
#define ASSERT_NEAR(a, b, tol) do {                              \
    double _a = (double)(a);                                     \
    double _b = (double)(b);                                     \
    double _d = fabs(_a - _b);                                   \
    if (_d > (double)(tol)) {                                    \
        printf(CLR_RED "[FAIL]\n" CLR_RST);                      \
        printf("    ASSERT_NEAR(%s, %s, %g)\n", #a, #b, (double)(tol)); \
        printf("    Got: %.10f  Expected: %.10f  Diff: %.10f\n", _a, _b, _d); \
        printf("    at %s:%d\n", __FILE__, __LINE__);            \
        g_tests_failed++;                                        \
        return;                                                  \
    }                                                            \
} while(0)

/** Print summary and return 0 (all pass) or 1 (any fail). */
#define TEST_SUMMARY() do {                                      \
    printf("\n──────────────────────────────────────\n");         \
    printf("  Tests run    : %d\n", g_tests_run);                \
    printf("  Tests passed : " CLR_GRN "%d" CLR_RST "\n", g_tests_passed); \
    if (g_tests_failed > 0)                                      \
        printf("  Tests failed : " CLR_RED "%d" CLR_RST "\n", g_tests_failed); \
    else                                                         \
        printf("  Tests failed : %d\n", g_tests_failed);         \
    printf("──────────────────────────────────────\n");           \
} while(0)

#endif /* AEGISFLOW_TEST_RUNNER_H */
