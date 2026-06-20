/**
 * @file utils.h
 * @brief Utility macros and helper declarations for AegisFlow.
 *
 * Provides:
 *   - Structured logging macros (LOG_DEBUG/INFO/WARN/ERROR)
 *   - Timestamp arithmetic helpers
 *   - Safe memory allocation wrappers (abort on OOM)
 *   - Portable min/max macros
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#ifndef AEGISFLOW_UTILS_H
#define AEGISFLOW_UTILS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Log levels
 * ========================================================================= */
typedef enum {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO  = 1,
    LOG_LEVEL_WARN  = 2,
    LOG_LEVEL_ERROR = 3,
    LOG_LEVEL_NONE  = 4
} LogLevel;

/** Set the minimum log level; messages below this level are suppressed. */
void utils_set_log_level(LogLevel level);

/** Get the current log level. */
LogLevel utils_get_log_level(void);

/**
 * @brief Internal logging function — use the macros below instead.
 */
void utils_log(LogLevel level, const char *file, int line,
               const char *fmt, ...)
    __attribute__((format(printf, 4, 5)));

/* =========================================================================
 * Logging macros
 *
 * Usage:
 *   LOG_INFO("Captured %u packets", count);
 *   LOG_ERROR("Failed to open device: %s", err);
 * ========================================================================= */
#ifndef NDEBUG
#  define LOG_DEBUG(...) utils_log(LOG_LEVEL_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#else
#  define LOG_DEBUG(...)  ((void)0)
#endif

#define LOG_INFO(...)  utils_log(LOG_LEVEL_INFO,  __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...)  utils_log(LOG_LEVEL_WARN,  __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) utils_log(LOG_LEVEL_ERROR, __FILE__, __LINE__, __VA_ARGS__)

/* =========================================================================
 * Timestamp helpers
 * ========================================================================= */

/**
 * @brief Compute the difference between two timevals in microseconds.
 * @return Signed difference: (b - a) in µs. Negative if a > b.
 */
int64_t timeval_diff_usec(const struct timeval *a, const struct timeval *b);

/**
 * @brief Convert a timeval to a double (seconds since epoch).
 */
double timeval_to_double(const struct timeval *tv);

/**
 * @brief Get current wall-clock time into *tv.
 * @return 0 on success, -1 on error.
 */
int utils_now(struct timeval *tv);

/* =========================================================================
 * Safe memory allocation (abort on failure — suitable for high-perf paths)
 * ========================================================================= */

/** malloc that aborts on NULL. */
void *xmalloc(size_t size);

/** calloc that aborts on NULL. */
void *xcalloc(size_t nmemb, size_t size);

/** realloc that aborts on NULL. */
void *xrealloc(void *ptr, size_t size);

/* =========================================================================
 * Portable min/max (type-generic via GCC __typeof__ extension)
 * ========================================================================= */
#ifdef __GNUC__
#  define AEGIS_MIN(a, b) \
       ({ __typeof__(a) _a = (a); __typeof__(b) _b = (b); _a < _b ? _a : _b; })
#  define AEGIS_MAX(a, b) \
       ({ __typeof__(a) _a = (a); __typeof__(b) _b = (b); _a > _b ? _a : _b; })
#else
#  define AEGIS_MIN(a, b) ((a) < (b) ? (a) : (b))
#  define AEGIS_MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

/* =========================================================================
 * IP address formatting helper
 * ========================================================================= */

/**
 * @brief Format a 32-bit IPv4 address (network byte order) into a string.
 * @param ip   IPv4 address in network byte order.
 * @param buf  Output buffer — must be at least 16 bytes.
 * @return buf
 */
char *utils_ipv4_to_str(uint32_t ip, char *buf, size_t buflen);

/* =========================================================================
 * Misc
 * ========================================================================= */

/** Return the number of elements in a stack-allocated array. */
#define ARRAY_LEN(arr) (sizeof(arr) / sizeof((arr)[0]))

/** Mark a variable as intentionally unused (suppress compiler warnings). */
#define UNUSED(x) ((void)(x))

#ifdef __cplusplus
}
#endif

#endif /* AEGISFLOW_UTILS_H */
