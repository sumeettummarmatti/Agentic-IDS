/**
 * @file utils.c
 * @brief Utility implementations for AegisFlow.
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#include "utils.h"

#include <arpa/inet.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* =========================================================================
 * Internal state
 * ========================================================================= */
static LogLevel g_log_level = LOG_LEVEL_INFO;

static const char *const LOG_LEVEL_NAMES[] = {
    "DEBUG", "INFO", "WARN", "ERROR", "NONE"
};

/* =========================================================================
 * Log level management
 * ========================================================================= */
void utils_set_log_level(LogLevel level) {
    g_log_level = level;
}

LogLevel utils_get_log_level(void) {
    return g_log_level;
}

/* =========================================================================
 * Internal logging implementation
 * ========================================================================= */
void utils_log(LogLevel level, const char *file, int line,
               const char *fmt, ...) {
    if (level < g_log_level) {
        return;
    }

    /* Get current time */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm tm_info;
    localtime_r(&tv.tv_sec, &tm_info);

    char timebuf[32];
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%dT%H:%M:%S", &tm_info);

    FILE *out = (level >= LOG_LEVEL_WARN) ? stderr : stdout;

    /* Extract just the filename from the full path */
    const char *basename = strrchr(file, '/');
    if (!basename) basename = strrchr(file, '\\');
    basename = basename ? basename + 1 : file;

    const char *level_str = (level < LOG_LEVEL_NONE)
                            ? LOG_LEVEL_NAMES[level] : "UNKNOWN";

    fprintf(out, "[%s.%03ld] [%-5s] [%s:%d] ",
            timebuf, (long)(tv.tv_usec / 1000),
            level_str, basename, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);

    fputc('\n', out);
    fflush(out);
}

/* =========================================================================
 * Timestamp helpers
 * ========================================================================= */
int64_t timeval_diff_usec(const struct timeval *a, const struct timeval *b) {
    int64_t sec_diff  = (int64_t)b->tv_sec  - (int64_t)a->tv_sec;
    int64_t usec_diff = (int64_t)b->tv_usec - (int64_t)a->tv_usec;
    return sec_diff * 1000000LL + usec_diff;
}

double timeval_to_double(const struct timeval *tv) {
    return (double)tv->tv_sec + (double)tv->tv_usec * 1e-6;
}

int utils_now(struct timeval *tv) {
    return gettimeofday(tv, NULL);
}

/* =========================================================================
 * Safe memory allocation
 * ========================================================================= */
void *xmalloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "[AEGISFLOW] FATAL: malloc(%zu) failed — out of memory\n", size);
        abort();
    }
    return ptr;
}

void *xcalloc(size_t nmemb, size_t size) {
    void *ptr = calloc(nmemb, size);
    if (!ptr) {
        fprintf(stderr, "[AEGISFLOW] FATAL: calloc(%zu, %zu) failed — out of memory\n",
                nmemb, size);
        abort();
    }
    return ptr;
}

void *xrealloc(void *old_ptr, size_t size) {
    void *ptr = realloc(old_ptr, size);
    if (!ptr && size > 0) {
        fprintf(stderr, "[AEGISFLOW] FATAL: realloc(%zu) failed — out of memory\n", size);
        abort();
    }
    return ptr;
}

/* =========================================================================
 * IPv4 address formatting
 * ========================================================================= */
char *utils_ipv4_to_str(uint32_t ip, char *buf, size_t buflen) {
    /* ip is in network byte order — inet_ntop expects the same */
    struct in_addr addr = { .s_addr = ip };
    inet_ntop(AF_INET, &addr, buf, (socklen_t)buflen);
    return buf;
}
