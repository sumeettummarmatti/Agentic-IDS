/**
 * @file aegisflow.h
 * @brief Top-level public API for the AegisFlow feature extraction engine.
 *
 * This is the single include header for downstream consumers.  It brings in
 * all required sub-headers and provides the high-level AegisFlow context
 * type that wires together capture → flow table → exporter.
 *
 * Typical usage:
 *
 *   #include "aegisflow.h"
 *
 *   AegisConfig cfg = aegisflow_default_config();
 *   cfg.device       = "eth0";
 *   cfg.output_json  = stdout;
 *
 *   AegisContext *ctx = aegisflow_create(&cfg);
 *   aegisflow_run(ctx);          // blocks until SIGINT / EOF
 *   aegisflow_destroy(ctx);
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#ifndef AEGISFLOW_H
#define AEGISFLOW_H

/* Pull in all component headers */
#include "flow.h"
#include "capture.h"
#include "flow_table.h"
#include "aegis_features.h"
#include "exporter.h"
#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Version
 * ========================================================================= */
#define AEGISFLOW_VERSION_MAJOR 1
#define AEGISFLOW_VERSION_MINOR 0
#define AEGISFLOW_VERSION_PATCH 0
#define AEGISFLOW_VERSION_STR   "1.0.0"

/* =========================================================================
 * AegisConfig — top-level configuration
 * ========================================================================= */
typedef struct {
    /* ── Capture settings ── */
    const char *device;          /**< Live interface (e.g. "eth0"); NULL if using pcap_file */
    const char *pcap_file;       /**< Offline PCAP path; NULL if using live device */
    const char *bpf_filter;      /**< BPF filter string (NULL = "ip") */
    int         promisc;         /**< Promiscuous mode (default 1) */

    /* ── Flow table settings ── */
    int         tcp_timeout_sec; /**< TCP flow idle timeout (default 120) */
    int         udp_timeout_sec; /**< UDP flow idle timeout (default 60) */
    uint32_t    max_flows;       /**< Max concurrent flows (0 = unlimited) */
    int         thread_safe;     /**< Enable mutex for flow table (default 0) */

    /* ── Export settings ── */
    FILE       *output_json;     /**< Write JSON exports here (NULL = disabled) */
    FILE       *output_csv;      /**< Write CSV exports here (NULL = disabled) */
    int         csv_header;      /**< Write CSV header line (default 1) */

    /* ── Runtime settings ── */
    uint64_t    max_packets;     /**< Stop after N packets (0 = unlimited) */
    int         expire_interval_sec; /**< How often to scan for timed-out flows (default 5) */
    LogLevel    log_level;       /**< Logging verbosity (default LOG_LEVEL_INFO) */
} AegisConfig;

/* =========================================================================
 * AegisContext — opaque engine handle
 * ========================================================================= */
struct AegisContext;
typedef struct AegisContext AegisContext;

/* =========================================================================
 * Public API
 * ========================================================================= */

/**
 * @brief Return a config struct initialised with sane defaults.
 *
 * All pointer fields are NULL.  Call this first, then override specific
 * fields before passing to aegisflow_create().
 */
AegisConfig aegisflow_default_config(void);

/**
 * @brief Create an AegisContext from the provided configuration.
 *
 * Opens the libpcap handle, creates the flow table, and wires up exporters.
 *
 * @param cfg     Non-NULL configuration.
 * @param errbuf  Buffer for error message (≥ 256 bytes).
 * @return Heap-allocated context on success, NULL on error (errbuf filled).
 */
AegisContext *aegisflow_create(const AegisConfig *cfg,
                               char *errbuf, size_t errbuflen);

/**
 * @brief Run the capture-process-export loop.
 *
 * Blocks until EOF (offline), SIGINT, or an error.  Internally calls
 * flow_table_expire() at the configured interval.
 *
 * @param ctx Non-NULL AegisContext.
 * @return 0 on clean exit, -1 on capture error.
 */
int aegisflow_run(AegisContext *ctx);

/**
 * @brief Request an in-progress run to stop (safe from signal handlers).
 * @param ctx Non-NULL AegisContext.
 */
void aegisflow_stop(AegisContext *ctx);

/**
 * @brief Retrieve combined runtime statistics.
 * @param ctx   Non-NULL AegisContext.
 * @param cap   Output capture statistics (may be NULL).
 * @param table Output flow table statistics (may be NULL).
 */
void aegisflow_get_stats(const AegisContext *ctx,
                         CaptureStats     *cap,
                         FlowTableStats   *table);

/**
 * @brief Flush remaining flows, close the capture device, and free all memory.
 * @param ctx Context to destroy (may be NULL — no-op).
 */
void aegisflow_destroy(AegisContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* AEGISFLOW_H */
