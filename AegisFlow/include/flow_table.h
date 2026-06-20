/**
 * @file flow_table.h
 * @brief uthash-based O(1) flow table for AegisFlow.
 *
 * Manages the lifetime of FlowRecord objects:
 *   - Insert (create) a new flow on first packet
 *   - Lookup an existing flow by 5-tuple key
 *   - Remove and export flows on FIN/RST or timeout
 *   - Periodic expiry scan
 *   - Optional pthread mutex for thread safety
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#ifndef AEGISFLOW_FLOW_TABLE_H
#define AEGISFLOW_FLOW_TABLE_H

#include <stdint.h>
#include <sys/time.h>

#include "flow.h"
#include "exporter.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * FlowTable — opaque handle
 * ========================================================================= */
struct FlowTable;
typedef struct FlowTable FlowTable;

/* =========================================================================
 * FlowTableConfig — parameters for creating a flow table
 * ========================================================================= */
typedef struct {
    int      tcp_timeout_sec;    /**< Inactivity timeout for TCP flows (default 120) */
    int      udp_timeout_sec;    /**< Inactivity timeout for UDP flows (default 60) */
    uint32_t max_flows;          /**< Max concurrent flows (0 = unlimited) */
    int      thread_safe;        /**< Non-zero = protect table with a mutex */

    /** Called when a flow is closed (FIN/RST or timeout). May be NULL. */
    ExportFn on_flow_close;
    void    *on_flow_close_ctx;  /**< Opaque context forwarded to on_flow_close */
} FlowTableConfig;

/* =========================================================================
 * Flow table statistics
 * ========================================================================= */
typedef struct {
    uint64_t flows_created;     /**< Total flows ever inserted */
    uint64_t flows_closed;      /**< Flows closed by FIN/RST */
    uint64_t flows_expired;     /**< Flows closed by timeout */
    uint64_t flows_active;      /**< Currently active flows */
    uint64_t packets_processed; /**< Total packets through the table */
} FlowTableStats;

/* =========================================================================
 * Public API
 * ========================================================================= */

/**
 * @brief Create and initialise a new FlowTable.
 *
 * @param cfg Non-NULL configuration struct (copied internally).
 * @return Heap-allocated FlowTable on success, NULL on memory failure.
 */
FlowTable *flow_table_create(const FlowTableConfig *cfg);

/**
 * @brief Process one packet through the flow table.
 *
 * - Looks up the 5-tuple key in the hash table.
 * - Creates a new FlowRecord if not found.
 * - Updates all statistics for the matching record (features_update).
 * - Closes the flow immediately if TCP FIN or RST is observed.
 *
 * Thread-safe if cfg.thread_safe was non-zero at creation.
 *
 * @param table Non-NULL FlowTable.
 * @param pkt   Non-NULL parsed packet descriptor.
 */
void flow_table_update(FlowTable *table, const PacketInfo *pkt);

/**
 * @brief Expire all flows whose last-seen timestamp is beyond the timeout.
 *
 * Should be called periodically (e.g., every 1–5 seconds) from the
 * capture loop.  Each expired flow is passed to on_flow_close before removal.
 *
 * Thread-safe if cfg.thread_safe was non-zero at creation.
 *
 * @param table Non-NULL FlowTable.
 * @param now   Current wall-clock time (may be NULL → gettimeofday() used).
 * @return Number of flows expired in this call.
 */
uint32_t flow_table_expire(FlowTable *table, const struct timeval *now);

/**
 * @brief Flush (close and export) ALL remaining active flows.
 *
 * Typically called at program shutdown to export incomplete flows.
 *
 * @param table Non-NULL FlowTable.
 * @return Number of flows flushed.
 */
uint32_t flow_table_flush(FlowTable *table);

/**
 * @brief Look up a flow by its canonical key (read-only).
 *
 * Returns a const pointer to the internal FlowRecord — do NOT modify
 * the returned record or call any mutating operations while holding it
 * unless the table is externally locked.
 *
 * @return Pointer to FlowRecord if found, NULL otherwise.
 */
const FlowRecord *flow_table_lookup(const FlowTable *table, const FlowKey *key);

/**
 * @brief Retrieve cumulative flow table statistics.
 * @param table Non-NULL FlowTable.
 * @param stats Output statistics struct.
 */
void flow_table_get_stats(const FlowTable *table, FlowTableStats *stats);

/**
 * @brief Return the number of currently active flows.
 */
uint32_t flow_table_count(const FlowTable *table);

/**
 * @brief Destroy the flow table and free all memory.
 *
 * Does NOT flush/export remaining flows — call flow_table_flush() first
 * if that is desired.
 *
 * @param table FlowTable to destroy (may be NULL — no-op).
 */
void flow_table_destroy(FlowTable *table);

#ifdef __cplusplus
}
#endif

#endif /* AEGISFLOW_FLOW_TABLE_H */
