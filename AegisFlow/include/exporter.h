/**
 * @file exporter.h
 * @brief Flow export interface for AegisFlow.
 *
 * Defines the generic ExportFn callback type plus concrete JSON and CSV
 * exporter implementations.  The interface is designed for forward
 * compatibility: a Kafka producer exporter would be a new source file
 * implementing the same ExportFn signature.
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#ifndef AEGISFLOW_EXPORTER_H
#define AEGISFLOW_EXPORTER_H

#include <stdio.h>
#include "flow.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Generic exporter interface
 *
 * Any exporter must implement this signature.  The flow table calls
 * on_flow_close(record, key, ctx) when a flow closes.
 *
 * @param record  Non-NULL, completed FlowRecord (read-only).
 * @param key     Non-NULL, canonical FlowKey for this record.
 * @param ctx     Opaque exporter context (e.g., FILE*, KafkaHandle*).
 * ========================================================================= */
typedef void (*ExportFn)(const FlowRecord *record,
                         const FlowKey    *key,
                         void             *ctx);

/* =========================================================================
 * JSON Exporter
 * ========================================================================= */

/**
 * @brief Serialise a FlowRecord to a heap-allocated JSON string.
 *
 * Uses cJSON internally.  The returned string is malloc'd and MUST be
 * freed by the caller with free().
 *
 * @param record  Non-NULL completed FlowRecord.
 * @param key     Non-NULL canonical FlowKey.
 * @return Heap-allocated null-terminated JSON string, or NULL on OOM.
 */
char *json_exporter_serialize(const FlowRecord *record, const FlowKey *key);

/**
 * @brief ExportFn-compatible JSON exporter.
 *
 * Writes a JSON object (terminated by newline) to the FILE* passed as ctx.
 * If ctx is NULL, writes to stdout.
 *
 * @param record Non-NULL FlowRecord.
 * @param key    Non-NULL FlowKey.
 * @param ctx    FILE* to write to (or NULL for stdout).
 */
void json_exporter_write(const FlowRecord *record,
                         const FlowKey    *key,
                         void             *ctx);

/* =========================================================================
 * CSV Exporter
 * ========================================================================= */

/**
 * @brief Write the CICFlowMeter-compatible CSV header line.
 *
 * Call once before the first csv_exporter_write() call.
 *
 * @param fp  Output FILE* (NULL → stdout).
 */
void csv_exporter_write_header(FILE *fp);

/**
 * @brief ExportFn-compatible CSV exporter.
 *
 * Writes one CSV data line for the given flow to the FILE* passed as ctx.
 * If ctx is NULL, writes to stdout.
 *
 * @param record Non-NULL FlowRecord.
 * @param key    Non-NULL FlowKey.
 * @param ctx    FILE* to write to (or NULL for stdout).
 */
void csv_exporter_write(const FlowRecord *record,
                        const FlowKey    *key,
                        void             *ctx);

/* =========================================================================
 * Composite exporter helper
 *
 * Allows chaining multiple exporters together so that one on_flow_close
 * callback fires all of them.
 * ========================================================================= */

#define AEGIS_MAX_EXPORTERS 8

typedef struct {
    ExportFn fns[AEGIS_MAX_EXPORTERS];
    void    *ctxs[AEGIS_MAX_EXPORTERS];
    int      count;
} ExporterChain;

/**
 * @brief Initialise an ExporterChain to zero exporters.
 */
void exporter_chain_init(ExporterChain *chain);

/**
 * @brief Register one exporter in a chain.
 * @return 0 on success, -1 if the chain is full.
 */
int exporter_chain_add(ExporterChain *chain, ExportFn fn, void *ctx);

/**
 * @brief ExportFn-compatible dispatcher — calls all registered exporters.
 * @param ctx Must point to an ExporterChain.
 */
void exporter_chain_dispatch(const FlowRecord *record,
                             const FlowKey    *key,
                             void             *ctx);

#ifdef __cplusplus
}
#endif

#endif /* AEGISFLOW_EXPORTER_H */
