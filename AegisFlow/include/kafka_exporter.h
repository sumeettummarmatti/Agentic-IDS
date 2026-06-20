/**
 * @file kafka_exporter.h
 * @brief Kafka ExportFn for AegisFlow — publishes closed flows to a Kafka topic.
 *
 * Implements the ExportFn interface defined in exporter.h using librdkafka.
 * Each closed flow is serialised to JSON (reusing json_exporter_serialize)
 * and produced to the configured Kafka topic, keyed by server_id so that
 * all flows from the same capture agent land on the same partition.
 *
 * Usage:
 *   KafkaExporterCtx *kctx = kafka_exporter_create(&cfg, errbuf, sizeof(errbuf));
 *   exporter_chain_add(&chain, kafka_exporter_write, kctx);
 *   // ... run aegisflow ...
 *   kafka_exporter_destroy(kctx);
 *
 * AegisFlow — Kafka Exporter Extension
 * SPDX-License-Identifier: MIT
 */

#ifndef AEGISFLOW_KAFKA_EXPORTER_H
#define AEGISFLOW_KAFKA_EXPORTER_H

#include "exporter.h"
#include "flow.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * KafkaExporterConfig — passed to kafka_exporter_create()
 * ========================================================================= */
typedef struct {
    const char *bootstrap_servers;  /**< Kafka broker list, e.g. "localhost:9092" */
    const char *topic;              /**< Target topic, e.g. "raw-flows" */
    const char *server_id;          /**< This capture agent's ID, e.g. "MUM-01" */
    const char *geo_region;         /**< Geographic label, e.g. "asia-south1" */
    int         queue_buffering_max_ms;  /**< Max linger before flush (default 5) */
    int         batch_num_messages;      /**< Max messages per batch (default 1000) */
    int         verbose;            /**< 1 = print delivery reports to stderr */
} KafkaExporterConfig;

/* =========================================================================
 * KafkaExporterCtx — opaque context
 * ========================================================================= */
struct KafkaExporterCtx;
typedef struct KafkaExporterCtx KafkaExporterCtx;

/* =========================================================================
 * Public API
 * ========================================================================= */

/**
 * @brief Create and connect a KafkaExporterCtx.
 *
 * @param cfg     Non-NULL config.
 * @param errbuf  Error message buffer (≥ 256 bytes).
 * @return Heap-allocated context on success, NULL on error.
 */
KafkaExporterCtx *kafka_exporter_create(const KafkaExporterConfig *cfg,
                                        char *errbuf, size_t errbuflen);

/**
 * @brief ExportFn-compatible callback — serialise flow to JSON and produce.
 *
 * ctx must be a KafkaExporterCtx*.  Non-blocking: messages are queued
 * internally by librdkafka and flushed periodically.
 */
void kafka_exporter_write(const FlowRecord *record,
                          const FlowKey    *key,
                          void             *ctx);

/**
 * @brief Flush any queued messages and free all resources.
 * @param ctx May be NULL (no-op).
 */
void kafka_exporter_destroy(KafkaExporterCtx *ctx);

/**
 * @brief Return the number of messages produced (for stats).
 */
uint64_t kafka_exporter_produced(const KafkaExporterCtx *ctx);

/**
 * @brief Return the number of delivery errors.
 */
uint64_t kafka_exporter_errors(const KafkaExporterCtx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* AEGISFLOW_KAFKA_EXPORTER_H */
