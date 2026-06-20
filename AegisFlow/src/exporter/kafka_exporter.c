/**
 * @file kafka_exporter.c
 * @brief librdkafka-based Kafka ExportFn for AegisFlow.
 *
 * When AegisFlow closes a flow (TCP FIN/RST, timeout, or EOF) it calls every
 * registered ExportFn.  This exporter:
 *
 *   1. Reuses json_exporter_serialize() to build the JSON payload — the exact
 *      same JSON that the existing json_exporter writes to a file.
 *   2. Injects extra fields:  server_id, geo_region, captured_at_us.
 *   3. Produces the message to Kafka, keyed by server_id, so all flows from
 *      one capture agent are ordered on a single partition.
 *   4. Tracks produced / error counters for the summary banner.
 *
 * Build requirements:
 *   - librdkafka  (apt: librdkafka-dev  |  brew: librdkafka)
 *   - cJSON       (already a dependency of AegisFlow core)
 *
 * SPDX-License-Identifier: MIT
 */

#include "kafka_exporter.h"
#include "exporter.h"          /* json_exporter_serialize */
#include "aegis_features.h"
#include "utils.h"

#include <librdkafka/rdkafka.h>
#include <cJSON.h>

#include <arpa/inet.h>
#include <inttypes.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

/* =========================================================================
 * Internal context
 * ========================================================================= */
struct KafkaExporterCtx {
    rd_kafka_t       *rk;           /* producer handle */
    rd_kafka_topic_t *rkt;          /* topic handle    */
    char              server_id[64];
    char              geo_region[64];
    int               verbose;
    atomic_uint_fast64_t produced;
    atomic_uint_fast64_t errors;
};

/* =========================================================================
 * Delivery report callback — called by rd_kafka_poll() for each message
 * ========================================================================= */
static void dr_msg_cb(rd_kafka_t *rk,
                      const rd_kafka_message_t *rkmessage,
                      void *opaque) {
    KafkaExporterCtx *kctx = (KafkaExporterCtx *)opaque;
    (void)rk;

    if (rkmessage->err) {
        atomic_fetch_add(&kctx->errors, 1);
        if (kctx->verbose) {
            fprintf(stderr, "[kafka_exporter] delivery error: %s\n",
                    rd_kafka_err2str(rkmessage->err));
        }
    } else {
        atomic_fetch_add(&kctx->produced, 1);
        if (kctx->verbose) {
            fprintf(stderr, "[kafka_exporter] delivered to %s [%" PRId32 "] "
                    "@ offset %" PRId64 "\n",
                    rd_kafka_topic_name(rkmessage->rkt),
                    rkmessage->partition,
                    rkmessage->offset);
        }
    }
}

/* =========================================================================
 * kafka_exporter_create
 * ========================================================================= */
KafkaExporterCtx *kafka_exporter_create(const KafkaExporterConfig *cfg,
                                        char *errbuf, size_t errbuflen) {
    if (!cfg || !cfg->bootstrap_servers || !cfg->topic) {
        snprintf(errbuf, errbuflen, "kafka_exporter: bootstrap_servers and topic are required");
        return NULL;
    }

    KafkaExporterCtx *kctx = calloc(1, sizeof(*kctx));
    if (!kctx) {
        snprintf(errbuf, errbuflen, "kafka_exporter: OOM allocating context");
        return NULL;
    }

    /* Copy string fields */
    snprintf(kctx->server_id,  sizeof(kctx->server_id),  "%s",
             cfg->server_id  ? cfg->server_id  : "unknown");
    snprintf(kctx->geo_region, sizeof(kctx->geo_region), "%s",
             cfg->geo_region ? cfg->geo_region : "unknown");
    kctx->verbose = cfg->verbose;

    atomic_init(&kctx->produced, 0);
    atomic_init(&kctx->errors,   0);

    /* ── Build rdkafka config ── */
    rd_kafka_conf_t *conf = rd_kafka_conf_new();
    char rd_errbuf[256];

    /* Broker list */
    if (rd_kafka_conf_set(conf, "bootstrap.servers", cfg->bootstrap_servers,
                          rd_errbuf, sizeof(rd_errbuf)) != RD_KAFKA_CONF_OK) {
        snprintf(errbuf, errbuflen, "kafka_exporter: bootstrap.servers: %s", rd_errbuf);
        rd_kafka_conf_destroy(conf);
        free(kctx);
        return NULL;
    }

    /* Linger — collect up to N ms of messages before flushing */
    int linger_ms = cfg->queue_buffering_max_ms > 0 ? cfg->queue_buffering_max_ms : 5;
    char linger_str[16];
    snprintf(linger_str, sizeof(linger_str), "%d", linger_ms);
    rd_kafka_conf_set(conf, "queue.buffering.max.ms", linger_str, NULL, 0);

    /* Batch size */
    int batch = cfg->batch_num_messages > 0 ? cfg->batch_num_messages : 1000;
    char batch_str[16];
    snprintf(batch_str, sizeof(batch_str), "%d", batch);
    rd_kafka_conf_set(conf, "batch.num.messages", batch_str, NULL, 0);

    /* Reliability: wait for leader ack */
    rd_kafka_conf_set(conf, "request.required.acks", "1", NULL, 0);

    /* Delivery report callback — gives us per-message success/error */
    rd_kafka_conf_set_dr_msg_cb(conf, dr_msg_cb);
    rd_kafka_conf_set_opaque(conf, kctx);

    /* ── Create producer ── */
    kctx->rk = rd_kafka_new(RD_KAFKA_PRODUCER, conf, rd_errbuf, sizeof(rd_errbuf));
    if (!kctx->rk) {
        snprintf(errbuf, errbuflen, "kafka_exporter: rd_kafka_new failed: %s", rd_errbuf);
        /* conf ownership transferred on success; destroy only on failure */
        rd_kafka_conf_destroy(conf);
        free(kctx);
        return NULL;
    }
    /* conf ownership transferred to rk — do not destroy separately */

    /* ── Create topic handle ── */
    rd_kafka_topic_conf_t *tconf = rd_kafka_topic_conf_new();
    kctx->rkt = rd_kafka_topic_new(kctx->rk, cfg->topic, tconf);
    if (!kctx->rkt) {
        snprintf(errbuf, errbuflen, "kafka_exporter: rd_kafka_topic_new('%s') failed: %s",
                 cfg->topic, rd_kafka_err2str(rd_kafka_last_error()));
        rd_kafka_destroy(kctx->rk);
        free(kctx);
        return NULL;
    }
    /* tconf ownership transferred to rkt */

    fprintf(stderr, "[kafka_exporter] connected → %s / topic=%s server_id=%s\n",
            cfg->bootstrap_servers, cfg->topic, kctx->server_id);
    return kctx;
}

/* =========================================================================
 * kafka_exporter_write  (ExportFn-compatible)
 * ========================================================================= */
void kafka_exporter_write(const FlowRecord *record,
                          const FlowKey    *key,
                          void             *ctx) {
    KafkaExporterCtx *kctx = (KafkaExporterCtx *)ctx;
    if (!kctx || !record) return;

    /* ── Step 1: Serialize flow to JSON using existing exporter ── */
    char *base_json = json_exporter_serialize(record, key);
    if (!base_json) {
        LOG_ERROR("kafka_exporter: json_exporter_serialize returned NULL");
        atomic_fetch_add(&kctx->errors, 1);
        return;
    }

    /* ── Step 2: Parse back into cJSON so we can inject extra fields ── */
    cJSON *root = cJSON_Parse(base_json);
    free(base_json);

    if (!root) {
        LOG_ERROR("kafka_exporter: cJSON_Parse failed");
        atomic_fetch_add(&kctx->errors, 1);
        return;
    }

    /* ── Step 3: Inject Kafka envelope fields ── */
    cJSON_AddStringToObject(root, "server_id",  kctx->server_id);
    cJSON_AddStringToObject(root, "geo_region", kctx->geo_region);

    /* Capture timestamp in microseconds since epoch */
    struct timeval now;
    gettimeofday(&now, NULL);
    double captured_us = (double)now.tv_sec * 1e6 + (double)now.tv_usec;
    cJSON_AddNumberToObject(root, "captured_at_us", captured_us);

    /* ── Step 4: Serialise final payload ── */
    char *payload = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (!payload) {
        LOG_ERROR("kafka_exporter: cJSON_PrintUnformatted returned NULL");
        atomic_fetch_add(&kctx->errors, 1);
        return;
    }

    size_t payload_len = strlen(payload);

    /* ── Step 5: Produce to Kafka, key = server_id ── */
    int rc = rd_kafka_produce(
        kctx->rkt,
        RD_KAFKA_PARTITION_UA,          /* let Kafka assign partition based on key */
        RD_KAFKA_MSG_F_COPY,            /* copy payload — we free it below */
        payload, payload_len,           /* value */
        kctx->server_id,               /* key — ensures same server → same partition */
        strlen(kctx->server_id),
        NULL                            /* per-message opaque */
    );

    free(payload);

    if (rc == -1) {
        LOG_ERROR("kafka_exporter: rd_kafka_produce failed: %s",
                  rd_kafka_err2str(rd_kafka_last_error()));
        atomic_fetch_add(&kctx->errors, 1);

        /* If queue is full, poll to drain then retry once */
        if (rd_kafka_last_error() == RD_KAFKA_RESP_ERR__QUEUE_FULL) {
            rd_kafka_poll(kctx->rk, 100 /* ms */);
        }
    }

    /* Poll to serve delivery reports without blocking */
    rd_kafka_poll(kctx->rk, 0 /* non-blocking */);
}

/* =========================================================================
 * kafka_exporter_destroy
 * ========================================================================= */
void kafka_exporter_destroy(KafkaExporterCtx *ctx) {
    if (!ctx) return;

    fprintf(stderr, "[kafka_exporter] flushing… (up to 10s)\n");

    /* Flush remaining messages */
    rd_kafka_flush(ctx->rk, 10 * 1000 /* ms */);

    fprintf(stderr,
            "[kafka_exporter] summary: produced=%" PRIu64 "  errors=%" PRIu64 "\n",
            atomic_load(&ctx->produced),
            atomic_load(&ctx->errors));

    rd_kafka_topic_destroy(ctx->rkt);
    rd_kafka_destroy(ctx->rk);
    free(ctx);
}

/* =========================================================================
 * Accessors
 * ========================================================================= */
uint64_t kafka_exporter_produced(const KafkaExporterCtx *ctx) {
    return ctx ? (uint64_t)atomic_load(&ctx->produced) : 0;
}

uint64_t kafka_exporter_errors(const KafkaExporterCtx *ctx) {
    return ctx ? (uint64_t)atomic_load(&ctx->errors) : 0;
}
