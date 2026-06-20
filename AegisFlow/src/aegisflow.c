/**
 * @file aegisflow.c
 * @brief Top-level AegisFlow engine — wires capture, flow table, exporters.
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#include "aegisflow.h"
#include "utils.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

/* =========================================================================
 * AegisContext — internal state
 * ========================================================================= */
struct AegisContext {
    AegisConfig     cfg;
    CaptureHandle  *capture;
    FlowTable      *flow_table;
    ExporterChain   exporter_chain;

    /* Output files (may be NULL) */
    FILE           *json_fp;
    FILE           *csv_fp;

    /* Last expiry scan time */
    struct timeval  last_expire;
};

/* =========================================================================
 * Per-packet callback: dispatches into the flow table
 * ========================================================================= */
static void on_packet(const PacketInfo *pkt, void *user) {
    AegisContext *ctx = (AegisContext *)user;

    flow_table_update(ctx->flow_table, pkt);

    /* Periodic expiry scan */
    struct timeval now;
    utils_now(&now);
    int64_t since_us = timeval_diff_usec(&ctx->last_expire, &now);
    int64_t interval_us = (int64_t)ctx->cfg.expire_interval_sec * 1000000LL;

    if (since_us >= interval_us) {
        uint32_t expired = flow_table_expire(ctx->flow_table, &now);
        if (expired > 0) {
            LOG_DEBUG("Expired %u timed-out flows", expired);
        }
        ctx->last_expire = now;
    }
}

/* =========================================================================
 * Public API
 * ========================================================================= */

AegisConfig aegisflow_default_config(void) {
    AegisConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.bpf_filter         = "ip";
    cfg.promisc            = 1;
    cfg.tcp_timeout_sec    = FLOW_TIMEOUT_TCP_SEC;
    cfg.udp_timeout_sec    = FLOW_TIMEOUT_UDP_SEC;
    cfg.max_flows          = 0;
    cfg.thread_safe        = 0;
    cfg.csv_header         = 1;
    cfg.max_packets        = 0;
    cfg.expire_interval_sec = 5;
    cfg.log_level          = LOG_LEVEL_INFO;
    return cfg;
}

AegisContext *aegisflow_create(const AegisConfig *cfg,
                               char *errbuf, size_t errbuflen) {
    utils_set_log_level(cfg->log_level);

    AegisContext *ctx = xcalloc(1, sizeof(AegisContext));
    ctx->cfg = *cfg;

    /* ── Set up exporters ── */
    exporter_chain_init(&ctx->exporter_chain);

    if (cfg->output_json) {
        ctx->json_fp = cfg->output_json;
        exporter_chain_add(&ctx->exporter_chain, json_exporter_write, ctx->json_fp);
        LOG_INFO("JSON exporter enabled");
    }

    if (cfg->output_csv) {
        ctx->csv_fp = cfg->output_csv;
        if (cfg->csv_header) {
            csv_exporter_write_header(ctx->csv_fp);
        }
        exporter_chain_add(&ctx->exporter_chain, csv_exporter_write, ctx->csv_fp);
        LOG_INFO("CSV exporter enabled");
    }

    /* ── Create flow table ── */
    FlowTableConfig tbl_cfg = {
        .tcp_timeout_sec    = cfg->tcp_timeout_sec,
        .udp_timeout_sec    = cfg->udp_timeout_sec,
        .max_flows          = cfg->max_flows,
        .thread_safe        = cfg->thread_safe,
        .on_flow_close      = exporter_chain_dispatch,
        .on_flow_close_ctx  = &ctx->exporter_chain
    };
    ctx->flow_table = flow_table_create(&tbl_cfg);
    if (!ctx->flow_table) {
        snprintf(errbuf, errbuflen, "Failed to create flow table (OOM)");
        free(ctx);
        return NULL;
    }

    /* ── Open capture ── */
    CaptureConfig cap_cfg = {
        .device     = cfg->device,
        .pcap_file  = cfg->pcap_file,
        .bpf_filter = cfg->bpf_filter,
        .snaplen    = 65535,
        .promisc    = cfg->promisc,
        .timeout_ms = 1000,
        .on_packet  = on_packet,
        .user       = ctx
    };
    ctx->capture = capture_open(&cap_cfg, errbuf, errbuflen);
    if (!ctx->capture) {
        flow_table_destroy(ctx->flow_table);
        free(ctx);
        return NULL;
    }

    utils_now(&ctx->last_expire);

    LOG_INFO("AegisFlow v%s context created", AEGISFLOW_VERSION_STR);
    return ctx;
}

int aegisflow_run(AegisContext *ctx) {
    if (!ctx) return -1;
    return capture_run(ctx->capture, ctx->cfg.max_packets);
}

void aegisflow_stop(AegisContext *ctx) {
    if (ctx) capture_stop(ctx->capture);
}

void aegisflow_get_stats(const AegisContext *ctx,
                         CaptureStats     *cap,
                         FlowTableStats   *table) {
    if (cap)   capture_get_stats(ctx->capture, cap);
    if (table) flow_table_get_stats(ctx->flow_table, table);
}

void aegisflow_destroy(AegisContext *ctx) {
    if (!ctx) return;

    /* Flush remaining active flows before closing */
    uint32_t remaining = flow_table_flush(ctx->flow_table);
    LOG_INFO("Flushed %u remaining flows at shutdown", remaining);

    capture_close(ctx->capture);
    flow_table_destroy(ctx->flow_table);
    free(ctx);

    LOG_INFO("AegisFlow context destroyed");
}
