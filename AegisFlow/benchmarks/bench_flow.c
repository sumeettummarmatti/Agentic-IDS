/**
 * @file bench_flow.c
 * @brief AegisFlow performance benchmark utility.
 *
 * Generates a synthetic stream of PacketInfo structs and measures:
 *   - Throughput (packets/second) through the flow table + feature engine
 *   - Memory footprint per active flow
 *   - Feature extraction latency per flow closure
 *
 * Usage:
 *   ./bench_flow [num_packets] [num_flows] [--no-export]
 *
 * Default: 1,000,000 packets across 10,000 distinct flows.
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#include "aegisflow.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>
#include <arpa/inet.h>

/* =========================================================================
 * Clock helper — nanosecond resolution via CLOCK_MONOTONIC
 * ========================================================================= */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* =========================================================================
 * Null exporter: discard output (benchmark throughput without I/O overhead)
 * ========================================================================= */
static uint64_t g_exported = 0;

static void null_exporter(const FlowRecord *r, const FlowKey *k, void *ctx) {
    (void)r; (void)k; (void)ctx;
    g_exported++;
}

/* =========================================================================
 * Build a synthetic PacketInfo for flow index `flow_idx`
 * ========================================================================= */
static PacketInfo make_synthetic_packet(uint32_t flow_idx,
                                        uint64_t pkt_seq,
                                        uint32_t num_flows) {
    PacketInfo p;
    memset(&p, 0, sizeof(p));

    /* Assign unique 5-tuple per flow */
    p.key.src_ip   = htonl(0x0A000001 + (flow_idx % num_flows));
    p.key.dst_ip   = htonl(0x08080808);
    p.key.src_port = htons((uint16_t)(40000 + (flow_idx % num_flows)));
    p.key.dst_port = htons(443);
    p.key.protocol = PROTO_TCP;

    /* Vary payload length 40–1460 bytes */
    p.payload_len = (uint32_t)(40 + (pkt_seq * 7 + flow_idx * 13) % 1420);

    /* Synthetic timestamps: 1µs per packet */
    uint64_t ts_us = pkt_seq;
    p.timestamp.tv_sec  = (time_t)(ts_us / 1000000);
    p.timestamp.tv_usec = (suseconds_t)(ts_us % 1000000);

    /* Mostly ACK; send a FIN every 100 packets to close flows */
    if (pkt_seq % 100 == 99) {
        p.tcp_flags = TCP_FLAG_FIN | TCP_FLAG_ACK;
    } else {
        p.tcp_flags = TCP_FLAG_ACK;
    }
    p.is_fwd = 1;

    return p;
}

/* =========================================================================
 * Main benchmark
 * ========================================================================= */
int main(int argc, char *argv[]) {
    uint64_t num_packets = 1000000ULL;
    uint32_t num_flows   = 10000;
    int      do_export   = 1;

    /* Parse CLI args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-export") == 0) {
            do_export = 0;
        } else if (i == 1) {
            num_packets = strtoull(argv[i], NULL, 10);
        } else if (i == 2) {
            num_flows = (uint32_t)atoi(argv[i]);
        }
    }

    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║  AegisFlow Performance Benchmark             ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");
    printf("  Packets   : %" PRIu64 "\n", num_packets);
    printf("  Flows     : %u (synthetic)\n", num_flows);
    printf("  Export    : %s\n\n", do_export ? "null (no I/O)" : "disabled");

    /* ── Create flow table ── */
    FlowTableConfig cfg = {
        .tcp_timeout_sec   = 120,
        .udp_timeout_sec   = 60,
        .max_flows         = 0,
        .thread_safe       = 0,
        .on_flow_close     = do_export ? null_exporter : NULL,
        .on_flow_close_ctx = NULL
    };
    FlowTable *ft = flow_table_create(&cfg);
    if (!ft) {
        fprintf(stderr, "Failed to create flow table\n");
        return 1;
    }

    /* ── Warmup: 10,000 packets ── */
    printf("  Warming up (10k packets)...\n");
    for (uint64_t i = 0; i < 10000; i++) {
        PacketInfo p = make_synthetic_packet((uint32_t)(i % num_flows), i, num_flows);
        flow_table_update(ft, &p);
    }
    FlowTableStats dummy;
    flow_table_flush(ft);
    flow_table_get_stats(ft, &dummy);
    g_exported = 0;
    (void)dummy;

    /* ── Main benchmark ── */
    printf("  Running benchmark...\n\n");
    double t_start = now_sec();

    for (uint64_t i = 0; i < num_packets; i++) {
        PacketInfo p = make_synthetic_packet((uint32_t)(i % num_flows), i, num_flows);
        flow_table_update(ft, &p);
    }

    double t_end = now_sec();
    double elapsed = t_end - t_start;

    /* ── Flush remaining ── */
    flow_table_flush(ft);

    /* ── Results ── */
    FlowTableStats stats;
    flow_table_get_stats(ft, &stats);

    double pkt_per_sec  = (double)num_packets / elapsed;
    double ns_per_pkt   = elapsed * 1e9 / (double)num_packets;
    double flow_mem_kb  = (double)(stats.flows_created * sizeof(FlowRecord)) / 1024.0;

    printf("──────────────── Benchmark Results ───────────────\n");
    printf("  Elapsed time        : %.3f seconds\n", elapsed);
    printf("  Throughput          : %.0f packets/sec\n",   pkt_per_sec);
    printf("  Latency (avg)       : %.1f ns/packet\n",     ns_per_pkt);
    printf("  Flows created       : %" PRIu64 "\n",        stats.flows_created);
    printf("  Flows closed (FIN)  : %" PRIu64 "\n",        stats.flows_closed);
    printf("  Flows expired       : %" PRIu64 "\n",        stats.flows_expired);
    printf("  Flows exported      : %" PRIu64 "\n",        g_exported);
    printf("  FlowRecord size     : %zu bytes\n",          sizeof(FlowRecord));
    printf("  Est. flow mem (peak): %.1f KB\n",            flow_mem_kb);
    printf("──────────────────────────────────────────────────\n");

    if (pkt_per_sec >= 100000.0) {
        printf("\n  ✓ Target achieved: >100,000 packets/sec\n");
    } else {
        printf("\n  ✗ Below target: %.0f < 100,000 packets/sec\n", pkt_per_sec);
    }
    printf("\n");

    flow_table_destroy(ft);
    return 0;
}
