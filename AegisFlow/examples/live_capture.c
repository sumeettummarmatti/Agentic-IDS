/**
 * @file live_capture.c
 * @brief AegisFlow live interface capture example.
 *
 * Demonstrates attaching AegisFlow to a live network interface,
 * exporting flows as JSON to stdout, and printing a summary on exit.
 *
 * Usage (requires root / CAP_NET_RAW):
 *   sudo ./live_capture <interface> [bpf_filter]
 *
 * Example:
 *   sudo ./live_capture eth0 "tcp port 443"
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#include "aegisflow.h"

#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

static AegisContext *g_ctx = NULL;

static void sig_handler(int sig) {
    (void)sig;
    fprintf(stderr, "\nSIGINT received — stopping capture...\n");
    if (g_ctx) aegisflow_stop(g_ctx);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <interface> [bpf_filter]\n", argv[0]);
        fprintf(stderr, "Example: sudo %s eth0 \"tcp port 443\"\n\n", argv[0]);
        fprintf(stderr, "Available interfaces:\n");
        capture_list_devices();
        return 1;
    }

    const char *device = argv[1];
    const char *filter = (argc >= 3) ? argv[2] : "ip";

    fprintf(stderr,
        "\n┌──────────────────────────────────────────┐\n"
        "│  AegisFlow — Live Capture Demo            │\n"
        "│  Interface : %-28s│\n"
        "│  Filter    : %-28s│\n"
        "└──────────────────────────────────────────┘\n\n",
        device, filter);

    /* Set up signal handlers */
    signal(SIGINT,  sig_handler);
    signal(SIGTERM, sig_handler);

    /* Configure AegisFlow */
    AegisConfig cfg = aegisflow_default_config();
    cfg.device          = device;
    cfg.bpf_filter      = filter;
    cfg.promisc         = 1;
    cfg.tcp_timeout_sec = 120;
    cfg.udp_timeout_sec = 60;
    cfg.output_json     = stdout;   /* JSON → stdout */
    cfg.output_csv      = NULL;
    cfg.log_level       = LOG_LEVEL_INFO;

    char errbuf[256];
    g_ctx = aegisflow_create(&cfg, errbuf, sizeof(errbuf));
    if (!g_ctx) {
        fprintf(stderr, "Error: %s\n", errbuf);
        return 1;
    }

    fprintf(stderr, "Capturing on %s — press Ctrl+C to stop\n\n", device);
    int rc = aegisflow_run(g_ctx);

    /* Print final statistics */
    CaptureStats   cap;
    FlowTableStats tbl;
    aegisflow_get_stats(g_ctx, &cap, &tbl);

    fprintf(stderr,
        "\n┌──────────────── Summary ────────────────┐\n"
        "│  Packets received : %-20" PRIu64 "│\n"
        "│  Packets parsed   : %-20" PRIu64 "│\n"
        "│  Packets dropped  : %-20" PRIu64 "│\n"
        "│  Flows created    : %-20" PRIu64 "│\n"
        "│  Flows closed     : %-20" PRIu64 "│\n"
        "│  Flows expired    : %-20" PRIu64 "│\n"
        "└─────────────────────────────────────────┘\n\n",
        cap.pkts_received, cap.pkts_parsed, cap.pkts_dropped,
        tbl.flows_created, tbl.flows_closed, tbl.flows_expired);

    aegisflow_destroy(g_ctx);
    return (rc == 0) ? 0 : 1;
}
