/**
 * @file pcap_replay.c
 * @brief AegisFlow offline PCAP replay example.
 *
 * Reads an existing .pcap or .pcapng file and exports all completed flows
 * to both JSON (stdout) and CSV (flows.csv).  Useful for:
 *   - Validating AegisFlow output against CICFlowMeter reference
 *   - Offline analysis of captured traffic
 *   - Generating feature datasets for ML training
 *
 * Usage:
 *   ./pcap_replay <pcap_file> [output.csv] [bpf_filter]
 *
 * Example:
 *   ./pcap_replay capture.pcap output.csv "tcp"
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#include "aegisflow.h"

#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <pcap_file> [output.csv] [bpf_filter]\n"
            "Example: %s capture.pcap flows.csv \"tcp\"\n", argv[0], argv[0]);
        return 1;
    }

    const char *pcap_file  = argv[1];
    const char *csv_file   = (argc >= 3) ? argv[2] : "flows.csv";
    const char *bpf_filter = (argc >= 4) ? argv[3] : "ip";

    fprintf(stderr,
        "\n┌──────────────────────────────────────────┐\n"
        "│  AegisFlow — PCAP Replay                  │\n"
        "│  Input  : %-30s│\n"
        "│  CSV    : %-30s│\n"
        "│  Filter : %-30s│\n"
        "└──────────────────────────────────────────┘\n\n",
        pcap_file, csv_file, bpf_filter);

    /* ── Open CSV output file ── */
    FILE *csv_fp = fopen(csv_file, "w");
    if (!csv_fp) {
        fprintf(stderr, "Error: cannot open CSV output '%s'\n", csv_file);
        return 1;
    }

    /* ── Configure AegisFlow ── */
    AegisConfig cfg = aegisflow_default_config();
    cfg.pcap_file       = pcap_file;
    cfg.bpf_filter      = bpf_filter;
    cfg.output_json     = NULL;        /* suppress JSON for clean CSV output */
    cfg.output_csv      = csv_fp;
    cfg.csv_header      = 1;
    cfg.tcp_timeout_sec = 120;
    cfg.udp_timeout_sec = 60;
    cfg.log_level       = LOG_LEVEL_INFO;

    char errbuf[256];
    AegisContext *ctx = aegisflow_create(&cfg, errbuf, sizeof(errbuf));
    if (!ctx) {
        fprintf(stderr, "Error: %s\n", errbuf);
        fclose(csv_fp);
        return 1;
    }

    /* ── Replay ── */
    time_t t_start = time(NULL);
    int rc = aegisflow_run(ctx);
    time_t elapsed = time(NULL) - t_start;

    /* ── Statistics ── */
    CaptureStats   cap;
    FlowTableStats tbl;
    aegisflow_get_stats(ctx, &cap, &tbl);

    fprintf(stderr,
        "\n┌──────────────── Replay Results ─────────────┐\n"
        "│  Elapsed           : %lds\n"
        "│  Packets received  : %" PRIu64 "\n"
        "│  Packets parsed    : %" PRIu64 "\n"
        "│  Packets skipped   : %" PRIu64 "\n"
        "│  Flows created     : %" PRIu64 "\n"
        "│  Flows closed/FIN  : %" PRIu64 "\n"
        "│  Flows expired     : %" PRIu64 "\n"
        "│  Output            : %s\n"
        "└─────────────────────────────────────────────┘\n\n",
        (long)elapsed,
        cap.pkts_received, cap.pkts_parsed, cap.pkts_skipped,
        tbl.flows_created, tbl.flows_closed, tbl.flows_expired,
        csv_file);

    aegisflow_destroy(ctx);
    fclose(csv_fp);

    return (rc == 0) ? 0 : 1;
}
