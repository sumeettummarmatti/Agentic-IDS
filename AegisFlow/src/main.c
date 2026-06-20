/**
 * @file main.c
 * @brief AegisFlow command-line entry point.
 *
 * Provides a feature-complete CLI for both live interface capture and
 * offline PCAP replay, with JSON and/or CSV output.
 *
 * Usage:
 *   aegisflow -i <interface>  [options]
 *   aegisflow -r <pcap_file>  [options]
 *
 * Options:
 *   -i <iface>     Live capture interface (e.g. eth0)
 *   -r <file>      Read from offline PCAP file
 *   -f <bpf>       BPF filter string (default: "ip")
 *   -j <file>      Write JSON output to file (- = stdout)
 *   -c <file>      Write CSV output to file  (- = stdout)
 *   -t <sec>       TCP flow timeout in seconds (default: 120)
 *   -u <sec>       UDP flow timeout in seconds (default: 60)
 *   -n <count>     Stop after N packets (default: unlimited)
 *   -e <sec>       Flow expiry scan interval in seconds (default: 5)
 *   -v             Verbose (DEBUG) logging
 *   -q             Quiet (WARN only) logging
 *   -l             List available network interfaces
 *   -h             Show this help message
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#include "aegisflow.h"

#include <getopt.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

/* =========================================================================
 * Global context pointer (for signal handler)
 * ========================================================================= */
static AegisContext *g_ctx = NULL;

/* =========================================================================
 * Signal handler — graceful shutdown on SIGINT / SIGTERM
 * ========================================================================= */
static void signal_handler(int sig) {
    (void)sig;
    if (g_ctx) {
        aegisflow_stop(g_ctx);
    }
}

/* =========================================================================
 * Help text
 * ========================================================================= */
static void print_usage(const char *prog) {
    fprintf(stderr,
        "AegisFlow v" AEGISFLOW_VERSION_STR " — CICFlowMeter-Compatible Feature Engine\n\n"
        "Usage:\n"
        "  %s -i <interface>  [options]   Live capture\n"
        "  %s -r <pcap_file>  [options]   Offline PCAP replay\n\n"
        "Options:\n"
        "  -i <iface>   Live capture interface (e.g. eth0, wlan0)\n"
        "  -r <file>    Read from offline PCAP / PCAPNG file\n"
        "  -f <bpf>     BPF filter string (default: \"ip\")\n"
        "  -j <file>    Write JSON output to file ('-' = stdout)\n"
        "  -c <file>    Write CSV output to file  ('-' = stdout)\n"
        "  -t <sec>     TCP flow timeout (default: %d)\n"
        "  -u <sec>     UDP flow timeout (default: %d)\n"
        "  -n <count>   Stop after N packets (default: unlimited)\n"
        "  -e <sec>     Expiry scan interval  (default: 5)\n"
        "  -v           Verbose / DEBUG logging\n"
        "  -q           Quiet / WARN-only logging\n"
        "  -l           List available network interfaces\n"
        "  -h           Show this help\n\n"
        "Examples:\n"
        "  sudo %s -i eth0 -j - -c flows.csv\n"
        "  %s -r capture.pcap -c output.csv\n",
        prog, prog, FLOW_TIMEOUT_TCP_SEC, FLOW_TIMEOUT_UDP_SEC, prog, prog
    );
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(int argc, char *argv[]) {
    /* ── Defaults ── */
    AegisConfig cfg = aegisflow_default_config();

    const char *json_path = NULL;
    const char *csv_path  = NULL;

    FILE *json_fp = NULL;
    FILE *csv_fp  = NULL;

    /* ── Parse arguments ── */
    int opt;
    while ((opt = getopt(argc, argv, "i:r:f:j:c:t:u:n:e:vqlh")) != -1) {
        switch (opt) {
        case 'i': cfg.device         = optarg;             break;
        case 'r': cfg.pcap_file      = optarg;             break;
        case 'f': cfg.bpf_filter     = optarg;             break;
        case 'j': json_path          = optarg;             break;
        case 'c': csv_path           = optarg;             break;
        case 't': cfg.tcp_timeout_sec = atoi(optarg);     break;
        case 'u': cfg.udp_timeout_sec = atoi(optarg);     break;
        case 'n': cfg.max_packets    = (uint64_t)strtoull(optarg, NULL, 10); break;
        case 'e': cfg.expire_interval_sec = atoi(optarg); break;
        case 'v': cfg.log_level      = LOG_LEVEL_DEBUG;   break;
        case 'q': cfg.log_level      = LOG_LEVEL_WARN;    break;
        case 'l':
            capture_list_devices();
            return 0;
        case 'h':
        default:
            print_usage(argv[0]);
            return (opt == 'h') ? 0 : 1;
        }
    }

    /* ── Validate required arguments ── */
    if (!cfg.device && !cfg.pcap_file) {
        fprintf(stderr, "Error: specify either -i <interface> or -r <pcap_file>\n\n");
        print_usage(argv[0]);
        return 1;
    }

    /* ── Open output files ── */
    if (json_path) {
        json_fp = (strcmp(json_path, "-") == 0) ? stdout : fopen(json_path, "w");
        if (!json_fp) {
            fprintf(stderr, "Error: cannot open JSON output '%s'\n", json_path);
            return 1;
        }
        cfg.output_json = json_fp;
        LOG_INFO("JSON output → %s", json_path);
    }

    if (csv_path) {
        csv_fp = (strcmp(csv_path, "-") == 0) ? stdout : fopen(csv_path, "w");
        if (!csv_fp) {
            fprintf(stderr, "Error: cannot open CSV output '%s'\n", csv_path);
            if (json_fp && json_fp != stdout) fclose(json_fp);
            return 1;
        }
        cfg.output_csv = csv_fp;
        LOG_INFO("CSV output → %s", csv_path);
    }

    if (!cfg.output_json && !cfg.output_csv) {
        /* Default: JSON to stdout */
        cfg.output_json = stdout;
        LOG_INFO("No output specified — defaulting to JSON on stdout");
    }

    /* ── Set up signal handlers ── */
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    /* ── Create AegisFlow context ── */
    char errbuf[256];
    g_ctx = aegisflow_create(&cfg, errbuf, sizeof(errbuf));
    if (!g_ctx) {
        fprintf(stderr, "Error: aegisflow_create failed: %s\n", errbuf);
        if (json_fp && json_fp != stdout) fclose(json_fp);
        if (csv_fp  && csv_fp  != stdout) fclose(csv_fp);
        return 1;
    }

    /* ── Print startup banner ── */
    fprintf(stderr,
        "\n╔══════════════════════════════════════════╗\n"
        "║  AegisFlow v%-30s║\n"
        "║  CICFlowMeter-Compatible Feature Engine  ║\n"
        "╚══════════════════════════════════════════╝\n\n",
        AEGISFLOW_VERSION_STR " ");

    if (cfg.device)    fprintf(stderr, "  Interface : %s\n", cfg.device);
    if (cfg.pcap_file) fprintf(stderr, "  PCAP file : %s\n", cfg.pcap_file);
    if (cfg.bpf_filter) fprintf(stderr, "  BPF filter: %s\n", cfg.bpf_filter);
    fprintf(stderr, "  TCP timeout: %ds   UDP timeout: %ds\n\n",
            cfg.tcp_timeout_sec, cfg.udp_timeout_sec);

    /* ── Run ── */
    time_t start = time(NULL);
    int rc = aegisflow_run(g_ctx);
    time_t elapsed = time(NULL) - start;

    /* ── Print summary ── */
    CaptureStats   cap_stats;
    FlowTableStats tbl_stats;
    aegisflow_get_stats(g_ctx, &cap_stats, &tbl_stats);

    fprintf(stderr,
        "\n──────────── AegisFlow Summary ────────────\n"
        "  Runtime         : %lds\n"
        "  Packets received: %" PRIu64 "\n"
        "  Packets parsed  : %" PRIu64 "\n"
        "  Packets dropped : %" PRIu64 "\n"
        "  Flows created   : %" PRIu64 "\n"
        "  Flows closed    : %" PRIu64 "\n"
        "  Flows expired   : %" PRIu64 "\n"
        "──────────────────────────────────────────\n\n",
        (long)elapsed,
        cap_stats.pkts_received,
        cap_stats.pkts_parsed,
        cap_stats.pkts_dropped,
        tbl_stats.flows_created,
        tbl_stats.flows_closed,
        tbl_stats.flows_expired);

    /* ── Cleanup ── */
    aegisflow_destroy(g_ctx);
    g_ctx = NULL;

    if (json_fp && json_fp != stdout) fclose(json_fp);
    if (csv_fp  && csv_fp  != stdout) fclose(csv_fp);

    return (rc == 0) ? 0 : 1;
}
