/**
 * @file kafka_main.c
 * @brief AegisFlow CLI that publishes closed flows to Kafka instead of files.
 *
 * Drop-in replacement for main.c.  Same flags as the standard binary, plus:
 *   -b <brokers>    Kafka bootstrap servers  (default: localhost:9092)
 *   -T <topic>      Kafka topic              (default: raw-flows)
 *   -s <server_id>  This node's ID           (default: $SERVER_ID env or "UNKNOWN")
 *   -g <geo_region> Geographic label         (default: $GEO_REGION env or "unknown")
 *   -V              Verbose Kafka delivery reports
 *
 * Reads $KAFKA_BOOTSTRAP_SERVERS, $KAFKA_RAW_FLOWS_TOPIC, $SERVER_ID,
 * $GEO_REGION from environment if not provided via flags.
 *
 * Build (handled by CMakeLists.txt with AEGISFLOW_KAFKA=ON):
 *   cmake -S . -B build -DAEGISFLOW_KAFKA=ON
 *   cmake --build build --target aegisflow_kafka
 *
 * Run (live capture):
 *   sudo ./build/aegisflow_kafka -i eth0 -b localhost:9092 -T raw-flows -s MUM-01
 *
 * Run (PCAP replay, no sudo needed):
 *   ./build/aegisflow_kafka -r capture.pcap -b localhost:9092 -T raw-flows -s MUM-01
 *
 * AegisFlow — Kafka Main Entry Point
 * SPDX-License-Identifier: MIT
 */

#include "aegisflow.h"
#include "kafka_exporter.h"

#include <getopt.h>
#include <unistd.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

/* ── Global context for signal handler ─────────────────────── */
static AegisContext    *g_aegis_ctx  = NULL;
static KafkaExporterCtx *g_kafka_ctx = NULL;

static void signal_handler(int sig) {
    (void)sig;
    if (g_aegis_ctx) aegisflow_stop(g_aegis_ctx);
}

/* ── Helper: read env var or return default ─────────────────── */
static const char *env_or(const char *var, const char *fallback) {
    const char *v = getenv(var);
    return (v && *v) ? v : fallback;
}

/* ── Usage ──────────────────────────────────────────────────── */
static void print_usage(const char *prog) {
    fprintf(stderr,
        "AegisFlow-Kafka — Packet Capture → Kafka Publisher\n\n"
        "Usage:\n"
        "  sudo %s -i <iface>    [options]   Live capture\n"
        "       %s -r <pcap>     [options]   Offline PCAP replay\n\n"
        "Capture flags (same as aegisflow):\n"
        "  -i <iface>    Network interface  (e.g. eth0)\n"
        "  -r <file>     Offline PCAP/PCAPNG\n"
        "  -f <bpf>      BPF filter string  (default: \"ip\")\n"
        "  -t <sec>      TCP flow timeout   (default: 120)\n"
        "  -u <sec>      UDP flow timeout   (default: 60)\n"
        "  -n <count>    Stop after N packets\n"
        "  -e <sec>      Flow expiry scan interval (default: 5)\n"
        "  -v            Verbose (DEBUG) logging\n"
        "  -q            Quiet (WARN only)\n"
        "  -l            List network interfaces\n\n"
        "Kafka flags:\n"
        "  -b <brokers>  Bootstrap servers  (default: $KAFKA_BOOTSTRAP_SERVERS or localhost:9092)\n"
        "  -T <topic>    Topic name         (default: $KAFKA_RAW_FLOWS_TOPIC or raw-flows)\n"
        "  -s <id>       Server/node ID     (default: $SERVER_ID or hostname)\n"
        "  -g <region>   Geo region label   (default: $GEO_REGION or unknown)\n"
        "  -V            Verbose Kafka delivery reports\n"
        "  -h            Show this help\n\n"
        "Environment variables (overridden by flags):\n"
        "  KAFKA_BOOTSTRAP_SERVERS, KAFKA_RAW_FLOWS_TOPIC, SERVER_ID, GEO_REGION\n",
        prog, prog
    );
}

/* ── main ───────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    /* ── AegisFlow defaults ── */
    AegisConfig cfg = aegisflow_default_config();

    /* ── Kafka defaults (from env) ── */
    const char *brokers   = env_or("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092");
    const char *topic     = env_or("KAFKA_RAW_FLOWS_TOPIC",   "raw-flows");
    const char *server_id = env_or("SERVER_ID",               NULL);
    const char *geo       = env_or("GEO_REGION",              "unknown");
    int kafka_verbose     = 0;

    /* Auto server_id from hostname if not set */
    static char hostname_buf[128] = {0};
    if (!server_id) {
        if (gethostname(hostname_buf, sizeof(hostname_buf)) == 0)
            server_id = hostname_buf;
        else
            server_id = "UNKNOWN";
    }

    /* ── Parse args ── */
    int opt;
    while ((opt = getopt(argc, argv, "i:r:f:t:u:n:e:b:T:s:g:vqVlh")) != -1) {
        switch (opt) {
        /* AegisFlow flags */
        case 'i': cfg.device              = optarg;                       break;
        case 'r': cfg.pcap_file           = optarg;                       break;
        case 'f': cfg.bpf_filter          = optarg;                       break;
        case 't': cfg.tcp_timeout_sec     = atoi(optarg);                 break;
        case 'u': cfg.udp_timeout_sec     = atoi(optarg);                 break;
        case 'n': cfg.max_packets         = (uint64_t)strtoull(optarg, NULL, 10); break;
        case 'e': cfg.expire_interval_sec = atoi(optarg);                 break;
        case 'v': cfg.log_level           = LOG_LEVEL_DEBUG;              break;
        case 'q': cfg.log_level           = LOG_LEVEL_WARN;               break;
        case 'l': capture_list_devices(); return 0;
        /* Kafka flags */
        case 'b': brokers    = optarg; break;
        case 'T': topic      = optarg; break;
        case 's': server_id  = optarg; break;
        case 'g': geo        = optarg; break;
        case 'V': kafka_verbose = 1;   break;
        case 'h': default:
            print_usage(argv[0]);
            return (opt == 'h') ? 0 : 1;
        }
    }

    /* ── Validate ── */
    if (!cfg.device && !cfg.pcap_file) {
        fprintf(stderr, "Error: specify -i <interface> or -r <pcap_file>\n\n");
        print_usage(argv[0]);
        return 1;
    }

    /* ── Print banner ── */
    fprintf(stderr,
        "\n╔══════════════════════════════════════════════════╗\n"
        "║  AegisFlow-Kafka v1.0  │  Packet Capture → Kafka ║\n"
        "╚══════════════════════════════════════════════════╝\n\n"
        "  Node      : %s  (%s)\n"
        "  Broker    : %s\n"
        "  Topic     : %s\n"
        "  Source    : %s\n\n",
        server_id, geo,
        brokers, topic,
        cfg.device ? cfg.device : cfg.pcap_file
    );

    /* ── Create Kafka exporter ── */
    KafkaExporterConfig kcfg = {
        .bootstrap_servers      = brokers,
        .topic                  = topic,
        .server_id              = server_id,
        .geo_region             = geo,
        .queue_buffering_max_ms = 5,
        .batch_num_messages     = 1000,
        .verbose                = kafka_verbose,
    };

    char errbuf[256];
    g_kafka_ctx = kafka_exporter_create(&kcfg, errbuf, sizeof(errbuf));
    if (!g_kafka_ctx) {
        fprintf(stderr, "Error: %s\n", errbuf);
        return 1;
    }

    /* ── Wire up the Kafka exporter ──
     * Every closed flow is passed to kafka_exporter_write() through the
     * engine's exporter chain. No file/stdout output is used in this binary. */
    cfg.output_json = NULL;
    cfg.output_csv  = NULL;
    cfg.extra_exporter     = kafka_exporter_write;
    cfg.extra_exporter_ctx = g_kafka_ctx;

    /* ── Signal handlers ── */
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    /* ── Create AegisFlow context ── */
    g_aegis_ctx = aegisflow_create(&cfg, errbuf, sizeof(errbuf));
    if (!g_aegis_ctx) {
        fprintf(stderr, "Error: aegisflow_create: %s\n", errbuf);
        kafka_exporter_destroy(g_kafka_ctx);
        return 1;
    }

    /* ── Run ── */
    time_t start = time(NULL);
    int rc = aegisflow_run(g_aegis_ctx);
    time_t elapsed = time(NULL) - start;

    /* ── Gather stats ── */
    CaptureStats   cap;
    FlowTableStats tbl;
    aegisflow_get_stats(g_aegis_ctx, &cap, &tbl);

    fprintf(stderr,
        "\n──────────── AegisFlow-Kafka Summary ────────────\n"
        "  Runtime         : %lds\n"
        "  Packets received: %" PRIu64 "\n"
        "  Packets parsed  : %" PRIu64 "\n"
        "  Flows created   : %" PRIu64 "\n"
        "  Flows exported  : %" PRIu64 "\n"
        "  Kafka produced  : %" PRIu64 "\n"
        "  Kafka errors    : %" PRIu64 "\n"
        "─────────────────────────────────────────────────\n\n",
        (long)elapsed,
        cap.pkts_received,
        cap.pkts_parsed,
        tbl.flows_created,
        tbl.flows_closed + tbl.flows_expired,
        kafka_exporter_produced(g_kafka_ctx),
        kafka_exporter_errors(g_kafka_ctx)
    );

    /* ── Cleanup ── */
    aegisflow_destroy(g_aegis_ctx);
    kafka_exporter_destroy(g_kafka_ctx);

    return (rc == 0) ? 0 : 1;
}
