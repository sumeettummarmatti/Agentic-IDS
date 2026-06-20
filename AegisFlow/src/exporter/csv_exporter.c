/**
 * @file csv_exporter.c
 * @brief CSV flow exporter for AegisFlow.
 *
 * Produces CICFlowMeter-compatible CSV output: one header line followed
 * by one data line per completed flow.  Column order matches CICFlowMeter's
 * default output for direct compatibility with ML pipelines trained on
 * CICIDS datasets.
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#include "exporter.h"
#include "aegis_features.h"
#include "utils.h"

#include <arpa/inet.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* =========================================================================
 * CSV column header (CICFlowMeter-compatible ordering)
 * ========================================================================= */
static const char *CSV_HEADER =
    "src_ip,dst_ip,src_port,dst_port,protocol,"
    "flow_duration,"
    "total_fwd_packets,total_bwd_packets,total_packets,"
    "total_length_fwd_pkts,total_length_bwd_pkts,"
    "pkt_length_min,pkt_length_max,pkt_length_mean,pkt_length_std,"
    "flow_bytes_per_sec,flow_pkts_per_sec,"
    "syn_flag_count,ack_flag_count,rst_flag_count,"
    "fin_flag_count,psh_flag_count,urg_flag_count,"
    "flow_iat_mean,flow_iat_std,flow_iat_min,flow_iat_max,"
    "fwd_pkt_length_mean,fwd_pkt_length_std,fwd_pkt_length_min,fwd_pkt_length_max,"
    "bwd_pkt_length_mean,bwd_pkt_length_std,bwd_pkt_length_min,bwd_pkt_length_max,"
    "fwd_iat_mean,fwd_iat_std,fwd_iat_min,fwd_iat_max,"
    "bwd_iat_mean,bwd_iat_std,bwd_iat_min,bwd_iat_max";

/* =========================================================================
 * Helper: safely print a double (emit 0 for inf/nan)
 * ========================================================================= */
static inline void fprint_double(FILE *fp, double v) {
    if (!isfinite(v)) v = 0.0;
    fprintf(fp, "%.6f", v);
}

/* =========================================================================
 * Public API
 * ========================================================================= */

void csv_exporter_write_header(FILE *fp) {
    if (!fp) fp = stdout;
    fputs(CSV_HEADER, fp);
    fputc('\n', fp);
    fflush(fp);
}

void csv_exporter_write(const FlowRecord *record,
                        const FlowKey    *key,
                        void             *ctx) {
    FILE *fp = ctx ? (FILE *)ctx : stdout;
    UNUSED(key);

    /* ── Extract feature vector ── */
    FlowFeatures f;
    features_extract(record, &f);

    /* ── Format IP addresses ── */
    char src_ip_str[INET_ADDRSTRLEN];
    char dst_ip_str[INET_ADDRSTRLEN];
    utils_ipv4_to_str(f.src_ip, src_ip_str, sizeof(src_ip_str));
    utils_ipv4_to_str(f.dst_ip, dst_ip_str, sizeof(dst_ip_str));

    /* ── Write CSV row ── */
    /* Identity */
    fprintf(fp, "%s,%s,%u,%u,%u,",
            src_ip_str, dst_ip_str,
            (unsigned)ntohs(f.src_port),
            (unsigned)ntohs(f.dst_port),
            (unsigned)f.protocol);

    /* Flow duration */
    fprint_double(fp, f.flow_duration_us); fputc(',', fp);

    /* Packet counts */
    fprintf(fp, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",",
            f.total_fwd_packets, f.total_bwd_packets, f.total_packets);

    /* Byte counts */
    fprintf(fp, "%" PRIu64 ",%" PRIu64 ",",
            f.total_len_fwd_pkts, f.total_len_bwd_pkts);

    /* Packet length stats */
    fprint_double(fp, f.pkt_len_min);  fputc(',', fp);
    fprint_double(fp, f.pkt_len_max);  fputc(',', fp);
    fprint_double(fp, f.pkt_len_mean); fputc(',', fp);
    fprint_double(fp, f.pkt_len_std);  fputc(',', fp);

    /* Rate features */
    fprint_double(fp, f.flow_bytes_per_sec); fputc(',', fp);
    fprint_double(fp, f.flow_pkts_per_sec);  fputc(',', fp);

    /* TCP flags */
    fprintf(fp, "%u,%u,%u,%u,%u,%u,",
            f.syn_count, f.ack_count, f.rst_count,
            f.fin_count, f.psh_count, f.urg_count);

    /* Flow IAT */
    fprint_double(fp, f.flow_iat_mean); fputc(',', fp);
    fprint_double(fp, f.flow_iat_std);  fputc(',', fp);
    fprint_double(fp, f.flow_iat_min);  fputc(',', fp);
    fprint_double(fp, f.flow_iat_max);  fputc(',', fp);

    /* Forward packet length */
    fprint_double(fp, f.fwd_pkt_len_mean); fputc(',', fp);
    fprint_double(fp, f.fwd_pkt_len_std);  fputc(',', fp);
    fprint_double(fp, f.fwd_pkt_len_min);  fputc(',', fp);
    fprint_double(fp, f.fwd_pkt_len_max);  fputc(',', fp);

    /* Backward packet length */
    fprint_double(fp, f.bwd_pkt_len_mean); fputc(',', fp);
    fprint_double(fp, f.bwd_pkt_len_std);  fputc(',', fp);
    fprint_double(fp, f.bwd_pkt_len_min);  fputc(',', fp);
    fprint_double(fp, f.bwd_pkt_len_max);  fputc(',', fp);

    /* Forward IAT */
    fprint_double(fp, f.fwd_iat_mean); fputc(',', fp);
    fprint_double(fp, f.fwd_iat_std);  fputc(',', fp);
    fprint_double(fp, f.fwd_iat_min);  fputc(',', fp);
    fprint_double(fp, f.fwd_iat_max);  fputc(',', fp);

    /* Backward IAT (last column — no trailing comma) */
    fprint_double(fp, f.bwd_iat_mean); fputc(',', fp);
    fprint_double(fp, f.bwd_iat_std);  fputc(',', fp);
    fprint_double(fp, f.bwd_iat_min);  fputc(',', fp);
    fprint_double(fp, f.bwd_iat_max);

    fputc('\n', fp);
    fflush(fp);
}

/* =========================================================================
 * ExporterChain implementation
 * ========================================================================= */

void exporter_chain_init(ExporterChain *chain) {
    chain->count = 0;
    for (int i = 0; i < AEGIS_MAX_EXPORTERS; i++) {
        chain->fns[i]  = NULL;
        chain->ctxs[i] = NULL;
    }
}

int exporter_chain_add(ExporterChain *chain, ExportFn fn, void *ctx) {
    if (chain->count >= AEGIS_MAX_EXPORTERS) {
        LOG_WARN("ExporterChain full (max=%d)", AEGIS_MAX_EXPORTERS);
        return -1;
    }
    chain->fns[chain->count]  = fn;
    chain->ctxs[chain->count] = ctx;
    chain->count++;
    return 0;
}

void exporter_chain_dispatch(const FlowRecord *record,
                             const FlowKey    *key,
                             void             *ctx) {
    ExporterChain *chain = (ExporterChain *)ctx;
    for (int i = 0; i < chain->count; i++) {
        if (chain->fns[i]) {
            chain->fns[i](record, key, chain->ctxs[i]);
        }
    }
}
