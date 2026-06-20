/**
 * @file json_exporter.c
 * @brief cJSON-based JSON flow exporter for AegisFlow.
 *
 * Serialises FlowFeatures (computed from a closed FlowRecord) into a
 * JSON object whose keys match CICFlowMeter's field names.
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#include "exporter.h"
#include "aegis_features.h"
#include "utils.h"

#include <cJSON.h>
#include <arpa/inet.h>
#include <math.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Helper: add a double to a JSON object, gracefully handling inf/NaN
 * ========================================================================= */
static void json_add_double(cJSON *obj, const char *key, double val) {
    /* cJSON encodes NaN/Inf as null; replace with 0 for safety */
    if (!isfinite(val)) val = 0.0;
    cJSON_AddNumberToObject(obj, key, val);
}

/* =========================================================================
 * Serialise FlowRecord → heap-allocated JSON string
 * ========================================================================= */
char *json_exporter_serialize(const FlowRecord *record, const FlowKey *key) {
    /* ── Extract feature vector ── */
    FlowFeatures f;
    features_extract(record, &f);

    /* ── Format IP addresses ── */
    char src_ip_str[INET_ADDRSTRLEN];
    char dst_ip_str[INET_ADDRSTRLEN];
    utils_ipv4_to_str(f.src_ip, src_ip_str, sizeof(src_ip_str));
    utils_ipv4_to_str(f.dst_ip, dst_ip_str, sizeof(dst_ip_str));

    UNUSED(key); /* key is embedded in features via record->key */

    /* ── Build cJSON object ── */
    cJSON *root = cJSON_CreateObject();
    if (!root) return NULL;

    /* Identity */
    cJSON_AddStringToObject(root, "src_ip",   src_ip_str);
    cJSON_AddStringToObject(root, "dst_ip",   dst_ip_str);
    cJSON_AddNumberToObject(root, "src_port", ntohs(f.src_port));
    cJSON_AddNumberToObject(root, "dst_port", ntohs(f.dst_port));
    cJSON_AddNumberToObject(root, "protocol", f.protocol);

    /* Flow duration */
    json_add_double(root, "flow_duration", f.flow_duration_us);

    /* Packet counts */
    cJSON_AddNumberToObject(root, "total_fwd_packets",  (double)f.total_fwd_packets);
    cJSON_AddNumberToObject(root, "total_bwd_packets",  (double)f.total_bwd_packets);
    cJSON_AddNumberToObject(root, "total_packets",      (double)f.total_packets);

    /* Byte counts */
    cJSON_AddNumberToObject(root, "total_length_fwd_pkts", (double)f.total_len_fwd_pkts);
    cJSON_AddNumberToObject(root, "total_length_bwd_pkts", (double)f.total_len_bwd_pkts);

    /* Packet length stats */
    json_add_double(root, "pkt_length_min",  f.pkt_len_min);
    json_add_double(root, "pkt_length_max",  f.pkt_len_max);
    json_add_double(root, "pkt_length_mean", f.pkt_len_mean);
    json_add_double(root, "pkt_length_std",  f.pkt_len_std);

    /* Rate features */
    json_add_double(root, "flow_bytes_per_sec", f.flow_bytes_per_sec);
    json_add_double(root, "flow_pkts_per_sec",  f.flow_pkts_per_sec);

    /* TCP flags */
    cJSON_AddNumberToObject(root, "syn_flag_count", f.syn_count);
    cJSON_AddNumberToObject(root, "ack_flag_count", f.ack_count);
    cJSON_AddNumberToObject(root, "rst_flag_count", f.rst_count);
    cJSON_AddNumberToObject(root, "fin_flag_count", f.fin_count);
    cJSON_AddNumberToObject(root, "psh_flag_count", f.psh_count);
    cJSON_AddNumberToObject(root, "urg_flag_count", f.urg_count);

    /* Flow IAT */
    json_add_double(root, "flow_iat_mean", f.flow_iat_mean);
    json_add_double(root, "flow_iat_std",  f.flow_iat_std);
    json_add_double(root, "flow_iat_min",  f.flow_iat_min);
    json_add_double(root, "flow_iat_max",  f.flow_iat_max);

    /* Forward packet length */
    json_add_double(root, "fwd_pkt_length_mean", f.fwd_pkt_len_mean);
    json_add_double(root, "fwd_pkt_length_std",  f.fwd_pkt_len_std);
    json_add_double(root, "fwd_pkt_length_min",  f.fwd_pkt_len_min);
    json_add_double(root, "fwd_pkt_length_max",  f.fwd_pkt_len_max);

    /* Backward packet length */
    json_add_double(root, "bwd_pkt_length_mean", f.bwd_pkt_len_mean);
    json_add_double(root, "bwd_pkt_length_std",  f.bwd_pkt_len_std);
    json_add_double(root, "bwd_pkt_length_min",  f.bwd_pkt_len_min);
    json_add_double(root, "bwd_pkt_length_max",  f.bwd_pkt_len_max);

    /* Forward IAT */
    json_add_double(root, "fwd_iat_mean", f.fwd_iat_mean);
    json_add_double(root, "fwd_iat_std",  f.fwd_iat_std);
    json_add_double(root, "fwd_iat_min",  f.fwd_iat_min);
    json_add_double(root, "fwd_iat_max",  f.fwd_iat_max);

    /* Backward IAT */
    json_add_double(root, "bwd_iat_mean", f.bwd_iat_mean);
    json_add_double(root, "bwd_iat_std",  f.bwd_iat_std);
    json_add_double(root, "bwd_iat_min",  f.bwd_iat_min);
    json_add_double(root, "bwd_iat_max",  f.bwd_iat_max);

    /* ── Serialise to string ── */
    char *json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    return json_str;  /* caller must free() */
}

/* =========================================================================
 * ExportFn-compatible writer
 * ========================================================================= */
void json_exporter_write(const FlowRecord *record,
                         const FlowKey    *key,
                         void             *ctx) {
    FILE *fp = ctx ? (FILE *)ctx : stdout;

    char *json_str = json_exporter_serialize(record, key);
    if (!json_str) {
        LOG_ERROR("json_exporter_serialize returned NULL (OOM?)");
        return;
    }

    fputs(json_str, fp);
    fputc('\n', fp);
    fflush(fp);

    free(json_str);
}
