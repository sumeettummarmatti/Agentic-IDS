/**
 * @file features.c
 * @brief Incremental feature computation engine — Welford + per-packet updates.
 *
 * All operations are O(1) per packet.  No packet data is ever buffered.
 * Population variance (÷N) is used to match CICFlowMeter's implementation.
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#include "aegis_features.h"
#include "utils.h"

#include <math.h>
#include <string.h>
#include <float.h>

/* =========================================================================
 * Welford's Online Algorithm
 * ========================================================================= */

void welford_init(WelfordState *state) {
    state->count = 0;
    state->mean  = 0.0;
    state->M2    = 0.0;
}

void welford_update(WelfordState *state, double value) {
    state->count++;
    double delta  = value - state->mean;
    state->mean  += delta / (double)state->count;
    double delta2 = value - state->mean;
    state->M2    += delta * delta2;
}

void welford_finalize(const WelfordState *state,
                      double *mean,
                      double *variance,
                      double *std_dev) {
    if (mean)     *mean     = state->mean;
    if (variance) *variance = (state->count < 2) ? 0.0 : state->M2 / (double)state->count;
    if (std_dev) {
        double var = (state->count < 2) ? 0.0 : state->M2 / (double)state->count;
        *std_dev = (var > 0.0) ? sqrt(var) : 0.0;
    }
}

/* =========================================================================
 * DirectionStats helpers
 * ========================================================================= */

void direction_stats_init(DirectionStats *ds) {
    memset(ds, 0, sizeof(*ds));
    ds->min_len = DBL_MAX;
    ds->max_len = 0.0;
    ds->min_iat = DBL_MAX;
    ds->max_iat = 0.0;
}

void direction_stats_update(DirectionStats *ds, double len,
                            const struct timeval *ts) {
    /* ── Packet count + byte count ── */
    ds->pkt_count++;
    ds->byte_count += (uint64_t)len;

    /* ── Packet length statistics ── */
    welford_update(&ds->len_stats, len);
    if (len < ds->min_len) ds->min_len = len;
    if (len > ds->max_len) ds->max_len = len;

    /* ── Inter-arrival time ── */
    if (ds->has_last_pkt) {
        int64_t iat_us = timeval_diff_usec(&ds->last_pkt_time, ts);
        if (iat_us < 0) iat_us = 0; /* clock skew guard */
        double iat = (double)iat_us;

        welford_update(&ds->iat_stats, iat);
        if (iat < ds->min_iat) ds->min_iat = iat;
        if (iat > ds->max_iat) ds->max_iat = iat;
    }

    ds->last_pkt_time = *ts;
    ds->has_last_pkt  = 1;
}

/* =========================================================================
 * Per-packet feature update
 * ========================================================================= */

void features_update(FlowRecord *record, const PacketInfo *pkt) {
    double len = (double)pkt->payload_len;
    const struct timeval *ts = &pkt->timestamp;

    /* ── Timestamps ── */
    record->last_seen = *ts;

    /* ── Direction-specific update ── */
    if (pkt->is_fwd) {
        direction_stats_update(&record->fwd, len, ts);
    } else {
        direction_stats_update(&record->bwd, len, ts);
    }

    /* ── Global packet length stats (fwd + bwd combined) ── */
    welford_update(&record->pkt_len_stats, len);
    if (len < record->min_pkt_len) record->min_pkt_len = len;
    if (len > record->max_pkt_len) record->max_pkt_len = len;

    /* ── Global inter-arrival time stats ── */
    if (record->has_last_pkt) {
        int64_t iat_us = timeval_diff_usec(&record->last_pkt_time, ts);
        if (iat_us < 0) iat_us = 0;
        double iat = (double)iat_us;

        welford_update(&record->iat_stats, iat);
        if (iat < record->min_iat) record->min_iat = iat;
        if (iat > record->max_iat) record->max_iat = iat;
    }
    record->last_pkt_time = *ts;
    record->has_last_pkt  = 1;

    /* ── TCP flag accumulators ── */
    if (pkt->tcp_flags & TCP_FLAG_SYN) record->syn_count++;
    if (pkt->tcp_flags & TCP_FLAG_ACK) record->ack_count++;
    if (pkt->tcp_flags & TCP_FLAG_RST) record->rst_count++;
    if (pkt->tcp_flags & TCP_FLAG_FIN) record->fin_count++;
    if (pkt->tcp_flags & TCP_FLAG_PSH) record->psh_count++;
    if (pkt->tcp_flags & TCP_FLAG_URG) record->urg_count++;
}

/* =========================================================================
 * Feature vector extraction
 * ========================================================================= */

void features_extract(const FlowRecord *record, FlowFeatures *features) {
    memset(features, 0, sizeof(*features));

    /* ── Identity ── */
    features->src_ip   = record->key.src_ip;
    features->dst_ip   = record->key.dst_ip;
    features->src_port = record->key.src_port;
    features->dst_port = record->key.dst_port;
    features->protocol = record->key.protocol;

    /* ── Duration ── */
    int64_t dur_us = timeval_diff_usec(&record->first_seen, &record->last_seen);
    if (dur_us < 0) dur_us = 0;
    features->flow_duration_us = (double)dur_us;

    /* ── Packet / byte counts ── */
    features->total_fwd_packets  = record->fwd.pkt_count;
    features->total_bwd_packets  = record->bwd.pkt_count;
    features->total_packets      = record->fwd.pkt_count + record->bwd.pkt_count;
    features->total_len_fwd_pkts = record->fwd.byte_count;
    features->total_len_bwd_pkts = record->bwd.byte_count;

    /* ── Global packet length stats ── */
    double pkt_mean, pkt_var, pkt_std;
    welford_finalize(&record->pkt_len_stats, &pkt_mean, &pkt_var, &pkt_std);
    features->pkt_len_mean = pkt_mean;
    features->pkt_len_std  = pkt_std;
    features->pkt_len_min  = (record->pkt_len_stats.count > 0) ? record->min_pkt_len : 0.0;
    features->pkt_len_max  = record->max_pkt_len;

    /* ── Rate features (computed at export time) ── */
    double dur_sec = features->flow_duration_us * 1e-6;
    if (dur_sec > 0.0) {
        double total_bytes = (double)(record->fwd.byte_count + record->bwd.byte_count);
        double total_pkts  = (double)(record->fwd.pkt_count  + record->bwd.pkt_count);
        features->flow_bytes_per_sec = total_bytes / dur_sec;
        features->flow_pkts_per_sec  = total_pkts  / dur_sec;
    } else {
        features->flow_bytes_per_sec = 0.0;
        features->flow_pkts_per_sec  = 0.0;
    }

    /* ── TCP flag counts ── */
    features->syn_count = record->syn_count;
    features->ack_count = record->ack_count;
    features->rst_count = record->rst_count;
    features->fin_count = record->fin_count;
    features->psh_count = record->psh_count;
    features->urg_count = record->urg_count;

    /* ── Global IAT ── */
    double iat_mean, iat_var, iat_std;
    welford_finalize(&record->iat_stats, &iat_mean, &iat_var, &iat_std);
    features->flow_iat_mean = iat_mean;
    features->flow_iat_std  = iat_std;
    features->flow_iat_min  = (record->iat_stats.count > 0) ? record->min_iat : 0.0;
    features->flow_iat_max  = record->max_iat;

    /* ── Forward packet length ── */
    {
        double mean, var, std;
        welford_finalize(&record->fwd.len_stats, &mean, &var, &std);
        features->fwd_pkt_len_mean = mean;
        features->fwd_pkt_len_std  = std;
        features->fwd_pkt_len_min  = (record->fwd.pkt_count > 0) ? record->fwd.min_len : 0.0;
        features->fwd_pkt_len_max  = record->fwd.max_len;
    }

    /* ── Backward packet length ── */
    {
        double mean, var, std;
        welford_finalize(&record->bwd.len_stats, &mean, &var, &std);
        features->bwd_pkt_len_mean = mean;
        features->bwd_pkt_len_std  = std;
        features->bwd_pkt_len_min  = (record->bwd.pkt_count > 0) ? record->bwd.min_len : 0.0;
        features->bwd_pkt_len_max  = record->bwd.max_len;
    }

    /* ── Forward IAT ── */
    {
        double mean, var, std;
        welford_finalize(&record->fwd.iat_stats, &mean, &var, &std);
        features->fwd_iat_mean = mean;
        features->fwd_iat_std  = std;
        features->fwd_iat_min  = (record->fwd.iat_stats.count > 0) ? record->fwd.min_iat : 0.0;
        features->fwd_iat_max  = record->fwd.max_iat;
    }

    /* ── Backward IAT ── */
    {
        double mean, var, std;
        welford_finalize(&record->bwd.iat_stats, &mean, &var, &std);
        features->bwd_iat_mean = mean;
        features->bwd_iat_std  = std;
        features->bwd_iat_min  = (record->bwd.iat_stats.count > 0) ? record->bwd.min_iat : 0.0;
        features->bwd_iat_max  = record->bwd.max_iat;
    }
}
