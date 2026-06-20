/**
 * @file flow.c
 * @brief FlowRecord lifecycle — create, initialise, and reset.
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#include "flow.h"
#include "aegis_features.h"
#include "utils.h"

#include <string.h>
#include <float.h>

/* =========================================================================
 * FlowRecord initialisation
 *
 * Called once when a new flow is created.  Sets all counters to zero,
 * min values to DBL_MAX (so the first packet wins), and records the
 * first-seen timestamp.
 * ========================================================================= */
void flow_record_init(FlowRecord *record,
                      const FlowKey *key,
                      const struct timeval *first_ts) {
    memset(record, 0, sizeof(*record));

    /* ── Key ── */
    record->key = *key;

    /* ── Timestamps ── */
    record->first_seen    = *first_ts;
    record->last_seen     = *first_ts;
    record->last_pkt_time = *first_ts;
    record->has_last_pkt  = 0;   /* IAT not defined until 2nd packet */

    /* ── Global packet length init ── */
    welford_init(&record->pkt_len_stats);
    record->min_pkt_len = DBL_MAX;
    record->max_pkt_len = 0.0;

    /* ── Global IAT init ── */
    welford_init(&record->iat_stats);
    record->min_iat = DBL_MAX;
    record->max_iat = 0.0;

    /* ── Directional stats init ── */
    direction_stats_init(&record->fwd);
    direction_stats_init(&record->bwd);

    /* ── Protocol (stored redundantly for quick access) ── */
    record->protocol = key->protocol;
}

/* =========================================================================
 * FlowKey comparison utilities
 * ========================================================================= */

/**
 * @brief Test whether two FlowKeys are identical.
 */
int flow_key_equal(const FlowKey *a, const FlowKey *b) {
    /* Compare all non-padding bytes.  Since _pad is explicitly zeroed
     * by the packet parser, memcmp is safe across the full struct. */
    return memcmp(a, b, sizeof(FlowKey)) == 0;
}

/**
 * @brief Return the canonical (forward) form of a 5-tuple.
 *
 * AegisFlow stores each bidirectional flow keyed on the FIRST packet's
 * direction (matching CICFlowMeter behaviour).  This function simply
 * copies the key as-is — canonicalisation is done at the call site in
 * flow_table.c by comparing the incoming 5-tuple against the stored key.
 */
void flow_key_copy(FlowKey *dst, const FlowKey *src) {
    *dst = *src;
    dst->_pad[0] = dst->_pad[1] = dst->_pad[2] = 0; /* ensure padding is clean */
}

/**
 * @brief Build the reverse-direction key from a given key.
 *
 * Used to look up a flow when the reply packet arrives.
 * Swaps src ↔ dst for both IP addresses and ports; protocol unchanged.
 */
void flow_key_reverse(FlowKey *rev, const FlowKey *key) {
    rev->src_ip   = key->dst_ip;
    rev->dst_ip   = key->src_ip;
    rev->src_port = key->dst_port;
    rev->dst_port = key->src_port;
    rev->protocol = key->protocol;
    rev->_pad[0]  = rev->_pad[1] = rev->_pad[2] = 0;
}
