/**
 * @file aegis_features.h
 * @brief Incremental feature computation engine for AegisFlow.
 *
 * NOTE: This file was previously named features.h, but that name conflicts
 * with glibc's system header <features.h>. All AegisFlow source files should
 * include this file instead.
 *
 * All feature computation is O(1) per packet.  No packet data is retained
 * beyond the current call.  Welford's online algorithm is used for all
 * mean/variance/std computations.
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#ifndef AEGISFLOW_AEGIS_FEATURES_H
#define AEGISFLOW_AEGIS_FEATURES_H

#include "flow.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Welford's Online Algorithm
 *
 * Provides numerically stable, single-pass mean + population variance.
 * Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
 * ========================================================================= */

/**
 * @brief Initialise a WelfordState to zero (empty).
 * @param state Non-NULL pointer to state to initialise.
 */
void welford_init(WelfordState *state);

/**
 * @brief Incorporate one new sample into the running statistics.
 *
 * Must be called once per sample in arrival order.  Thread-unsafe;
 * external locking required if called concurrently on the same state.
 *
 * @param state Non-NULL Welford state.
 * @param value The new sample value.
 */
void welford_update(WelfordState *state, double value);

/**
 * @brief Extract mean, population variance, and std deviation.
 *
 * Safe to call at any time (including with 0 or 1 samples).
 * With 0 samples: mean=0, variance=0, std_dev=0.
 * With 1 sample:  mean=value, variance=0, std_dev=0.
 *
 * @param state    Non-NULL Welford state (const — does not modify).
 * @param mean     Output: running mean (may be NULL).
 * @param variance Output: population variance (may be NULL).
 * @param std_dev  Output: population standard deviation (may be NULL).
 */
void welford_finalize(const WelfordState *state,
                      double *mean,
                      double *variance,
                      double *std_dev);

/* =========================================================================
 * Per-packet feature update
 * ========================================================================= */

/**
 * @brief Update all statistics in a FlowRecord for one incoming packet.
 *
 * Updates (in O(1)):
 *   - Direction-specific packet count and byte count
 *   - Direction-specific packet length mean/std (Welford)
 *   - Direction-specific IAT mean/std (Welford)
 *   - Global packet length mean/std (Welford)
 *   - Global IAT mean/std (Welford)
 *   - TCP flag counters
 *   - Timestamps (last_seen, direction last_pkt_time)
 *
 * @param record  Non-NULL FlowRecord to update (must already be initialised).
 * @param pkt     Non-NULL parsed packet descriptor.
 */
void features_update(FlowRecord *record, const PacketInfo *pkt);

/* =========================================================================
 * Feature vector extraction
 * ========================================================================= */

/**
 * @brief Compute the final FlowFeatures vector from a completed FlowRecord.
 *
 * Should be called once when a flow is closed (FIN/RST/timeout).
 * Rate features (bytes/sec, pkts/sec) are computed here from raw counters.
 *
 * @param record   Non-NULL source FlowRecord.
 * @param features Non-NULL output FlowFeatures (caller-allocated).
 */
void features_extract(const FlowRecord *record, FlowFeatures *features);

/* =========================================================================
 * DirectionStats helpers
 * ========================================================================= */

/**
 * @brief Initialise a DirectionStats struct to zero.
 */
void direction_stats_init(DirectionStats *ds);

/**
 * @brief Update a DirectionStats with one new packet.
 *
 * @param ds      Direction stats to update.
 * @param len     Packet payload length.
 * @param ts      Packet timestamp.
 */
void direction_stats_update(DirectionStats *ds, double len,
                            const struct timeval *ts);

#ifdef __cplusplus
}
#endif

#endif /* AEGISFLOW_AEGIS_FEATURES_H */
