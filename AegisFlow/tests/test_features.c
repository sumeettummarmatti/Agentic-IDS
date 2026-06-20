/**
 * @file test_features.c
 * @brief Unit tests for the Welford online algorithm and feature computation.
 *
 * Verifies numerical correctness of:
 *   - welford_update / welford_finalize (mean, variance, std dev)
 *   - direction_stats_update (IAT, packet length min/max)
 *   - features_update / features_extract (full pipeline)
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#include "test_runner.h"
#include "../include/aegis_features.h"
#include "../include/flow.h"

#include <float.h>
#include <math.h>
#include <string.h>

/* =========================================================================
 * Helper: build a simple PacketInfo for testing
 * ========================================================================= */
static PacketInfo make_pkt(uint32_t len, long sec, long usec,
                           uint8_t flags, uint8_t is_fwd) {
    PacketInfo p;
    memset(&p, 0, sizeof(p));
    p.payload_len        = len;
    p.timestamp.tv_sec   = sec;
    p.timestamp.tv_usec  = usec;
    p.tcp_flags          = flags;
    p.is_fwd             = is_fwd;
    p.key.protocol       = PROTO_TCP;
    return p;
}

/* =========================================================================
 * Helper: build a zeroed FlowRecord
 * ========================================================================= */
static FlowRecord make_record(void) {
    FlowRecord r;
    memset(&r, 0, sizeof(r));

    welford_init(&r.pkt_len_stats);
    welford_init(&r.iat_stats);
    r.min_pkt_len = DBL_MAX;
    r.max_pkt_len = 0.0;
    r.min_iat = DBL_MAX;
    r.max_iat = 0.0;

    /* Init directions */
    direction_stats_init(&r.fwd);
    direction_stats_init(&r.bwd);

    r.first_seen.tv_sec  = 1000;
    r.first_seen.tv_usec = 0;

    return r;
}

/* =========================================================================
 * Welford: empty state
 * ========================================================================= */
TEST(welford_empty) {
    WelfordState s;
    welford_init(&s);
    double m, v, sd;
    welford_finalize(&s, &m, &v, &sd);
    ASSERT_NEAR(m,  0.0, 1e-10);
    ASSERT_NEAR(v,  0.0, 1e-10);
    ASSERT_NEAR(sd, 0.0, 1e-10);
    ASSERT_EQ(s.count, 0);
}

/* =========================================================================
 * Welford: single sample
 * ========================================================================= */
TEST(welford_single) {
    WelfordState s;
    welford_init(&s);
    welford_update(&s, 42.0);
    double m, v, sd;
    welford_finalize(&s, &m, &v, &sd);
    ASSERT_NEAR(m,  42.0, 1e-10);
    ASSERT_NEAR(v,  0.0,  1e-10);
    ASSERT_NEAR(sd, 0.0,  1e-10);
    ASSERT_EQ(s.count, 1);
}

/* =========================================================================
 * Welford: known dataset [2, 4, 4, 4, 5, 5, 7, 9]
 * Population mean = 5.0, population variance = 4.0, std = 2.0
 * Reference: Wikipedia "Algorithms for calculating variance"
 * ========================================================================= */
TEST(welford_known_dataset) {
    WelfordState s;
    welford_init(&s);
    double data[] = {2, 4, 4, 4, 5, 5, 7, 9};
    for (int i = 0; i < 8; i++) welford_update(&s, data[i]);

    double m, v, sd;
    welford_finalize(&s, &m, &v, &sd);

    ASSERT_EQ(s.count, 8);
    ASSERT_NEAR(m,  5.0, 1e-9);
    ASSERT_NEAR(v,  4.0, 1e-9);
    ASSERT_NEAR(sd, 2.0, 1e-9);
}

/* =========================================================================
 * Welford: large values (numerical stability check)
 * ========================================================================= */
TEST(welford_large_values) {
    WelfordState s;
    welford_init(&s);
    /* Two values 1e9 ± 1 — mean should be 1e9, variance should be 1 */
    welford_update(&s, 1e9 + 1.0);
    welford_update(&s, 1e9 - 1.0);
    double m, v, sd;
    welford_finalize(&s, &m, &v, &sd);
    ASSERT_NEAR(m,  1e9, 1e-3);
    ASSERT_NEAR(v,  1.0, 1e-3);
    ASSERT_NEAR(sd, 1.0, 1e-3);
}

/* =========================================================================
 * direction_stats: min/max tracking
 * ========================================================================= */
TEST(direction_stats_min_max) {
    DirectionStats ds;
    direction_stats_init(&ds);

    struct timeval ts = {1000, 0};
    direction_stats_update(&ds, 100.0, &ts);
    ts.tv_usec = 500000;
    direction_stats_update(&ds, 50.0, &ts);
    ts.tv_sec = 1001; ts.tv_usec = 0;
    direction_stats_update(&ds, 200.0, &ts);

    ASSERT_EQ(ds.pkt_count, 3);
    ASSERT_EQ(ds.byte_count, 350);
    ASSERT_NEAR(ds.min_len, 50.0,  1e-9);
    ASSERT_NEAR(ds.max_len, 200.0, 1e-9);
}

/* =========================================================================
 * direction_stats: IAT computation (500ms, 500ms)
 * ========================================================================= */
TEST(direction_stats_iat) {
    DirectionStats ds;
    direction_stats_init(&ds);

    struct timeval t0 = {1000, 0};
    struct timeval t1 = {1000, 500000};   /* +500ms */
    struct timeval t2 = {1001, 0};        /* +500ms */

    direction_stats_update(&ds, 64.0, &t0);
    direction_stats_update(&ds, 64.0, &t1);
    direction_stats_update(&ds, 64.0, &t2);

    /* 2 IAT values, both 500000 µs */
    ASSERT_NEAR(ds.min_iat, 500000.0, 1.0);
    ASSERT_NEAR(ds.max_iat, 500000.0, 1.0);

    double mean, var, sd;
    welford_finalize(&ds.iat_stats, &mean, &var, &sd);
    ASSERT_NEAR(mean, 500000.0, 1.0);
    ASSERT_NEAR(sd,   0.0,      1.0);  /* identical IATs → std=0 */
}

/* =========================================================================
 * features_update / features_extract: basic pipeline
 * ========================================================================= */
TEST(features_pipeline_basic) {
    FlowRecord r = make_record();

    /* 3 forward packets, 1 backward */
    PacketInfo pkts[] = {
        make_pkt(100, 1000, 0,       TCP_FLAG_SYN, 1),
        make_pkt(200, 1000, 100000,  TCP_FLAG_ACK, 1),
        make_pkt( 50, 1000, 200000,  TCP_FLAG_ACK, 0),  /* bwd */
        make_pkt(150, 1000, 300000,  TCP_FLAG_PSH|TCP_FLAG_ACK, 1),
    };
    r.first_seen = pkts[0].timestamp;

    for (int i = 0; i < 4; i++) features_update(&r, &pkts[i]);

    FlowFeatures f;
    features_extract(&r, &f);

    ASSERT_EQ(f.total_fwd_packets, 3);
    ASSERT_EQ(f.total_bwd_packets, 1);
    ASSERT_EQ(f.total_packets, 4);
    ASSERT_EQ(f.total_len_fwd_pkts, 450);  /* 100+200+150 */
    ASSERT_EQ(f.total_len_bwd_pkts, 50);

    ASSERT_EQ(f.syn_count, 1);
    ASSERT_EQ(f.ack_count, 3);
    ASSERT_EQ(f.psh_count, 1);
    ASSERT_EQ(f.rst_count, 0);

    /* Flow duration = 300ms = 300000 µs */
    ASSERT_NEAR(f.flow_duration_us, 300000.0, 1.0);

    /* All packet lengths: 100, 200, 50, 150  → mean=125, std=√(3125)≈55.9 */
    ASSERT_NEAR(f.pkt_len_mean, 125.0, 1e-6);
    ASSERT_NEAR(f.pkt_len_min,  50.0,  1e-6);
    ASSERT_NEAR(f.pkt_len_max,  200.0, 1e-6);
}

/* =========================================================================
 * features_extract: rate features
 * ========================================================================= */
TEST(features_rate_features) {
    FlowRecord r = make_record();

    /* 1 packet per second × 4 packets, 100 bytes each */
    for (int i = 0; i < 4; i++) {
        PacketInfo p = make_pkt(100, 1000 + i, 0, TCP_FLAG_ACK, 1);
        if (i == 0) r.first_seen = p.timestamp;
        features_update(&r, &p);
    }

    FlowFeatures f;
    features_extract(&r, &f);

    /* duration = 3s = 3000000 µs */
    ASSERT_NEAR(f.flow_duration_us, 3000000.0, 1.0);
    /* bytes/sec = 400 / 3 ≈ 133.33 */
    ASSERT_NEAR(f.flow_bytes_per_sec, 400.0 / 3.0, 0.01);
    /* pkts/sec = 4 / 3 ≈ 1.333 */
    ASSERT_NEAR(f.flow_pkts_per_sec, 4.0 / 3.0, 0.01);
}

/* =========================================================================
 * Test main
 * ========================================================================= */
int main(void) {
    printf("\n══════════ AegisFlow: Feature Engine Tests ══════════\n\n");

    RUN_TEST(welford_empty);
    RUN_TEST(welford_single);
    RUN_TEST(welford_known_dataset);
    RUN_TEST(welford_large_values);
    RUN_TEST(direction_stats_min_max);
    RUN_TEST(direction_stats_iat);
    RUN_TEST(features_pipeline_basic);
    RUN_TEST(features_rate_features);

    TEST_SUMMARY();
    return (g_tests_failed > 0) ? 1 : 0;
}
