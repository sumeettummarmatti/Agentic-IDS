/**
 * @file test_flow.c
 * @brief Unit tests for FlowRecord lifecycle and FlowTable operations.
 *
 * Tests:
 *   - FlowKey equality, copy, reverse
 *   - FlowRecord initialisation
 *   - Flow table create/update/expire/flush/destroy
 *   - Bidirectional flow detection
 *   - TCP FIN/RST flow closure
 *   - Timeout expiry
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#include "test_runner.h"
#include "../include/flow.h"
#include "../include/flow_table.h"
#include "../include/aegis_features.h"

#include <string.h>
#include <stdlib.h>
#include <float.h>

/* =========================================================================
 * Exported from flow.c (internal helpers used by tests)
 * ========================================================================= */
extern void flow_record_init(FlowRecord *record, const FlowKey *key,
                             const struct timeval *first_ts);
extern void flow_key_reverse(FlowKey *rev, const FlowKey *key);

/* =========================================================================
 * Test helpers
 * ========================================================================= */

/* Closure count tracker */
static int g_close_count = 0;

static void counting_exporter(const FlowRecord *r, const FlowKey *k, void *ctx) {
    (void)r; (void)k; (void)ctx;
    g_close_count++;
}

static FlowKey make_key(uint32_t sip, uint32_t dip,
                        uint16_t sp, uint16_t dp, uint8_t proto) {
    FlowKey k;
    memset(&k, 0, sizeof(k));
    k.src_ip   = sip;
    k.dst_ip   = dip;
    k.src_port = sp;
    k.dst_port = dp;
    k.protocol = proto;
    return k;
}

static PacketInfo make_pkt_with_key(FlowKey key, uint32_t len,
                                    long sec, long usec,
                                    uint8_t flags) {
    PacketInfo p;
    memset(&p, 0, sizeof(p));
    p.key         = key;
    p.payload_len = len;
    p.timestamp.tv_sec  = sec;
    p.timestamp.tv_usec = usec;
    p.tcp_flags   = flags;
    p.is_fwd      = 1;
    return p;
}

/* =========================================================================
 * FlowKey: equality
 * ========================================================================= */
TEST(flow_key_equal) {
    FlowKey a = make_key(0x01010101, 0x02020202, 1234, 80, PROTO_TCP);
    FlowKey b = a;
    ASSERT_TRUE(memcmp(&a, &b, sizeof(FlowKey)) == 0);
}

/* =========================================================================
 * FlowKey: reverse
 * ========================================================================= */
TEST(flow_key_reverse) {
    FlowKey orig = make_key(0x0A000001, 0x08080808, 54321, 443, PROTO_TCP);
    FlowKey rev;
    flow_key_reverse(&rev, &orig);

    ASSERT_EQ(rev.src_ip,   orig.dst_ip);
    ASSERT_EQ(rev.dst_ip,   orig.src_ip);
    ASSERT_EQ(rev.src_port, orig.dst_port);
    ASSERT_EQ(rev.dst_port, orig.src_port);
    ASSERT_EQ(rev.protocol, orig.protocol);
}

/* =========================================================================
 * FlowRecord: initialisation
 * ========================================================================= */
TEST(flow_record_init_zeroed) {
    FlowKey k = make_key(1, 2, 100, 80, PROTO_TCP);
    struct timeval ts = {5000, 12345};

    FlowRecord r;
    flow_record_init(&r, &k, &ts);

    ASSERT_EQ(r.fwd.pkt_count, 0);
    ASSERT_EQ(r.bwd.pkt_count, 0);
    ASSERT_EQ(r.syn_count, 0);
    ASSERT_EQ(r.has_last_pkt, 0);
    ASSERT_EQ(r.first_seen.tv_sec, 5000);
    ASSERT_NEAR(r.min_pkt_len, DBL_MAX, 0.0);
    ASSERT_NEAR(r.max_pkt_len, 0.0,     0.0);
}

/* =========================================================================
 * FlowTable: basic creation and destruction
 * ========================================================================= */
TEST(flow_table_create_destroy) {
    FlowTableConfig cfg = {
        .tcp_timeout_sec  = 120,
        .udp_timeout_sec  = 60,
        .max_flows        = 0,
        .thread_safe      = 0,
        .on_flow_close    = NULL,
        .on_flow_close_ctx = NULL
    };
    FlowTable *ft = flow_table_create(&cfg);
    ASSERT_NOT_NULL(ft);
    ASSERT_EQ(flow_table_count(ft), 0);
    flow_table_destroy(ft);
}

/* =========================================================================
 * FlowTable: insert and count
 * ========================================================================= */
TEST(flow_table_insert) {
    g_close_count = 0;
    FlowTableConfig cfg = {
        .tcp_timeout_sec = 120, .udp_timeout_sec = 60,
        .on_flow_close = counting_exporter
    };
    FlowTable *ft = flow_table_create(&cfg);
    ASSERT_NOT_NULL(ft);

    /* Insert 3 distinct flows */
    for (int i = 0; i < 3; i++) {
        FlowKey k = make_key(i, 0x0808, 1000 + i, 80, PROTO_TCP);
        PacketInfo p = make_pkt_with_key(k, 100, 1000, 0, TCP_FLAG_SYN);
        flow_table_update(ft, &p);
    }

    ASSERT_EQ(flow_table_count(ft), 3);

    FlowTableStats s;
    flow_table_get_stats(ft, &s);
    ASSERT_EQ(s.flows_created, 3);
    ASSERT_EQ(s.packets_processed, 3);

    flow_table_destroy(ft);
}

/* =========================================================================
 * FlowTable: TCP FIN closes flow
 * ========================================================================= */
TEST(flow_table_fin_closes_flow) {
    g_close_count = 0;
    FlowTableConfig cfg = {
        .tcp_timeout_sec = 120, .udp_timeout_sec = 60,
        .on_flow_close = counting_exporter
    };
    FlowTable *ft = flow_table_create(&cfg);

    FlowKey k = make_key(0x0A000001, 0x0808, 54321, 80, PROTO_TCP);

    /* SYN */
    PacketInfo p = make_pkt_with_key(k, 40, 1000, 0, TCP_FLAG_SYN);
    flow_table_update(ft, &p);
    ASSERT_EQ(flow_table_count(ft), 1);

    /* ACK + PSH data */
    p = make_pkt_with_key(k, 512, 1000, 100000, TCP_FLAG_ACK | TCP_FLAG_PSH);
    flow_table_update(ft, &p);
    ASSERT_EQ(flow_table_count(ft), 1);

    /* FIN — should close */
    p = make_pkt_with_key(k, 20, 1000, 200000, TCP_FLAG_FIN | TCP_FLAG_ACK);
    flow_table_update(ft, &p);

    /* Flow should be gone */
    ASSERT_EQ(flow_table_count(ft), 0);
    ASSERT_EQ(g_close_count, 1);

    FlowTableStats s;
    flow_table_get_stats(ft, &s);
    ASSERT_EQ(s.flows_closed, 1);

    flow_table_destroy(ft);
}

/* =========================================================================
 * FlowTable: TCP RST closes flow
 * ========================================================================= */
TEST(flow_table_rst_closes_flow) {
    g_close_count = 0;
    FlowTableConfig cfg = {
        .tcp_timeout_sec = 120,
        .on_flow_close = counting_exporter
    };
    FlowTable *ft = flow_table_create(&cfg);

    FlowKey k = make_key(0x01, 0x02, 9999, 443, PROTO_TCP);
    PacketInfo p = make_pkt_with_key(k, 100, 2000, 0, TCP_FLAG_SYN);
    flow_table_update(ft, &p);

    p = make_pkt_with_key(k, 0, 2000, 50000, TCP_FLAG_RST);
    flow_table_update(ft, &p);

    ASSERT_EQ(flow_table_count(ft), 0);
    ASSERT_EQ(g_close_count, 1);
    flow_table_destroy(ft);
}

/* =========================================================================
 * FlowTable: timeout expiry
 * ========================================================================= */
TEST(flow_table_expire) {
    g_close_count = 0;
    FlowTableConfig cfg = {
        .tcp_timeout_sec = 1,   /* very short timeout */
        .udp_timeout_sec = 1,
        .on_flow_close = counting_exporter
    };
    FlowTable *ft = flow_table_create(&cfg);

    FlowKey k = make_key(0x01, 0x02, 1111, 80, PROTO_TCP);
    PacketInfo p = make_pkt_with_key(k, 64, 1000, 0, TCP_FLAG_ACK);
    flow_table_update(ft, &p);
    ASSERT_EQ(flow_table_count(ft), 1);

    /* Set 'now' to 5 seconds later — beyond the 1-second timeout */
    struct timeval now = {1005, 0};
    uint32_t expired = flow_table_expire(ft, &now);

    ASSERT_EQ(expired, 1);
    ASSERT_EQ(flow_table_count(ft), 0);
    ASSERT_EQ(g_close_count, 1);

    FlowTableStats s;
    flow_table_get_stats(ft, &s);
    ASSERT_EQ(s.flows_expired, 1);

    flow_table_destroy(ft);
}

/* =========================================================================
 * FlowTable: bidirectional flow detection (reply packet reuses same record)
 * ========================================================================= */
TEST(flow_table_bidirectional) {
    g_close_count = 0;
    FlowTableConfig cfg = {
        .tcp_timeout_sec = 120,
        .on_flow_close = counting_exporter
    };
    FlowTable *ft = flow_table_create(&cfg);

    FlowKey fwd = make_key(0x0A000001, 0x0A000002, 50000, 80, PROTO_TCP);
    FlowKey rev = make_key(0x0A000002, 0x0A000001, 80, 50000, PROTO_TCP);

    /* Forward SYN */
    PacketInfo p = make_pkt_with_key(fwd, 40, 1000, 0, TCP_FLAG_SYN);
    flow_table_update(ft, &p);
    ASSERT_EQ(flow_table_count(ft), 1);

    /* Reverse SYN-ACK — should match the SAME flow (not create a new one) */
    p = make_pkt_with_key(rev, 44, 1000, 100000, TCP_FLAG_SYN | TCP_FLAG_ACK);
    flow_table_update(ft, &p);
    ASSERT_EQ(flow_table_count(ft), 1);  /* still just 1 flow */

    FlowTableStats s;
    flow_table_get_stats(ft, &s);
    ASSERT_EQ(s.flows_created, 1);
    ASSERT_EQ(s.packets_processed, 2);

    flow_table_flush(ft);
    flow_table_destroy(ft);
}

/* =========================================================================
 * FlowTable: flush exports remaining flows
 * ========================================================================= */
TEST(flow_table_flush) {
    g_close_count = 0;
    FlowTableConfig cfg = {
        .tcp_timeout_sec = 120,
        .on_flow_close = counting_exporter
    };
    FlowTable *ft = flow_table_create(&cfg);

    /* Create 5 flows without closing them */
    for (int i = 0; i < 5; i++) {
        FlowKey k = make_key(i + 1, 0xFF, 2000 + i, 443, PROTO_TCP);
        PacketInfo p = make_pkt_with_key(k, 100, 3000, i * 1000, TCP_FLAG_ACK);
        flow_table_update(ft, &p);
    }
    ASSERT_EQ(flow_table_count(ft), 5);

    uint32_t flushed = flow_table_flush(ft);
    ASSERT_EQ(flushed, 5);
    ASSERT_EQ(flow_table_count(ft), 0);
    ASSERT_EQ(g_close_count, 5);

    flow_table_destroy(ft);
}

/* =========================================================================
 * Test main
 * ========================================================================= */
int main(void) {
    printf("\n══════════ AegisFlow: Flow Lifecycle Tests ══════════\n\n");

    RUN_TEST(flow_key_equal);
    RUN_TEST(flow_key_reverse);
    RUN_TEST(flow_record_init_zeroed);
    RUN_TEST(flow_table_create_destroy);
    RUN_TEST(flow_table_insert);
    RUN_TEST(flow_table_fin_closes_flow);
    RUN_TEST(flow_table_rst_closes_flow);
    RUN_TEST(flow_table_expire);
    RUN_TEST(flow_table_bidirectional);
    RUN_TEST(flow_table_flush);

    TEST_SUMMARY();
    return (g_tests_failed > 0) ? 1 : 0;
}
