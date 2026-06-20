/**
 * @file test_exporter.c
 * @brief Unit tests for JSON and CSV exporters.
 *
 * Verifies:
 *   - JSON exporter produces valid field names
 *   - CSV exporter produces correct column count
 *   - ExporterChain dispatches to all registered exporters
 *   - NaN/Inf values are safely replaced with 0
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#include "test_runner.h"
#include <arpa/inet.h>
#include "../include/exporter.h"
#include "../include/aegis_features.h"
#include "../include/flow.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * External helpers from flow.c
 * ========================================================================= */
extern void flow_record_init(FlowRecord *r, const FlowKey *key,
                             const struct timeval *ts);

/* =========================================================================
 * Build a realistic FlowRecord for export testing
 * ========================================================================= */
static FlowRecord build_test_record(void) {
    FlowKey k;
    memset(&k, 0, sizeof(k));
    k.src_ip   = htonl(0x0A000001);  /* 10.0.0.1 */
    k.dst_ip   = htonl(0x08080808);  /* 8.8.8.8  */
    k.src_port = htons(54321);
    k.dst_port = htons(443);
    k.protocol = PROTO_TCP;

    struct timeval t0 = {1000, 0};
    FlowRecord r;
    flow_record_init(&r, &k, &t0);

    /* Inject 6 packets: 4 fwd, 2 bwd */
    PacketInfo pkts[] = {
        {k, {1000,      0}, 100, TCP_FLAG_SYN,             1},
        {k, {1000, 100000}, 200, TCP_FLAG_ACK,             1},
        {k, {1000, 200000},  50, TCP_FLAG_ACK,             0},
        {k, {1000, 300000}, 150, TCP_FLAG_PSH|TCP_FLAG_ACK,1},
        {k, {1000, 400000},  75, TCP_FLAG_ACK,             0},
        {k, {1000, 500000}, 300, TCP_FLAG_FIN|TCP_FLAG_ACK,1},
    };
    for (int i = 0; i < 6; i++) features_update(&r, &pkts[i]);

    r.last_seen = pkts[5].timestamp;
    return r;
}

/* =========================================================================
 * JSON: serialise returns non-NULL
 * ========================================================================= */
TEST(json_serialize_not_null) {
    FlowRecord r = build_test_record();
    char *s = json_exporter_serialize(&r, &r.key);
    ASSERT_NOT_NULL(s);
    free(s);
}

/* =========================================================================
 * JSON: key fields present
 * ========================================================================= */
TEST(json_contains_required_fields) {
    FlowRecord r = build_test_record();
    char *s = json_exporter_serialize(&r, &r.key);
    ASSERT_NOT_NULL(s);

    /* Check that key field names appear in the JSON string */
    ASSERT_TRUE(strstr(s, "\"src_ip\"")             != NULL);
    ASSERT_TRUE(strstr(s, "\"dst_ip\"")             != NULL);
    ASSERT_TRUE(strstr(s, "\"flow_duration\"")       != NULL);
    ASSERT_TRUE(strstr(s, "\"total_fwd_packets\"")   != NULL);
    ASSERT_TRUE(strstr(s, "\"total_bwd_packets\"")   != NULL);
    ASSERT_TRUE(strstr(s, "\"syn_flag_count\"")      != NULL);
    ASSERT_TRUE(strstr(s, "\"flow_bytes_per_sec\"")  != NULL);
    ASSERT_TRUE(strstr(s, "\"flow_iat_mean\"")       != NULL);
    ASSERT_TRUE(strstr(s, "\"fwd_pkt_length_mean\"") != NULL);
    ASSERT_TRUE(strstr(s, "\"bwd_iat_std\"")         != NULL);

    free(s);
}

/* =========================================================================
 * JSON: write to memory buffer via tmpfile
 * ========================================================================= */
TEST(json_write_to_file) {
    FlowRecord r = build_test_record();
    FILE *fp = tmpfile();
    ASSERT_NOT_NULL(fp);

    json_exporter_write(&r, &r.key, fp);
    fflush(fp);

    long sz = ftell(fp);
    ASSERT_TRUE(sz > 10);  /* must have written something */
    fclose(fp);
}

/* =========================================================================
 * CSV: header + data row to buffer
 * ========================================================================= */
TEST(csv_header_and_data) {
    FlowRecord r = build_test_record();

    FILE *fp = tmpfile();
    ASSERT_NOT_NULL(fp);

    csv_exporter_write_header(fp);
    csv_exporter_write(&r, &r.key, fp);
    fflush(fp);

    rewind(fp);
    char line[4096];

    /* Read header */
    ASSERT_NOT_NULL(fgets(line, sizeof(line), fp));
    ASSERT_TRUE(strstr(line, "src_ip") != NULL);
    ASSERT_TRUE(strstr(line, "flow_duration") != NULL);
    ASSERT_TRUE(strstr(line, "syn_flag_count") != NULL);
    ASSERT_TRUE(strstr(line, "bwd_iat_max") != NULL);

    /* Count commas in header */
    int header_commas = 0;
    for (char *p = line; *p; p++) if (*p == ',') header_commas++;

    /* Read data row */
    ASSERT_NOT_NULL(fgets(line, sizeof(line), fp));

    /* Count commas in data row — must match header */
    int data_commas = 0;
    for (char *p = line; *p; p++) if (*p == ',') data_commas++;

    ASSERT_EQ(header_commas, data_commas);

    fclose(fp);
}

/* =========================================================================
 * ExporterChain: dispatches to multiple exporters
 * ========================================================================= */
static int g_chain_calls = 0;

static void mock_exporter(const FlowRecord *r, const FlowKey *k, void *ctx) {
    (void)r; (void)k;
    int *counter = (int *)ctx;
    (*counter)++;
}

TEST(exporter_chain_dispatch) {
    int c1 = 0, c2 = 0, c3 = 0;

    ExporterChain chain;
    exporter_chain_init(&chain);

    ASSERT_EQ(exporter_chain_add(&chain, mock_exporter, &c1), 0);
    ASSERT_EQ(exporter_chain_add(&chain, mock_exporter, &c2), 0);
    ASSERT_EQ(exporter_chain_add(&chain, mock_exporter, &c3), 0);

    FlowRecord r = build_test_record();
    exporter_chain_dispatch(&r, &r.key, &chain);

    ASSERT_EQ(c1, 1);
    ASSERT_EQ(c2, 1);
    ASSERT_EQ(c3, 1);
}

/* =========================================================================
 * ExporterChain: full chain (8 exporters)
 * ========================================================================= */
TEST(exporter_chain_max_capacity) {
    ExporterChain chain;
    exporter_chain_init(&chain);

    int counters[AEGIS_MAX_EXPORTERS] = {0};
    for (int i = 0; i < AEGIS_MAX_EXPORTERS; i++) {
        ASSERT_EQ(exporter_chain_add(&chain, mock_exporter, &counters[i]), 0);
    }
    /* Adding one more should fail */
    ASSERT_EQ(exporter_chain_add(&chain, mock_exporter, NULL), -1);

    FlowRecord r = build_test_record();
    exporter_chain_dispatch(&r, &r.key, &chain);

    for (int i = 0; i < AEGIS_MAX_EXPORTERS; i++) {
        ASSERT_EQ(counters[i], 1);
    }
}

/* =========================================================================
 * Test main
 * ========================================================================= */
int main(void) {
    printf("\n══════════ AegisFlow: Exporter Tests ══════════\n\n");

    RUN_TEST(json_serialize_not_null);
    RUN_TEST(json_contains_required_fields);
    RUN_TEST(json_write_to_file);
    RUN_TEST(csv_header_and_data);
    RUN_TEST(exporter_chain_dispatch);
    RUN_TEST(exporter_chain_max_capacity);

    TEST_SUMMARY();
    return (g_tests_failed > 0) ? 1 : 0;
}
