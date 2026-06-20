/**
 * @file flow_table.c
 * @brief uthash-based O(1) flow table implementation.
 *
 * Manages FlowRecord objects using uthash keyed on the 5-tuple FlowKey.
 * Bidirectionality: when a packet arrives, we first try the exact key;
 * if not found, try the reversed key (reply direction).  If neither
 * matches, a new flow is created keyed on the original key (fwd direction).
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#include "flow_table.h"
#include "flow.h"
#include "aegis_features.h"
#include "exporter.h"
#include "utils.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>

/* =========================================================================
 * Internal declarations (forward) from flow.c
 * ========================================================================= */
void flow_record_init(FlowRecord *record,
                      const FlowKey *key,
                      const struct timeval *first_ts);
void flow_key_reverse(FlowKey *rev, const FlowKey *key);

/* =========================================================================
 * FlowTable structure
 * ========================================================================= */
struct FlowTable {
    FlowRecord      *table;          /**< uthash head pointer (NULL = empty) */
    FlowTableConfig  cfg;            /**< Copied configuration */
    FlowTableStats   stats;          /**< Cumulative statistics */

    pthread_mutex_t  mutex;          /**< Optional mutex */
    int              mutex_init;     /**< 1 if mutex was successfully initialised */
};

/* =========================================================================
 * Internal helpers
 * ========================================================================= */

static void lock(FlowTable *ft) {
    if (ft->cfg.thread_safe) pthread_mutex_lock(&ft->mutex);
}

static void unlock(FlowTable *ft) {
    if (ft->cfg.thread_safe) pthread_mutex_unlock(&ft->mutex);
}

/** Close a flow, call the export callback, and free the record. */
static void close_and_export(FlowTable *ft, FlowRecord *rec) {
    /* Remove from hash table first so the callback cannot re-trigger */
    HASH_DEL(ft->table, rec);

    if (ft->cfg.on_flow_close) {
        ft->cfg.on_flow_close(rec, &rec->key, ft->cfg.on_flow_close_ctx);
    }

    free(rec);
    ft->stats.flows_active = HASH_COUNT(ft->table);
}

/* =========================================================================
 * Public API
 * ========================================================================= */

FlowTable *flow_table_create(const FlowTableConfig *cfg) {
    FlowTable *ft = xcalloc(1, sizeof(FlowTable));
    ft->table = NULL;
    ft->cfg   = *cfg;

    /* Apply defaults */
    if (ft->cfg.tcp_timeout_sec <= 0) ft->cfg.tcp_timeout_sec = FLOW_TIMEOUT_TCP_SEC;
    if (ft->cfg.udp_timeout_sec <= 0) ft->cfg.udp_timeout_sec = FLOW_TIMEOUT_UDP_SEC;

    /* Initialise mutex if requested */
    if (cfg->thread_safe) {
        if (pthread_mutex_init(&ft->mutex, NULL) == 0) {
            ft->mutex_init = 1;
        } else {
            LOG_WARN("Failed to initialise flow table mutex — proceeding without locking");
            ft->cfg.thread_safe = 0;
        }
    }

    LOG_INFO("Flow table created (tcp_timeout=%ds, udp_timeout=%ds, max_flows=%u, thread_safe=%d)",
             ft->cfg.tcp_timeout_sec, ft->cfg.udp_timeout_sec,
             ft->cfg.max_flows, ft->cfg.thread_safe);

    return ft;
}

void flow_table_update(FlowTable *ft, const PacketInfo *pkt) {
    lock(ft);

    FlowRecord *rec = NULL;
    PacketInfo *mutable_pkt = (PacketInfo *)pkt;

    /* ── Step 1: Look up by exact (forward) key ── */
    HASH_FIND(hh, ft->table, &pkt->key, sizeof(FlowKey), rec);
    if (rec) {
        mutable_pkt->is_fwd = 1;
    }

    /* ── Step 2: Look up by reversed key (reply direction) ── */
    if (!rec) {
        FlowKey rev;
        flow_key_reverse(&rev, &pkt->key);
        HASH_FIND(hh, ft->table, &rev, sizeof(FlowKey), rec);
        if (rec) {
            mutable_pkt->is_fwd = 0;
        }
    }

    /* ── Step 3: Create new flow if not found ── */
    if (!rec) {
        /* Enforce max_flows limit */
        if (ft->cfg.max_flows > 0 &&
            HASH_COUNT(ft->table) >= ft->cfg.max_flows) {
            LOG_WARN("Flow table full (max=%u) — dropping packet", ft->cfg.max_flows);
            unlock(ft);
            return;
        }

        rec = xmalloc(sizeof(FlowRecord));
        flow_record_init(rec, &pkt->key, &pkt->timestamp);

        HASH_ADD(hh, ft->table, key, sizeof(FlowKey), rec);

        ft->stats.flows_created++;
        ft->stats.flows_active = HASH_COUNT(ft->table);

        LOG_DEBUG("New flow created (total active: %u)", ft->stats.flows_active);
        mutable_pkt->is_fwd = 1;
    }

    /* ── Step 4: Determine direction (fwd = key matches original key) ── */
    /* Handled above during lookup and insertion */

    /* ── Step 5: Update features ── */
    features_update(rec, pkt);
    ft->stats.packets_processed++;

    /* ── Step 6: Check for TCP FIN / RST termination ── */
    if (pkt->tcp_flags & (TCP_FLAG_FIN | TCP_FLAG_RST)) {
        LOG_DEBUG("Flow closed by TCP FIN/RST");
        ft->stats.flows_closed++;
        close_and_export(ft, rec);
        rec = NULL;
    }

    unlock(ft);
}

uint32_t flow_table_expire(FlowTable *ft, const struct timeval *now_in) {
    struct timeval now;
    if (now_in) {
        now = *now_in;
    } else {
        utils_now(&now);
    }

    lock(ft);

    uint32_t expired = 0;
    FlowRecord *rec, *tmp;

    HASH_ITER(hh, ft->table, rec, tmp) {
        int timeout_sec = (rec->protocol == PROTO_TCP)
                          ? ft->cfg.tcp_timeout_sec
                          : ft->cfg.udp_timeout_sec;

        int64_t idle_us = timeval_diff_usec(&rec->last_seen, &now);
        int64_t timeout_us = (int64_t)timeout_sec * 1000000LL;

        if (idle_us >= timeout_us) {
            LOG_DEBUG("Expiring flow (idle %.1fs > timeout %ds)",
                      (double)idle_us / 1e6, timeout_sec);
            ft->stats.flows_expired++;
            close_and_export(ft, rec);   /* rec freed inside */
            expired++;
        }
    }

    unlock(ft);
    return expired;
}

uint32_t flow_table_flush(FlowTable *ft) {
    lock(ft);

    uint32_t flushed = 0;
    FlowRecord *rec, *tmp;

    HASH_ITER(hh, ft->table, rec, tmp) {
        if (ft->cfg.on_flow_close) {
            ft->cfg.on_flow_close(rec, &rec->key, ft->cfg.on_flow_close_ctx);
        }
        HASH_DEL(ft->table, rec);
        free(rec);
        flushed++;
    }

    ft->stats.flows_active = 0;

    unlock(ft);

    LOG_INFO("Flushed %u remaining active flows", flushed);
    return flushed;
}

const FlowRecord *flow_table_lookup(const FlowTable *ft, const FlowKey *key) {
    FlowRecord *rec = NULL;
    /* NOTE: cast away const for HASH_FIND (uthash limitation) */
    HASH_FIND(hh, ((FlowTable*)ft)->table, key, sizeof(FlowKey), rec);
    return rec;
}

void flow_table_get_stats(const FlowTable *ft, FlowTableStats *stats) {
    *stats = ft->stats;
}

uint32_t flow_table_count(const FlowTable *ft) {
    return (uint32_t)HASH_COUNT(ft->table);
}

void flow_table_destroy(FlowTable *ft) {
    if (!ft) return;

    /* Free all remaining records WITHOUT calling export callbacks */
    FlowRecord *rec, *tmp;
    HASH_ITER(hh, ft->table, rec, tmp) {
        HASH_DEL(ft->table, rec);
        free(rec);
    }

    if (ft->mutex_init) {
        pthread_mutex_destroy(&ft->mutex);
    }

    free(ft);
    LOG_DEBUG("Flow table destroyed");
}
