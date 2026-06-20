/**
 * @file capture.c
 * @brief libpcap packet capture and Ethernet/IP/TCP/UDP parser.
 *
 * Supports both live interface capture and offline PCAP file replay.
 * Parsed packets are emitted as PacketInfo structs to a user-supplied
 * callback.  Only IPv4 TCP and UDP packets are currently handled;
 * all others are counted as skipped.
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#include "capture.h"
#include "utils.h"

#include <arpa/inet.h>
#include <net/ethernet.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <pcap/pcap.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

/* =========================================================================
 * Ethernet / IP header size constants
 * ========================================================================= */
#define ETHER_HDR_LEN  14    /* standard Ethernet II header */
#define IP_HDR_MIN_LEN 20    /* minimum IPv4 header (no options) */
#define TCP_HDR_MIN_LEN 20   /* minimum TCP header (no options) */
#define UDP_HDR_LEN    8     /* fixed-size UDP header */

/* =========================================================================
 * CaptureHandle — internal state
 * ========================================================================= */
struct CaptureHandle {
    pcap_t          *pcap;           /**< libpcap handle */
    CaptureConfig    cfg;            /**< Copied configuration */
    CaptureStats     stats;          /**< Accumulated counters */
    atomic_int       stop_requested; /**< Set to 1 by capture_stop() */
};

/* =========================================================================
 * Per-packet parse callback (called by pcap_dispatch)
 * ========================================================================= */
static void pcap_callback(u_char *user,
                          const struct pcap_pkthdr *hdr,
                          const u_char *pkt_data) {
    CaptureHandle *h = (CaptureHandle *)user;

    h->stats.pkts_received++;

    /* ── Validate minimum captured length ── */
    if (hdr->caplen < ETHER_HDR_LEN + IP_HDR_MIN_LEN) {
        h->stats.pkts_skipped++;
        return;
    }

    /* ── Skip Ethernet header ── */
    const u_char *ip_start = pkt_data + ETHER_HDR_LEN;

    /* Check EtherType: only process IPv4 (0x0800) */
    uint16_t ethertype = ntohs(*(const uint16_t *)(pkt_data + 12));
    if (ethertype != 0x0800) {
        /* Not IPv4 — skip (could extend for VLAN, IPv6 later) */
        h->stats.pkts_skipped++;
        return;
    }

    /* ── Parse IPv4 header ── */
    const struct ip *iph = (const struct ip *)ip_start;
    uint8_t  ip_version  = iph->ip_v;
    uint8_t  ip_hlen     = (uint8_t)(iph->ip_hl * 4u);  /* header length in bytes */
    uint16_t ip_total    = ntohs(iph->ip_len);
    uint8_t  protocol    = iph->ip_p;

    if (ip_version != 4 || ip_hlen < IP_HDR_MIN_LEN) {
        h->stats.pkts_skipped++;
        return;
    }

    /* ── Only process TCP, UDP, and ICMP ── */
    if (protocol != PROTO_TCP && protocol != PROTO_UDP && protocol != PROTO_ICMP) {
        h->stats.pkts_skipped++;
        return;
    }

    /* ── Ensure we have enough bytes for the transport header ── */
    uint32_t transport_offset = ETHER_HDR_LEN + ip_hlen;
    uint32_t min_transport_len = 0;
    if (protocol == PROTO_TCP) {
        min_transport_len = TCP_HDR_MIN_LEN;
    } else if (protocol == PROTO_UDP) {
        min_transport_len = UDP_HDR_LEN;
    } else if (protocol == PROTO_ICMP) {
        min_transport_len = 8;
    }

    if (hdr->caplen < transport_offset + min_transport_len) {
        h->stats.pkts_skipped++;
        return;
    }

    const u_char *transport = pkt_data + transport_offset;

    /* ── Build PacketInfo ── */
    PacketInfo pkt;
    memset(&pkt, 0, sizeof(pkt));

    /* Timestamps from pcap header */
    pkt.timestamp.tv_sec  = hdr->ts.tv_sec;
    pkt.timestamp.tv_usec = hdr->ts.tv_usec;

    /* IP addresses */
    pkt.key.src_ip   = iph->ip_src.s_addr;
    pkt.key.dst_ip   = iph->ip_dst.s_addr;
    pkt.key.protocol = protocol;

    if (protocol == PROTO_TCP) {
        const struct tcphdr *tcph = (const struct tcphdr *)transport;
        pkt.key.src_port  = tcph->th_sport;
        pkt.key.dst_port  = tcph->th_dport;
        pkt.tcp_flags     = tcph->th_flags & 0xFFu;

        uint32_t tcp_hlen = (uint32_t)(tcph->th_off * 4u);
        /* payload = IP total - IP header - TCP header */
        uint32_t ip_payload = (ip_total > ip_hlen) ? ip_total - ip_hlen : 0;
        pkt.payload_len  = (ip_payload > tcp_hlen) ? ip_payload - tcp_hlen : 0;

    } else if (protocol == PROTO_UDP) {
        const struct udphdr *udph = (const struct udphdr *)transport;
        pkt.key.src_port  = udph->uh_sport;
        pkt.key.dst_port  = udph->uh_dport;
        pkt.tcp_flags     = 0;
        /* UDP payload = total - IP hdr - UDP hdr */
        uint16_t udp_total = ntohs(udph->uh_ulen);
        pkt.payload_len  = (udp_total > UDP_HDR_LEN) ?
                           udp_total - UDP_HDR_LEN : 0;
    } else { /* ICMP */
        pkt.key.src_port  = 0;
        pkt.key.dst_port  = 0;
        pkt.tcp_flags     = 0;
        uint32_t ip_payload = (ip_total > ip_hlen) ? ip_total - ip_hlen : 0;
        pkt.payload_len  = (ip_payload > 8) ? ip_payload - 8 : 0;
    }

    /* is_fwd is always 1 here; the flow table determines direction by
     * trying both the exact key and its reverse. */
    pkt.is_fwd = 1;

    h->stats.pkts_parsed++;

    /* ── Invoke user callback ── */
    h->cfg.on_packet(&pkt, h->cfg.user);
}

/* =========================================================================
 * Public API
 * ========================================================================= */

CaptureHandle *capture_open(const CaptureConfig *cfg,
                            char *errbuf, size_t errbuflen) {
    if (!cfg || !cfg->on_packet) {
        snprintf(errbuf, errbuflen, "on_packet callback must be non-NULL");
        return NULL;
    }

    char pcap_errbuf[PCAP_ERRBUF_SIZE];
    pcap_t *pcap = NULL;

    if (cfg->pcap_file) {
        /* ── Offline PCAP replay ── */
        pcap = pcap_open_offline(cfg->pcap_file, pcap_errbuf);
        if (!pcap) {
            snprintf(errbuf, errbuflen, "pcap_open_offline('%s'): %s",
                     cfg->pcap_file, pcap_errbuf);
            return NULL;
        }
        LOG_INFO("Opened PCAP file: %s", cfg->pcap_file);

    } else if (cfg->device) {
        /* ── Live capture ── */
        int snaplen    = cfg->snaplen   > 0 ? cfg->snaplen   : 65535;
        int timeout_ms = cfg->timeout_ms > 0 ? cfg->timeout_ms : 1000;

        pcap = pcap_open_live(cfg->device, snaplen,
                              cfg->promisc, timeout_ms, pcap_errbuf);
        if (!pcap) {
            snprintf(errbuf, errbuflen, "pcap_open_live('%s'): %s",
                     cfg->device, pcap_errbuf);
            return NULL;
        }
        LOG_INFO("Opened live interface: %s (promisc=%d)", cfg->device, cfg->promisc);

    } else {
        snprintf(errbuf, errbuflen, "Either device or pcap_file must be specified");
        return NULL;
    }

    /* ── Apply BPF filter ── */
    if (cfg->bpf_filter && *cfg->bpf_filter) {
        struct bpf_program fp;
        if (pcap_compile(pcap, &fp, cfg->bpf_filter, 1, PCAP_NETMASK_UNKNOWN) < 0) {
            snprintf(errbuf, errbuflen, "pcap_compile('%s'): %s",
                     cfg->bpf_filter, pcap_geterr(pcap));
            pcap_close(pcap);
            return NULL;
        }
        if (pcap_setfilter(pcap, &fp) < 0) {
            snprintf(errbuf, errbuflen, "pcap_setfilter: %s", pcap_geterr(pcap));
            pcap_freecode(&fp);
            pcap_close(pcap);
            return NULL;
        }
        pcap_freecode(&fp);
        LOG_INFO("BPF filter applied: %s", cfg->bpf_filter);
    }

    /* ── Build handle ── */
    CaptureHandle *h = xcalloc(1, sizeof(CaptureHandle));
    h->pcap = pcap;
    h->cfg  = *cfg;
    atomic_store(&h->stop_requested, 0);

    return h;
}

int capture_run(CaptureHandle *handle, uint64_t max_packets) {
    if (!handle) return -1;

    LOG_INFO("Starting capture loop (max_packets=%" PRIu64 ")", max_packets);

    uint64_t dispatched = 0;
    const int BATCH = 256;

    while (!atomic_load(&handle->stop_requested)) {
        int count = pcap_dispatch(handle->pcap, BATCH, pcap_callback,
                                  (u_char *)handle);

        if (count == 0) {
            /* Timeout / no packets; check for offline EOF */
            if (handle->cfg.pcap_file) {
                LOG_INFO("End of PCAP file reached");
                break;
            }
            continue;
        }

        if (count < 0) {
            if (count == PCAP_ERROR_BREAK) {
                /* pcap_breakloop() was called */
                break;
            }
            LOG_ERROR("pcap_dispatch error: %s", pcap_geterr(handle->pcap));
            return -1;
        }

        dispatched += (uint64_t)count;

        if (max_packets > 0 && dispatched >= max_packets) {
            LOG_INFO("Reached max_packets limit (%" PRIu64 ")", max_packets);
            break;
        }
    }

    /* Retrieve kernel drop stats for live captures */
    if (!handle->cfg.pcap_file) {
        struct pcap_stat ps;
        if (pcap_stats(handle->pcap, &ps) == 0) {
            handle->stats.pkts_dropped = ps.ps_drop;
        }
    }

    LOG_INFO("Capture loop ended (parsed=%" PRIu64 ", skipped=%" PRIu64 ", dropped=%" PRIu64 ")",
             handle->stats.pkts_parsed, handle->stats.pkts_skipped,
             handle->stats.pkts_dropped);

    return 0;
}

void capture_stop(CaptureHandle *handle) {
    if (!handle) return;
    atomic_store(&handle->stop_requested, 1);
    pcap_breakloop(handle->pcap);
}

void capture_get_stats(const CaptureHandle *handle, CaptureStats *stats) {
    if (handle && stats) {
        *stats = handle->stats;
    }
}

void capture_close(CaptureHandle *handle) {
    if (!handle) return;
    if (handle->pcap) {
        pcap_close(handle->pcap);
    }
    free(handle);
    LOG_DEBUG("Capture handle closed");
}

int capture_list_devices(void) {
    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_if_t *devs = NULL;

    if (pcap_findalldevs(&devs, errbuf) < 0) {
        fprintf(stderr, "pcap_findalldevs: %s\n", errbuf);
        return -1;
    }

    printf("Available network interfaces:\n");
    int idx = 0;
    for (pcap_if_t *d = devs; d; d = d->next, idx++) {
        printf("  [%d] %s", idx, d->name);
        if (d->description) printf(" (%s)", d->description);
        if (d->flags & PCAP_IF_LOOPBACK) printf(" [LOOPBACK]");
        if (d->flags & PCAP_IF_UP)       printf(" [UP]");
        if (d->flags & PCAP_IF_RUNNING)  printf(" [RUNNING]");
        putchar('\n');
    }

    pcap_freealldevs(devs);
    return 0;
}
