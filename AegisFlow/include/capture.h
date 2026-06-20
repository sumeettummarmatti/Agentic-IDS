/**
 * @file capture.h
 * @brief Packet capture interface for AegisFlow.
 *
 * Provides an abstraction over libpcap for both live interface capture
 * and offline PCAP file replay.  Parsed packets are dispatched to a
 * user-supplied callback as PacketInfo structs.
 *
 * AegisFlow — High-Performance CICFlowMeter-Compatible Feature Engine
 * SPDX-License-Identifier: MIT
 */

#ifndef AEGISFLOW_CAPTURE_H
#define AEGISFLOW_CAPTURE_H

#include <stdint.h>
#include <stddef.h>

#include "flow.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Forward declarations
 * ========================================================================= */
struct CaptureHandle;
typedef struct CaptureHandle CaptureHandle;

/* =========================================================================
 * Packet callback
 *
 * Called once per successfully parsed packet.
 * @param pkt  Pointer to parsed packet info (valid only during callback).
 * @param user Opaque user data pointer supplied at open time.
 * ========================================================================= */
typedef void (*PacketCallback)(const PacketInfo *pkt, void *user);

/* =========================================================================
 * CaptureConfig — parameters for opening a capture session
 * ========================================================================= */
typedef struct {
    const char    *device;        /**< Interface name (e.g. "eth0") or NULL for pcap file */
    const char    *pcap_file;     /**< Path to offline .pcap / .pcapng file (or NULL) */
    const char    *bpf_filter;    /**< BPF filter string (NULL = no filter) */
    int            snaplen;       /**< Snapshot length in bytes (0 = default 65535) */
    int            promisc;       /**< Non-zero = promiscuous mode (live capture) */
    int            timeout_ms;    /**< Read timeout in milliseconds (0 = default 1000) */
    PacketCallback on_packet;     /**< Callback invoked per parsed packet (required) */
    void          *user;          /**< Opaque user data forwarded to on_packet */
} CaptureConfig;

/* =========================================================================
 * Capture statistics
 * ========================================================================= */
typedef struct {
    uint64_t pkts_received;   /**< Total packets received by libpcap */
    uint64_t pkts_dropped;    /**< Packets dropped by kernel (libpcap stat) */
    uint64_t pkts_parsed;     /**< Packets successfully parsed and dispatched */
    uint64_t pkts_skipped;    /**< Packets skipped (unsupported L3/L4 type) */
} CaptureStats;

/* =========================================================================
 * Public API
 * ========================================================================= */

/**
 * @brief Open a capture session (live or offline).
 *
 * Validates the config, opens a libpcap handle, applies the BPF filter,
 * and returns an opaque CaptureHandle.
 *
 * @param cfg     Non-NULL configuration struct.
 * @param errbuf  Buffer for human-readable error messages (≥ 256 bytes).
 * @return Heap-allocated handle on success, NULL on error (errbuf filled).
 */
CaptureHandle *capture_open(const CaptureConfig *cfg, char *errbuf, size_t errbuflen);

/**
 * @brief Run the capture loop.
 *
 * Calls pcap_dispatch() repeatedly, parsing each packet and invoking
 * cfg->on_packet for each valid IP/TCP/UDP frame.  Returns when:
 *   - An EOF is reached (offline file)
 *   - capture_stop() is called from a signal handler
 *   - An unrecoverable pcap error occurs
 *   - max_packets packets have been dispatched (0 = unlimited)
 *
 * @param handle        Non-NULL handle from capture_open().
 * @param max_packets   Stop after N packets (0 = run until stop/EOF).
 * @return 0 on clean exit, -1 on error.
 */
int capture_run(CaptureHandle *handle, uint64_t max_packets);

/**
 * @brief Request the capture loop to stop (safe to call from signal handlers).
 * @param handle Non-NULL handle.
 */
void capture_stop(CaptureHandle *handle);

/**
 * @brief Retrieve capture statistics.
 * @param handle Non-NULL handle.
 * @param stats  Output stats struct.
 */
void capture_get_stats(const CaptureHandle *handle, CaptureStats *stats);

/**
 * @brief Close and free the capture handle.
 *
 * Closes the libpcap device/file and frees all resources.
 * After this call the handle pointer is invalid.
 *
 * @param handle Handle to close (may be NULL — no-op).
 */
void capture_close(CaptureHandle *handle);

/**
 * @brief List available network interfaces (for informational purposes).
 *
 * Prints available interfaces to stdout.  Caller does not need to free anything.
 * @return 0 on success, -1 on error.
 */
int capture_list_devices(void);

#ifdef __cplusplus
}
#endif

#endif /* AEGISFLOW_CAPTURE_H */
