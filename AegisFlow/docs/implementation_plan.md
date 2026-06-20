# AegisFlow — Implementation Plan

## Overview

AegisFlow is a production-grade, low-latency network flow feature extraction engine written in C17, designed as a CICFlowMeter-compatible replacement for real-time IDS/IPS integration. It captures live traffic via libpcap, tracks 5-tuple flows in a hash table (uthash), computes features incrementally using Welford's algorithm, and exports complete flow feature vectors as JSON or CSV — with a clean interface for future Kafka integration.

---

## Architecture Overview

```
libpcap → Packet Capture Layer → Flow Manager (uthash)
           ↓                         ↓
      Packet Parser            Feature Engine (incremental / Welford)
                                     ↓
                             Flow Exporter (JSON / CSV)
                                     ↓
                           stdout / file / socket (future Kafka)
```

---

## Proposed Project Structure

```
aegisflow/
├── include/
│   ├── aegisflow.h          # Top-level public API
│   ├── capture.h            # Packet capture interface
│   ├── flow.h               # FlowKey + FlowRecord definitions
│   ├── flow_table.h         # Hash table API
│   ├── features.h           # Feature computation interface
│   ├── exporter.h           # Export interface (JSON/CSV)
│   └── utils.h              # Logging, timers, misc utils
├── src/
│   ├── capture/
│   │   └── capture.c        # libpcap packet capture loop
│   ├── flow/
│   │   ├── flow.c           # FlowRecord lifecycle management
│   │   └── flow_table.c     # uthash-based flow table ops
│   ├── features/
│   │   └── features.c       # Incremental stat updates (Welford)
│   ├── exporter/
│   │   ├── json_exporter.c  # cJSON-based JSON export
│   │   └── csv_exporter.c   # CSV line export
│   └── utils/
│       └── utils.c          # Logging, timestamp helpers
├── tests/
│   ├── test_flow.c          # Unit tests for flow lifecycle
│   ├── test_features.c      # Unit tests for Welford / stats
│   └── test_exporter.c      # Unit tests for JSON/CSV output
├── examples/
│   ├── live_capture.c       # Full live capture demo
│   └── pcap_replay.c        # Offline PCAP replay demo
├── cmake/
│   └── FindLibpcap.cmake    # Optional CMake find module
├── docs/
│   └── architecture.md      # Architecture documentation
├── benchmarks/
│   └── bench_flow.c         # Performance benchmarking
├── CMakeLists.txt           # Root CMake build file
└── README.md
```

---

## Component Design

### 1. `include/flow.h` — Core Data Structures

**FlowKey** (5-tuple, used as uthash key):
```c
typedef struct {
    uint32_t src_ip, dst_ip;
    uint16_t src_port, dst_port;
    uint8_t  protocol;
} FlowKey;
```

**WelfordState** (streaming mean + variance):
```c
typedef struct {
    uint64_t count;
    double   mean;
    double   M2;     // sum of squared deviations
} WelfordState;
```

**FlowRecord** (all per-flow state — no packet storage):
- Timestamps: `first_seen`, `last_seen` (struct timeval)
- Fwd/Bwd: packet count, byte count, IAT WelfordState, length WelfordState
- Global: packet length WelfordState, IAT WelfordState
- TCP flags: syn/ack/rst/fin/psh/urg counters
- Min/max: packet length, IAT
- uthash handle

### 2. `src/capture/capture.c` — Packet Capture

- Initialize libpcap on interface or pcap file
- BPF filter support (e.g., `"tcp or udp"`)
- Callback-based packet dispatch: `pcap_loop()` / `pcap_dispatch()`
- Parse Ethernet → IP → TCP/UDP headers
- Extract 5-tuple + payload length + TCP flags + timestamp
- Normalize direction (bidirectional flows use canonical key ordering)
- Call `flow_table_update()` per packet

### 3. `src/flow/flow_table.c` — Flow Table

- uthash-based O(1) insert/lookup/delete
- `flow_table_update(key, pkt_info)`:
  - Lookup or create FlowRecord
  - Update features incrementally
  - Check for FIN/RST → close flow
- `flow_table_expire(timeout_tcp, timeout_udp)`:
  - Iterate all flows, close timed-out ones
  - Called periodically from capture loop
- Thread-safe lock (pthread mutex) wrapping the hash table

### 4. `src/features/features.c` — Feature Engine

All computation is **O(1) per packet**, no packet buffering:

- `welford_update(state, value)` — online mean/variance
- `welford_finalize(state, &mean, &variance, &std)` — O(1) finalization
- `features_update_packet(record, pkt_info, direction)`:
  - Update direction-specific counters
  - Update IAT for flow + direction (using `last_seen`)
  - Update length stats for flow + direction
  - Accumulate TCP flag counts

### 5. `src/exporter/` — Exporters

**JSON Exporter** (cJSON):
- `json_export_flow(record, key)` → `char*` (caller frees)
- Outputs all 30+ CICFlowMeter-compatible features
- Rate features computed at export time: `bytes/duration`, `pkts/duration`

**CSV Exporter**:
- Header line: column names
- `csv_export_flow(record, key, FILE*)` → writes one CSV line
- Compatible with CICFlowMeter CSV column order

**Exporter Interface** (`exporter.h`):
```c
typedef void (*ExportFn)(const FlowRecord*, const FlowKey*, void* ctx);
```
Future Kafka producer: implement `ExportFn` and swap in.

### 6. Flow Lifecycle

```
Packet arrives → lookup 5-tuple
  [not found] → create FlowRecord, set first_seen
  [found]     → update features
               → check: TCP FIN/RST? → export + delete
               → update last_seen

Periodic timer (e.g. every 1s) → scan for timed-out flows → export + delete
```

### 7. Utils

- Structured logging: `LOG_INFO`, `LOG_WARN`, `LOG_ERROR` macros
- Timestamp helpers: `timeval_diff_usec()`, `timeval_to_double()`
- Portable min/max macros
- Safe memory wrappers: `xmalloc`, `xcalloc`

---

## Unit Tests

| Test File | Coverage |
|---|---|
| `test_flow.c` | FlowRecord create/update/expire lifecycle |
| `test_features.c` | Welford correctness, IAT stats, TCP flags |
| `test_exporter.c` | JSON/CSV output correctness and field names |

Tests use a lightweight custom test harness (no external framework dependency).

---

## Benchmark Utility

`benchmarks/bench_flow.c`:
- Generates synthetic packet stream (100k–1M packets)
- Times per-packet processing through flow table + feature engine
- Reports throughput (pkt/sec), memory usage, and flow table saturation

---

## Build System (CMake)

- `CMakeLists.txt`: C17, finds libpcap, uthash, cJSON
- Separate targets: `aegisflow` (library), `aegisflow_capture` (executable), tests, benchmarks
- `cmake -DBUILD_TESTS=ON` to build tests
- `cmake -DBUILD_BENCHMARKS=ON` for benchmarks

---

## Documentation

`docs/architecture.md`:
- Full architecture walkthrough
- Data flow diagram
- Performance notes
- Integration guide for Kafka / XGBoost

---

## Verification Plan

### Automated Tests
```bash
cd build && cmake .. -DBUILD_TESTS=ON && make && ctest -V
```

### Manual Verification
- Replay a known PCAP file and compare output CSV columns with CICFlowMeter reference output
- Run live capture on loopback, send known traffic (curl/wget), verify features
- Run benchmark, confirm >100k pkt/sec on a modern host

---

## Open Questions

> [!IMPORTANT]
> **Canonical flow key direction**: Should the flow key always be normalized so that (src < dst) ensures bidirectionality, or should each 5-tuple direction produce separate flows (matching CICFlowMeter's behavior which produces one bidirectional flow)?
> **Recommendation**: Use CICFlowMeter's approach — create one flow keyed on the FIRST packet's direction, and classify subsequent packets as fwd or bwd based on whether they match the original direction.

> [!NOTE]
> **uthash bundling**: uthash is a header-only library. It will be bundled directly in `include/uthash.h` for ease of build. cJSON will also be vendored or found via CMake `find_package`.

> [!NOTE]
> **WSL2 note**: libpcap on WSL2 requires running as root or with `CAP_NET_RAW`. The build will work normally; live capture requires privilege escalation at runtime.
