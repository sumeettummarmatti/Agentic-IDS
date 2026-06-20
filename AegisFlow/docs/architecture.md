# AegisFlow Architecture

> High-Performance CICFlowMeter-Compatible Flow Feature Extraction Engine

---

## System Architecture

```
                        ┌────────────────────────────────────┐
                        │        AegisFlow Engine             │
                        │                                     │
  Network Interface ───►│  ┌──────────┐   ┌──────────────┐  │
  or PCAP File          │  │  Packet  │──►│   Flow       │  │
                        │  │  Capture │   │   Manager    │  │
                        │  │(libpcap) │   │  (uthash)    │  │
                        │  └──────────┘   └──────┬───────┘  │
                        │                        │           │
                        │               ┌────────▼────────┐  │
                        │               │  Feature Engine  │  │
                        │               │  (Welford O(1)) │  │
                        │               └────────┬────────┘  │
                        │                        │           │
                        │               ┌────────▼────────┐  │
                        │               │  Flow Exporter  │  │
                        │               │ JSON │ CSV │ ... │  │
                        └───────────────┴───────────────────┘
                                               │
                            ┌──────────────────┼──────────────────┐
                            ▼                  ▼                  ▼
                         stdout /          flows.csv          (future)
                         JSON file                           Kafka topic
```

---

## Component Descriptions

### 1. Packet Capture (`src/capture/capture.c`)

| Aspect | Detail |
|---|---|
| Library | `libpcap` |
| Modes | Live interface, offline PCAP/PCAPNG |
| Parser | Ethernet II → IPv4 → TCP/UDP |
| Output | `PacketInfo` struct per packet |
| Non-IP | Skipped with counter increment |
| Stop | Atomic flag (safe from signal handlers) |

**Key design choices:**
- `pcap_dispatch()` with batch size 256 for maximum throughput
- BPF filter applied at kernel level (zero-copy filter in kernel)
- `atomic_int stop_requested` allows clean shutdown from `SIGINT`

### 2. Flow Manager (`src/flow/flow_table.c`)

| Aspect | Detail |
|---|---|
| Data structure | uthash hash table (O(1) avg insert/lookup) |
| Key | 5-tuple: src_ip, dst_ip, src_port, dst_port, protocol |
| Bidirectionality | First tries exact key, then reversed key |
| Thread safety | Optional `pthread_mutex_t` wrapper |
| Max flows | Configurable hard cap (0 = unlimited) |

**Bidirectional flow detection:**
```
Packet arrives with key K:
  1. HASH_FIND(K)     → found  → it's a forward packet
  2. HASH_FIND(rev(K))→ found  → it's a backward packet  
  3. Not found        → create new flow keyed on K
```

This matches CICFlowMeter's behaviour: one bidirectional `FlowRecord` per 5-tuple, created on the first packet's direction.

### 3. Feature Engine (`src/features/features.c`)

All statistics are computed **O(1) per packet** using **Welford's online algorithm**. No packets are ever stored.

#### Welford's Algorithm

```
Initialize: count=0, mean=0, M2=0

Update(x):
  count  += 1
  delta   = x - mean
  mean   += delta / count
  delta2  = x - mean
  M2     += delta * delta2

Finalize:
  variance = M2 / count          (population variance)
  std_dev  = sqrt(variance)
```

**Why population variance (÷N)?** CICFlowMeter uses population variance for all statistics. This matches the CICIDS2017 dataset column values.

#### Per-Packet Update (O(1))

For each arriving packet:
1. Update direction-specific packet count + byte count
2. Update direction `len_stats` (Welford) and min/max
3. Compute IAT from `last_pkt_time`, update direction `iat_stats`
4. Update global `pkt_len_stats` and `iat_stats`
5. Increment TCP flag counters

#### Feature Extraction (at export time)

Called once when a flow closes. Calls `welford_finalize()` for all stats and computes rate features from raw counters + duration.

### 4. Flow Exporter (`src/exporter/`)

#### Generic Interface

```c
typedef void (*ExportFn)(const FlowRecord *record,
                         const FlowKey    *key,
                         void             *ctx);
```

Any new output backend (Kafka, gRPC, shared memory) implements this signature and is registered in an `ExporterChain`.

#### JSON Exporter

- Uses `cJSON` for safe, correct JSON serialisation
- `NaN` and `±Inf` values replaced with `0.0` for safety
- Field names match CICFlowMeter naming convention
- Output: one JSON object per line (NDJSON)

#### CSV Exporter

- Column order matches CICFlowMeter default
- 44 columns total — directly usable as ML feature input
- `csv_exporter_write_header()` writes the header once
- Each `csv_exporter_write()` appends one data row

---

## Flow Lifecycle

```
Packet 1 (SYN) ──────────────────► CREATE FlowRecord
                                        │
Packets 2-N (data/ACK) ─────────────► UPDATE features (O(1))
                                        │
                           ┌────────────┴──────────────┐
                           │                            │
                      FIN or RST                   Idle timeout
                     observed                  (120s TCP / 60s UDP)
                           │                            │
                           └────────────┬───────────────┘
                                        │
                                  EXPORT features
                                  DELETE FlowRecord
```

---

## Data Structures

### FlowKey (16 bytes, cache-line friendly)

```c
typedef struct {
    uint32_t src_ip;    // 4 bytes
    uint32_t dst_ip;    // 4 bytes
    uint16_t src_port;  // 2 bytes
    uint16_t dst_port;  // 2 bytes
    uint8_t  protocol;  // 1 byte
    uint8_t  _pad[3];   // 3 bytes (explicit — no hidden padding)
} FlowKey;              // = 16 bytes total
```

`_pad` is explicitly zeroed so `memcmp`-based hashing is deterministic.

### WelfordState (24 bytes)

```c
typedef struct {
    uint64_t count;  // 8 bytes
    double   mean;   // 8 bytes
    double   M2;     // 8 bytes
} WelfordState;     // = 24 bytes
```

### FlowRecord (~600 bytes)

Contains 2× `DirectionStats` + 2× global `WelfordState` + TCP flag counters + timestamps + uthash handle.

---

## Performance Analysis

| Operation | Complexity | Notes |
|---|---|---|
| Flow lookup | O(1) avg | uthash with MurmurHash |
| Feature update | O(1) | Welford + simple arithmetic |
| Feature extraction | O(1) | Single pass through state |
| Flow expiry scan | O(N) | Linear scan, done infrequently |
| JSON serialisation | O(F) | F = number of features (~44) |
| CSV serialisation | O(F) | F = number of features (~44) |

**Memory per flow:** ~600 bytes for `FlowRecord`. At 100,000 concurrent flows: ~60 MB.

---

## Feature List (44 columns, CICFlowMeter-compatible)

| # | Feature Name | Description |
|---|---|---|
| 1 | `src_ip` | Source IPv4 address |
| 2 | `dst_ip` | Destination IPv4 address |
| 3 | `src_port` | Source port |
| 4 | `dst_port` | Destination port |
| 5 | `protocol` | IP protocol (6=TCP, 17=UDP) |
| 6 | `flow_duration` | Duration in microseconds |
| 7 | `total_fwd_packets` | Fwd packet count |
| 8 | `total_bwd_packets` | Bwd packet count |
| 9 | `total_packets` | Combined count |
| 10 | `total_length_fwd_pkts` | Fwd total bytes |
| 11 | `total_length_bwd_pkts` | Bwd total bytes |
| 12 | `pkt_length_min` | Min payload size |
| 13 | `pkt_length_max` | Max payload size |
| 14 | `pkt_length_mean` | Mean payload size |
| 15 | `pkt_length_std` | Std dev payload size |
| 16 | `flow_bytes_per_sec` | Bytes/sec rate |
| 17 | `flow_pkts_per_sec` | Packets/sec rate |
| 18 | `syn_flag_count` | TCP SYN count |
| 19 | `ack_flag_count` | TCP ACK count |
| 20 | `rst_flag_count` | TCP RST count |
| 21 | `fin_flag_count` | TCP FIN count |
| 22 | `psh_flag_count` | TCP PSH count |
| 23 | `urg_flag_count` | TCP URG count |
| 24 | `flow_iat_mean` | Mean inter-arrival time |
| 25 | `flow_iat_std` | Std dev IAT |
| 26 | `flow_iat_min` | Min IAT |
| 27 | `flow_iat_max` | Max IAT |
| 28 | `fwd_pkt_length_mean` | Fwd mean packet length |
| 29 | `fwd_pkt_length_std` | Fwd std packet length |
| 30 | `fwd_pkt_length_min` | Fwd min packet length |
| 31 | `fwd_pkt_length_max` | Fwd max packet length |
| 32 | `bwd_pkt_length_mean` | Bwd mean packet length |
| 33 | `bwd_pkt_length_std` | Bwd std packet length |
| 34 | `bwd_pkt_length_min` | Bwd min packet length |
| 35 | `bwd_pkt_length_max` | Bwd max packet length |
| 36 | `fwd_iat_mean` | Fwd mean IAT |
| 37 | `fwd_iat_std` | Fwd std IAT |
| 38 | `fwd_iat_min` | Fwd min IAT |
| 39 | `fwd_iat_max` | Fwd max IAT |
| 40 | `bwd_iat_mean` | Bwd mean IAT |
| 41 | `bwd_iat_std` | Bwd std IAT |
| 42 | `bwd_iat_min` | Bwd min IAT |
| 43 | `bwd_iat_max` | Bwd max IAT |

---

## Future Integration Points

### Kafka Producer

Implement `ExportFn` backed by a `librdkafka` producer:

```c
void kafka_exporter(const FlowRecord *r, const FlowKey *k, void *ctx) {
    KafkaHandle *h = (KafkaHandle *)ctx;
    char *json = json_exporter_serialize(r, k);
    rd_kafka_produce(h->topic, ..., json, strlen(json), ...);
    free(json);
}
// Register:
exporter_chain_add(&chain, kafka_exporter, &kafka_handle);
```

### XGBoost / ML Pipeline Integration

The `FlowFeatures` struct maps directly to a feature vector:

```c
FlowFeatures f;
features_extract(record, &f);
// f is now a 44-element feature vector suitable for:
//   - XGBoost inference
//   - ONNX Runtime
//   - TensorFlow Lite
```

### Multi-threaded Capture

Enable thread safety in `FlowTableConfig`:
```c
FlowTableConfig cfg = { .thread_safe = 1 };
```

Then run multiple `pcap_dispatch()` threads each calling `flow_table_update()`.

---

## Build Requirements

| Requirement | Version | Notes |
|---|---|---|
| GCC or Clang | GCC ≥ 9, Clang ≥ 10 | C17 support required |
| CMake | ≥ 3.16 | FetchContent for uthash/cJSON |
| libpcap | ≥ 1.9 | `libpcap-dev` on Ubuntu |
| pthreads | POSIX | For optional thread safety |
| Internet | Required at first build | FetchContent downloads uthash + cJSON |

### Ubuntu / WSL2 Quick Setup

```bash
sudo apt update
sudo apt install -y build-essential cmake libpcap-dev git

cd AegisFlow
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON
cmake --build build -j$(nproc)

# Run tests
cd build && ctest -V

# Run benchmark
./bench_flow 1000000 10000

# Live capture (requires root)
sudo ./aegisflow -i eth0 -j - -c flows.csv
```
