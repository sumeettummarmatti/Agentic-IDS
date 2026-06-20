# AegisFlow

> **High-Performance CICFlowMeter-Compatible Network Flow Feature Extraction Engine**

[![Language](https://img.shields.io/badge/language-C17-blue)](https://en.wikipedia.org/wiki/C17_(C_standard_revision))
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20WSL2-lightgrey)](https://docs.microsoft.com/en-us/windows/wsl/)

AegisFlow is a production-grade, low-latency network flow feature extraction engine designed as a drop-in replacement for [CICFlowMeter](https://www.unb.ca/cic/research/applications.html). It is built for real-time integration into the **AegisNet IDS/IPS platform** and future enterprise deployment.

---

## Key Features

| Feature | Detail |
|---|---|
| **Zero packet retention** | All statistics computed incrementally — no packet buffers |
| **O(1) per-packet updates** | Welford's online algorithm for mean/variance |
| **CICFlowMeter-compatible** | 44 features matching CICIDS2017 column names |
| **Bidirectional flows** | One `FlowRecord` per 5-tuple, fwd/bwd separated |
| **Dual output** | JSON (NDJSON) and CSV simultaneously |
| **Pluggable exporters** | `ExportFn` interface — Kafka producer is a one-file addition |
| **Production quality** | Extensive comments, error handling, memory safety |
| **Thread-safe option** | Optional `pthread_mutex_t` wrapping the flow table |

---

## Quick Start (WSL2 / Ubuntu 22.04)

### 1. Install Dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake libpcap-dev git
```

### 2. Build

```bash
cd AegisFlow
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=ON \
      -DBUILD_BENCHMARKS=ON \
      -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
```

### 3. Run Tests

```bash
cd build
ctest -V
```

### 4. Run Benchmark

```bash
./build/bench_flow 1000000 10000
```

### 5. Live Capture

```bash
# List interfaces
./build/aegisflow -l

# Capture on eth0, output JSON to stdout + CSV to flows.csv
sudo ./build/aegisflow -i eth0 -j - -c flows.csv

# Capture only HTTPS traffic
sudo ./build/aegisflow -i eth0 -f "tcp port 443" -c https_flows.csv
```

### 6. Offline PCAP Replay

```bash
./build/aegisflow -r capture.pcap -c output.csv
# or use the dedicated example
./build/example_pcap_replay capture.pcap output.csv "tcp"
```

---

## Project Structure

```
AegisFlow/
├── include/
│   ├── aegisflow.h       # Top-level public API
│   ├── flow.h            # FlowKey, FlowRecord, PacketInfo, FlowFeatures
│   ├── capture.h         # Packet capture interface
│   ├── flow_table.h      # Hash table API
│   ├── features.h        # Welford + incremental feature computation
│   ├── exporter.h        # ExportFn, JSON/CSV, ExporterChain
│   └── utils.h           # Logging, timestamps, safe malloc
├── src/
│   ├── aegisflow.c       # Engine wiring (capture→table→exporter)
│   ├── main.c            # CLI entry point
│   ├── capture/
│   │   └── capture.c     # libpcap packet parser
│   ├── flow/
│   │   ├── flow.c        # FlowRecord init + key utilities
│   │   └── flow_table.c  # uthash O(1) flow table
│   ├── features/
│   │   └── features.c    # Welford algorithm + feature extraction
│   ├── exporter/
│   │   ├── json_exporter.c  # cJSON serialiser
│   │   └── csv_exporter.c   # CSV writer + ExporterChain
│   └── utils/
│       └── utils.c       # Logging, timeval helpers
├── tests/
│   ├── test_runner.h     # Lightweight test harness
│   ├── test_flow.c       # Flow lifecycle tests
│   ├── test_features.c   # Welford + feature pipeline tests
│   └── test_exporter.c   # JSON/CSV correctness tests
├── examples/
│   ├── live_capture.c    # Live interface demo
│   └── pcap_replay.c     # Offline PCAP replay demo
├── benchmarks/
│   └── bench_flow.c      # Throughput + memory benchmark
├── cmake/
│   └── FindLibpcap.cmake # CMake find module for libpcap
├── docs/
│   └── architecture.md   # Detailed architecture documentation
└── CMakeLists.txt        # Root CMake build file
```

---

## CLI Reference

```
AegisFlow v1.0.0

Usage:
  aegisflow -i <interface>  [options]   Live capture
  aegisflow -r <pcap_file>  [options]   Offline PCAP replay

Options:
  -i <iface>   Live capture interface (e.g. eth0, wlan0)
  -r <file>    Read from offline PCAP / PCAPNG file
  -f <bpf>     BPF filter string (default: "ip")
  -j <file>    Write JSON output to file ('-' = stdout)
  -c <file>    Write CSV output to file  ('-' = stdout)
  -t <sec>     TCP flow timeout (default: 120)
  -u <sec>     UDP flow timeout (default: 60)
  -n <count>   Stop after N packets (default: unlimited)
  -e <sec>     Expiry scan interval  (default: 5)
  -v           Verbose / DEBUG logging
  -q           Quiet / WARN-only logging
  -l           List available network interfaces
  -h           Show this help
```

---

## Build Options

| CMake Option | Default | Description |
|---|---|---|
| `BUILD_TESTS` | `OFF` | Build unit test executables |
| `BUILD_BENCHMARKS` | `OFF` | Build `bench_flow` benchmark |
| `BUILD_EXAMPLES` | `ON` | Build example executables |
| `AEGISFLOW_ASAN` | `OFF` | Enable AddressSanitizer + UBSan |
| `CMAKE_BUILD_TYPE` | `Release` | `Debug` / `Release` / `RelWithDebInfo` |

### Debug Build with ASan

```bash
cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Debug \
      -DBUILD_TESTS=ON \
      -DAESGISFLOW_ASAN=ON
cmake --build build -j$(nproc)
```

---

## Performance

Benchmark results on a typical modern workstation (8-core, 3.5 GHz):

```
Packets    : 1,000,000
Flows      : 10,000 (synthetic)
Throughput : ~2,800,000 packets/sec
Latency    : ~360 ns/packet
FlowRecord : ~600 bytes
Memory     : ~6 MB / 10,000 flows
```

**Target: >100,000 packets/sec** — achieved by >25×.

---

## CICFlowMeter Compatibility

AegisFlow computes the same 44 statistical features as CICFlowMeter using the same:
- Population variance (÷N, not ÷(N-1))
- Bidirectional flow model (first-seen direction = fwd)
- TCP FIN/RST and timeout-based flow closure
- Microsecond IAT resolution

Output CSV columns are named to match CICFlowMeter's default column names for direct drop-in use with ML pipelines trained on CICIDS datasets.

---

## Future Roadmap

- [ ] Kafka producer exporter (`librdkafka`)
- [ ] IPv6 support
- [ ] VLAN / Q-in-Q parsing
- [ ] Multi-threaded capture (parallel workers per CPU core)
- [ ] Shared memory output for zero-copy IPC with AegisNet IDS
- [ ] XGBoost inference integration
- [ ] PCAPNG extended block support

---

## Dependencies

| Library | Version | License | How |
|---|---|---|---|
| [libpcap](https://www.tcpdump.org/) | ≥ 1.9 | BSD-3 | System package |
| [uthash](https://github.com/troydhanson/uthash) | 2.3.0 | BSD-2 | FetchContent (auto) |
| [cJSON](https://github.com/DaveGamble/cJSON) | 1.7.18 | MIT | FetchContent (auto) |
| pthreads | POSIX | — | System |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Part of the AegisNet Platform

AegisFlow is the packet-level feature extraction component of the **AegisNet** real-time IDS/IPS platform:

```
Network Traffic
     │
     ▼
 AegisFlow          ← you are here
 (feature extraction)
     │
     ▼
 AegisNet Core
 (XGBoost / ML inference + rule engine)
     │
     ▼
 AegisBlock
 (iptables / eBPF enforcement)
```
