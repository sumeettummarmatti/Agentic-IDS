#!/usr/bin/env bash
# build_aegisflow.sh
# ─────────────────────────────────────────────────────────────────────────────
# Builds the AegisFlow C binary with Kafka support (aegisflow_kafka).
#
# Usage:
#   ./build_aegisflow.sh             # standard build (no Kafka)
#   ./build_aegisflow.sh --kafka     # build with Kafka exporter
#   ./build_aegisflow.sh --clean     # wipe build dir and rebuild
#
# Prerequisites:
#   - cmake >= 3.16
#   - libpcap-dev    (brew install libpcap  |  apt install libpcap-dev)
#   - librdkafka-dev (brew install librdkafka | apt install librdkafka-dev)
#     [only needed with --kafka]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AEGISFLOW_DIR="$SCRIPT_DIR/AegisFlow"
BUILD_DIR="$AEGISFLOW_DIR/build"

KAFKA=OFF
CLEAN=OFF
BUILD_TYPE=Release

for arg in "$@"; do
  case "$arg" in
    --kafka)  KAFKA=ON   ;;
    --clean)  CLEAN=ON   ;;
    --debug)  BUILD_TYPE=Debug ;;
    --help|-h)
      echo "Usage: $0 [--kafka] [--clean] [--debug]"
      echo "  --kafka   Build aegisflow_kafka binary (requires librdkafka)"
      echo "  --clean   Wipe build dir before building"
      echo "  --debug   Debug build with symbols"
      exit 0
      ;;
  esac
done

# ── Install librdkafka if missing and --kafka was requested ──────────────────
if [[ "$KAFKA" == "ON" ]]; then
  if ! pkg-config --exists rdkafka 2>/dev/null && \
     ! [ -f /opt/homebrew/lib/librdkafka.dylib ] && \
     ! [ -f /usr/lib/x86_64-linux-gnu/librdkafka.so ]; then
    echo "⚠  librdkafka not found. Installing..."
    if command -v brew &>/dev/null; then
      brew install librdkafka
    elif command -v apt-get &>/dev/null; then
      sudo apt-get update && sudo apt-get install -y librdkafka-dev
    else
      echo "❌ Cannot auto-install librdkafka. Install it manually then retry."
      exit 1
    fi
  fi
fi

# ── Clean ────────────────────────────────────────────────────────────────────
if [[ "$CLEAN" == "ON" ]] && [[ -d "$BUILD_DIR" ]]; then
  echo "🧹 Cleaning $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

# ── Configure ────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "  AegisFlow Build"
echo "  Type:  $BUILD_TYPE"
echo "  Kafka: $KAFKA"
echo "══════════════════════════════════════════════"
echo ""

cmake -S "$AEGISFLOW_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DAEGISFLOW_KAFKA="$KAFKA" \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_TESTS=OFF

# ── Build ────────────────────────────────────────────────────────────────────
JOBS=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
cmake --build "$BUILD_DIR" -j"$JOBS"

# ── Report ───────────────────────────────────────────────────────────────────
echo ""
echo "✅ Build complete. Binaries:"
echo ""

if [[ -f "$BUILD_DIR/aegisflow" ]]; then
  echo "  $BUILD_DIR/aegisflow"
  echo "    Standard binary (file output only)"
  echo "    Usage: sudo $BUILD_DIR/aegisflow -i eth0 -j - -c flows.csv"
fi

if [[ -f "$BUILD_DIR/aegisflow_kafka" ]]; then
  echo ""
  echo "  $BUILD_DIR/aegisflow_kafka"
  echo "    Kafka-enabled binary (publishes flows to Kafka)"
  echo "    Usage: sudo $BUILD_DIR/aegisflow_kafka \\"
  echo "             -i eth0 \\"
  echo "             -b localhost:9092 \\"
  echo "             -T raw-flows \\"
  echo "             -s MUM-01 \\"
  echo "             -g asia-south1"
  echo ""
  echo "    Or with env vars:"
  echo "      export SERVER_ID=MUM-01"
  echo "      export GEO_REGION=asia-south1"
  echo "      export KAFKA_BOOTSTRAP_SERVERS=localhost:9092"
  echo "      sudo $BUILD_DIR/aegisflow_kafka -i eth0"
fi

echo ""
