#!/usr/bin/env bash
set -euo pipefail

# --- Configuration & Helpers ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Automatically find the first json file in the current directory
LAYOUT_FILE=$(find "${SCRIPT_DIR}" -maxdepth 1 -name "*.json" -print -quit)
PORT=8765

# Check if a layout file exists
[[ -z "${LAYOUT_FILE}" ]] && { echo "❌ Error: No JSON layout file found in ${SCRIPT_DIR}"; exit 1; }

# Find an available port between 8765 and 8795
while ss -Htnl sport ":${PORT}" 2>/dev/null | grep -q . || netstat -tuln 2>/dev/null | grep -q ":${PORT} "; do
  ((PORT++))
  if [ $PORT -gt 8795 ]; then echo "❌ Error: No free ports found"; exit 1; fi
done

# --- Layout Encoding (JSON -> Gzip -> Base64Url) ---
# Uses Python standard library to avoid external dependencies like zstandard
ENCODED_LAYOUT=$(python3 -c "
import json, base64, gzip, sys
try:
    with open('${LAYOUT_FILE}', 'r') as f: data = json.load(f)
    # Compress JSON and strip Base64 padding for URL safety
    compressed = gzip.compress(json.dumps(data, separators=(',', ':')).encode())
    print(base64.urlsafe_b64encode(compressed).decode().rstrip('='))
except Exception as e: print(f'Error: {e}', file=sys.stderr); sys.exit(1)
")

# --- Execution ---
echo "▶ Starting foxglove_bridge on port ${PORT}"
# Launch bridge in background
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:="${PORT}" &
BRIDGE_PID=$!

# Cleanup on exit
trap "kill $BRIDGE_PID 2>/dev/null; echo -e '\n🛑 Stopped'; exit 0" EXIT INT TERM
sleep 2

# --- Generate URL & Open Browser ---
WS_URL_ENC=$(python3 -c "import urllib.parse; print(urllib.parse.quote('ws://127.0.0.1:${PORT}'))")
FOXGLOVE_URL="https://app.foxglove.dev/~/view?ds=foxglove-websocket&ds.url=${WS_URL_ENC}&layout=${ENCODED_LAYOUT}"

echo -e "✅ Success!\n🔗 ${FOXGLOVE_URL}"

# Open browser based on OS
if command -v xdg-open &>/dev/null; then xdg-open "${FOXGLOVE_URL}"
elif command -v open &>/dev/null; then open "${FOXGLOVE_URL}"
else echo "Please copy the link above to your browser"; fi

wait $BRIDGE_PID