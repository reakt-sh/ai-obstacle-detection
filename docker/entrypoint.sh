#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

if [ -z "${INPUT_URL:-}" ]; then
    echo "INPUT_URL is required." >&2
    exit 2
fi

mkdir -p "${EXPORT_DIR:-/models/exports}"

requested_rtp_buffer="${RTP_BUFFER_BYTES:-4194304}"
if ! [[ "$requested_rtp_buffer" =~ ^[0-9]+$ ]] || [ "$requested_rtp_buffer" -le 0 ]; then
    echo "RTP_BUFFER_BYTES must be an integer > 0." >&2
    exit 2
fi
host_rmem_max="$(cat /proc/sys/net/core/rmem_max 2>/dev/null || echo 212992)"
effective_rtp_buffer="$requested_rtp_buffer"
if [ "$host_rmem_max" -lt "$requested_rtp_buffer" ]; then
    effective_rtp_buffer="$host_rmem_max"
    echo "Warning: host net.core.rmem_max=${host_rmem_max}; using RTP buffer ${effective_rtp_buffer} instead of ${requested_rtp_buffer}." >&2
fi
sed -i "s/^udpReadBufferSize:.*/udpReadBufferSize: ${effective_rtp_buffer}/" /etc/mediamtx.yml

echo "ReaktRailAi configuration:"
echo "  platform=${RAIL_AI_PLATFORM:-auto}"
echo "  device=${DEVICE:-auto} fallback=${DEVICE_FALLBACK:-cpu}"
echo "  model=${MODEL_FORMAT:-pytorch}/${MODEL_PRECISION:-fp32}"
echo "  input=rtsp ${INPUT_URL}"
echo "  output=WHEP/WebRTC encoder=${WEBRTC_ENCODER:-auto}"
echo "  rtp_buffer=${effective_rtp_buffer}"

/usr/local/bin/mediamtx /etc/mediamtx.yml &
mediamtx_pid=$!

cleanup() {
    kill -TERM "${ai_pid:-}" "${mediamtx_pid:-}" 2>/dev/null || true
    wait "${ai_pid:-}" "${mediamtx_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

mediamtx_ready=0
for _attempt in $(seq 1 50); do
    if (exec 3<>/dev/tcp/127.0.0.1/8889) 2>/dev/null; then
        exec 3>&-
        mediamtx_ready=1
        break
    fi
    if ! kill -0 "$mediamtx_pid" 2>/dev/null; then
        echo "MediaMTX terminated during startup." >&2
        exit 1
    fi
    sleep 0.1
done
if [ "$mediamtx_ready" -ne 1 ]; then
    echo "MediaMTX did not open WebRTC port 8889 within 5 seconds." >&2
    exit 1
fi

python3 -m rail_ai &
ai_pid=$!

set +e
wait -n "$ai_pid" "$mediamtx_pid"
status=$?
set -e
exit "$status"
