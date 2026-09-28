#!/bin/sh
set -eu

export WEBUI_AI_HOST="${WEBUI_AI_HOST:-}"
export WEBUI_AI_API_PORT="${WEBUI_AI_API_PORT:-5000}"
export WEBUI_AI_WHEP_PORT="${WEBUI_AI_WHEP_PORT:-8889}"
export WEBUI_AI_API_SCHEME="${WEBUI_AI_API_SCHEME:-http}"
export WEBUI_AI_WHEP_SCHEME="${WEBUI_AI_WHEP_SCHEME:-http}"
export WEBUI_STATUS_INTERVAL_MS="${WEBUI_STATUS_INTERVAL_MS:-1000}"
export WEBUI_DEBUG="${WEBUI_DEBUG:-0}"

envsubst \
  '${WEBUI_AI_HOST} ${WEBUI_AI_API_PORT} ${WEBUI_AI_WHEP_PORT} ${WEBUI_AI_API_SCHEME} ${WEBUI_AI_WHEP_SCHEME} ${WEBUI_STATUS_INTERVAL_MS} ${WEBUI_DEBUG}' \
  < /usr/share/nginx/html/config.js.template \
  > /usr/share/nginx/html/config.js

exec nginx -g 'daemon off;'
