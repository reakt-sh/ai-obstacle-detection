FROM nginx:1.29-alpine

RUN apk add --no-cache gettext

COPY webui/nginx.conf /etc/nginx/conf.d/default.conf
COPY webui/index.html /usr/share/nginx/html/index.html
COPY webui/styles.css /usr/share/nginx/html/styles.css
COPY webui/app.js /usr/share/nginx/html/app.js
COPY webui/assets /usr/share/nginx/html/assets
COPY webui/config.js.template /usr/share/nginx/html/config.js.template
COPY webui/entrypoint.sh /usr/local/bin/reaktrailai-webui-entrypoint

RUN chmod +x /usr/local/bin/reaktrailai-webui-entrypoint

EXPOSE 8080 8081 8082 8083

HEALTHCHECK --interval=10s --timeout=3s --start-period=3s --retries=3 \
  CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1

ENTRYPOINT ["/usr/local/bin/reaktrailai-webui-entrypoint"]
