# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY nobitex_trader ./nobitex_trader
RUN pip install --no-cache-dir --prefix=/install ".[socks]"

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Tehran \
    NBX_CONFIG=/app/config.yaml \
    NBX_DATA_DIR=/app/data \
    NBX_DASHBOARD__HOST=0.0.0.0
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 trader \
    && mkdir -p /app/data && chown trader:trader /app/data
COPY --from=build /install /usr/local
WORKDIR /app
USER trader
VOLUME ["/app/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["nobitex-trader", "healthcheck"]
ENTRYPOINT ["nobitex-trader"]
CMD ["run"]
