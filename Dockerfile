FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_ROUTER_HOST=0.0.0.0 \
    MODEL_ROUTER_PORT=8080

RUN groupadd --system --gid 10001 modelrouter \
    && useradd --system --uid 10001 --gid modelrouter --home-dir /app modelrouter

WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir '.[production]' \
    && chown -R modelrouter:modelrouter /app

USER 10001:10001
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8080/healthz', timeout=2).read()"

CMD ["gunicorn", "--config", "deploy/gunicorn.conf.py", "model_router.api:create_app()"]
