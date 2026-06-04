#!/usr/bin/env bash
set -euo pipefail

if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Copy .env.example to .env and configure it first."
    exit 1
fi

set -a
source .env
set +a

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
PORT="${PORT:-8000}"
BASE_URL="http://localhost:${PORT}"

echo "[1/4] git pull --ff-only origin main"
git pull --ff-only origin main

echo "[2/4] docker compose up --build"
docker compose -f "$COMPOSE_FILE" up -d --build

echo "[3/4] health check (max 30s)"
for i in $(seq 1 30); do
    if curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
        echo "Service ready at attempt $i"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: health check failed after 30s"
        docker compose -f "$COMPOSE_FILE" logs ml-api
        exit 1
    fi
    sleep 1
done

echo "[4/4] cleanup dangling images"
docker image prune -f --filter "dangling=true" || true

echo "Deploy complete: $(curl -s "$BASE_URL/health")"
