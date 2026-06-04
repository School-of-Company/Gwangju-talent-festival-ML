#!/usr/bin/env bash
set -euo pipefail

{
    if [ ! -f .env ]; then
        echo "ERROR: .env file not found. Copy .env.example to .env and configure it first."
        exit 1
    fi

    set -a
    source .env
    set +a

    LOG_LEVEL=$(echo "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')
    export LOG_LEVEL

    COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
    PORT="${PORT:-8000}"
    BASE_URL="http://localhost:${PORT}"

    echo "[1/5] git checkout main && git pull --ff-only origin main"
    if ! git diff --quiet HEAD 2>/dev/null; then
        echo "ERROR: Working tree has uncommitted changes. Commit or stash before deploying."
        exit 1
    fi
    git checkout main
    git pull --ff-only origin main

    echo "[2/5] build new image"
    docker tag ml-api-prod:latest ml-api-prod:rollback 2>/dev/null || true
    docker compose -f "$COMPOSE_FILE" build ml-api

    echo "[3/5] deploy"
    docker compose -f "$COMPOSE_FILE" up -d --no-build

    echo "[4/5] health check (max 30s)"
    health_response=""
    for i in {1..30}; do
        if health_response=$(curl -sf --connect-timeout 2 --max-time 3 "$BASE_URL/health" 2>/dev/null); then
            echo "Service ready at attempt $i"
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "ERROR: health check failed after 30s. Rolling back..."
            docker compose -f "$COMPOSE_FILE" logs ml-api
            docker compose -f "$COMPOSE_FILE" stop ml-api
            if docker tag ml-api-prod:rollback ml-api-prod:latest 2>/dev/null; then
                docker compose -f "$COMPOSE_FILE" up -d --no-build
                echo "Rollback complete. Previous image restored."
            else
                echo "No rollback image available (first deploy?). Service is down."
            fi
            exit 1
        fi
        sleep 1
    done

    if echo "$health_response" | grep -q '"modelLoaded": *false'; then
        echo "WARNING: modelLoaded=false — model.joblib not found. /anomaly-score will return 503."
    fi

    echo "[5/5] cleanup dangling images"
    docker image prune -f --filter "dangling=true" || true

    echo "Deploy complete: ${health_response:-healthy (response body unavailable)}"
}
