#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

check() {
    local label="$1"
    local status="$2"
    if [ "$status" -eq 0 ]; then
        echo "[PASS] $label"
    else
        echo "[FAIL] $label"
        exit 1
    fi
}

echo "=== Smoke Test: $BASE_URL ==="

echo ""
echo "--- GET /health ---"
HEALTH=$(curl -sf "$BASE_URL/health")
echo "$HEALTH"
echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok', d" && check "/health status=ok" 0 || check "/health status=ok" 1
echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'modelLoaded' in d, d" && check "/health modelLoaded field exists" 0 || check "/health modelLoaded field exists" 1

echo ""
echo "--- GET /model-info ---"
INFO=$(curl -sf "$BASE_URL/model-info")
echo "$INFO"
echo "$INFO" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for k in ['modelLoaded','modelPath','featureColumns','domainCategories','metricNameCategories','dayOfWeekCategories','contamination','n_estimators','random_state']:
    assert k in d, f'missing key: {k}'
print('all required keys present')
" && check "/model-info required keys" 0 || check "/model-info required keys" 1

echo ""
echo "--- POST /anomaly-score (normal case) ---"
SCORE=$(curl -sf -X POST "$BASE_URL/anomaly-score" \
    -H "Content-Type: application/json" \
    -d '{"domain":"SEAT","metricName":"failure_rate","value":0.01,"hourOfDay":9,"dayOfWeek":1}')
echo "$SCORE"
echo "$SCORE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for k in ['anomalyScore','predictedLabel','modelVersion','modelLoaded']:
    assert k in d, f'missing key: {k}'
assert d['predictedLabel'] in ('normal','anomaly'), d
assert d['modelLoaded'] is True, d
print('all fields valid')
" && check "/anomaly-score normal input" 0 || check "/anomaly-score normal input" 1

echo ""
echo "--- POST /anomaly-score (anomaly case) ---"
SCORE2=$(curl -sf -X POST "$BASE_URL/anomaly-score" \
    -H "Content-Type: application/json" \
    -d '{"domain":"SEAT","metricName":"failure_rate","value":0.95,"hourOfDay":15,"dayOfWeek":1}')
echo "$SCORE2"
echo "$SCORE2" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert d['predictedLabel'] in ('normal','anomaly'), d
print('predictedLabel:', d['predictedLabel'])
" && check "/anomaly-score anomaly input" 0 || check "/anomaly-score anomaly input" 1

echo ""
echo "--- POST /anomaly-score (invalid domain, expect 422) ---"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/anomaly-score" \
    -H "Content-Type: application/json" \
    -d '{"domain":"INVALID","metricName":"failure_rate","value":0.01,"hourOfDay":9,"dayOfWeek":1}')
[ "$HTTP_CODE" = "422" ] && check "/anomaly-score invalid domain -> 422" 0 || check "/anomaly-score invalid domain -> 422" 1

echo ""
echo "--- POST /anomaly-score (NaN value, expect 422) ---"
HTTP_CODE2=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/anomaly-score" \
    -H "Content-Type: application/json" \
    -d '{"domain":"SEAT","metricName":"failure_rate","value":null,"hourOfDay":9,"dayOfWeek":1}')
[ "$HTTP_CODE2" = "422" ] && check "/anomaly-score null value -> 422" 0 || check "/anomaly-score null value -> 422" 1

echo ""
echo "=== All smoke tests passed ==="
