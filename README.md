# Gwangju Talent Festival ML

Spring Boot 서버에서 export한 좌석/심사 이상 탐지 학습용 CSV 데이터셋을 기반으로
Isolation Forest 모델을 학습하고 FastAPI model-server로 이상 점수를 제공하는 Python ML 레포지토리.

## 프로젝트 목적

- Spring Boot 서버가 export한 domain, metricName, value 기반 시계열 지표에서 이상 패턴 탐지
- label == "normal" 데이터만으로 비지도 학습 (one-class classification)
- 학습된 모델은 FastAPI model-server를 통해 단건 이상 점수 API로 제공

---

## 전체 구조

```text
app/
  main.py              - FastAPI 서버 (health, model-info, anomaly-score)
  schemas.py           - Pydantic request/response 스키마
  model_loader.py      - model.joblib 로드, ModelArtifact 관리
  inference.py         - feature 전처리, IsolationForest 추론
trainer/
  train_iforest.py     - CLI 진입점
  preprocessing.py     - 데이터 검증, feature 생성
  metrics.py           - 이진 분류 지표
data/
  sample_dataset.csv   - 예시 데이터 (git 포함)
models/
  .gitkeep             - artifact 출력 디렉토리 (*.joblib, *.json, *.csv는 git 제외)
scripts/
  train.sh             - trainer 편의 스크립트
  smoke_test.sh        - API smoke test 스크립트
```

---

## 데이터셋 스키마

| 컬럼 | 타입 | feature 처리 | 설명 |
|---|---|---|---|
| domain | string | one-hot | SEAT, JUDGE (Spring export 대문자) |
| metricName | string | one-hot | failure_rate, p95_duration |
| timestamp | ISO8601 | 미사용 | 정렬/검증용 |
| value | float | numeric | 실제 지표값 |
| hourOfDay | int 0-23 | numeric | 시간대 |
| dayOfWeek | int 1-7 | one-hot | 1=월요일, Java DayOfWeek 기준 |
| label | string | train/eval 분리 | normal 또는 anomaly 만 허용 |

---

## 로컬 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Trainer 실행

```bash
python trainer/train_iforest.py \
  --dataset-path ./data/sample_dataset.csv \
  --output-dir ./models/iforest-v1
```

스크립트 버전:

```bash
./scripts/train.sh ./data/sample_dataset.csv ./models/iforest-v1
```

### CLI 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| --dataset-path | 필수 | 입력 CSV 경로 |
| --output-dir | 필수 | artifact 저장 디렉토리 |
| --contamination | auto | IsolationForest contamination (float 또는 auto) |
| --n-estimators | 100 | Decision tree 개수 |

### contamination 기본값: auto 선택 이유

contamination=0.05(float)를 설정하면 sklearn이 학습 데이터의 하위 5%를 이상치 임계값으로 고정한다.
본 trainer는 normal 데이터만으로 학습하므로, float contamination은 정상 데이터의 일부를 강제 이상치로 분류하는 부작용이 있다.
auto는 IsolationForest 원 논문의 고정 offset(-0.5)을 사용해 anomaly rate 가정 없이 결정 경계를 설정한다.
실제 anomaly rate가 알려진 경우 --contamination 0.05 형태로 override한다. (범위: 0 초과, 0.5 이하)

---

## Model-Server 실행

### MODEL_PATH 환경 변수

모델 파일 경로를 `MODEL_PATH` 환경 변수로 주입한다. 기본값은 `./models/iforest-v1/model.joblib`.

```bash
export MODEL_PATH=./models/iforest-v1/model.joblib
```

### uvicorn으로 실행

```bash
uvicorn app.main:app --reload
```

모델 파일이 없으면 서버는 정상 기동하되 `/health`에서 `modelLoaded: false`를 반환하고,
`/anomaly-score` 요청 시 503을 반환한다.

---

## API

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "modelLoaded": true
}
```

### GET /model-info

```bash
curl http://localhost:8000/model-info
```

```json
{
  "modelLoaded": true,
  "modelPath": "models/iforest-v1/model.joblib",
  "featureColumns": ["value", "hourOfDay", "domain_JUDGE", "domain_SEAT", "..."],
  "domainCategories": ["JUDGE", "SEAT"],
  "metricNameCategories": ["failure_rate", "p95_duration"],
  "dayOfWeekCategories": ["1", "2", "3", "4", "5", "6", "7"],
  "contamination": "auto",
  "n_estimators": 100,
  "random_state": 42
}
```

### POST /anomaly-score

**Request:**

| 필드 | 타입 | 허용값 |
|---|---|---|
| domain | string | SEAT, JUDGE |
| metricName | string | failure_rate, p95_duration |
| value | float | finite number (NaN/Infinity 거부) |
| hourOfDay | int | 0-23 |
| dayOfWeek | int | 1-7 (Java DayOfWeek) |

```bash
curl -X POST http://localhost:8000/anomaly-score \
  -H "Content-Type: application/json" \
  -d '{"domain":"SEAT","metricName":"failure_rate","value":0.01,"hourOfDay":9,"dayOfWeek":1}'
```

**Response:**

```json
{
  "anomalyScore": -0.024,
  "predictedLabel": "normal",
  "modelVersion": "iforest-v1",
  "modelLoaded": true
}
```

### anomalyScore 계산 방식

`anomalyScore = -decision_function(X)`

`IsolationForest.decision_function`은 정상일수록 양수, 이상일수록 음수를 반환한다.
부호를 반전해 anomalyScore가 클수록 이상하다는 직관적인 의미를 갖도록 한다.

`predictedLabel`은 `model.predict(X)` 결과 `1 → normal`, `-1 → anomaly`로 매핑한다.

---

## Smoke Test

서버가 실행 중인 상태에서 아래 스크립트로 전체 API를 검증한다.

```bash
./scripts/smoke_test.sh http://localhost:8000
```

---

## Docker 실행

Dockerfile의 기본 CMD는 uvicorn FastAPI 서버다.
Trainer 실행 시에는 docker run에서 커맨드를 override한다.

**FastAPI 서버 기동:**

```bash
docker build -t gwangju-talent-festival-ml .
docker compose up -d
curl http://localhost:8000/health
docker compose down
```

**MODEL_PATH 커스텀 (docker compose):**

`docker-compose.yml`의 `environment` 항목을 수정하거나 실행 시 override한다.

```bash
MODEL_PATH=./models/custom/model.joblib docker compose up -d
```

**Trainer를 Docker로 실행 (CMD override):**

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  gwangju-talent-festival-ml \
  python trainer/train_iforest.py \
    --dataset-path /app/data/sample_dataset.csv \
    --output-dir /app/models/iforest-v1
```

---

## 산출물

학습 후 --output-dir 에 아래 파일이 생성된다.

| 파일 | 설명 |
|---|---|
| model.joblib | IsolationForest 모델 + feature_columns + metadata |
| metrics.json | 전체 데이터 기준 precision / recall / f1 등 |
| predictions.csv | trueLabel, predictedLabel, anomalyScore 전체 행 |

### model.joblib 구조

```python
{
    "model": IsolationForest,
    "feature_columns": [...],
    "label_mapping": {
        "iforest_predict_1":  "normal",
        "iforest_predict_-1": "anomaly"
    },
    "metadata": {
        "contamination": "auto",
        "n_estimators": 100,
        "random_state": 42,
        "featureColumns": [...],
        "domainCategories": ["JUDGE", "SEAT"],
        "metricNameCategories": ["failure_rate", "p95_duration"],
        "dayOfWeekCategories": ["1", "2", "3", "4", "5", "6", "7"]
    }
}
```

---

## 현재 제외 범위

- POST /anomaly-score/batch (배치 추론)
- Spring 서버 연동
- GPU 서버 CI/CD
- model registry
