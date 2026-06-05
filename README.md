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
  deploy.sh            - 서버 배포 스크립트
.github/workflows/
  ci.yml               - CI (테스트, Docker 빌드)
  deploy.yml           - CD (main push 시 서버 자동 배포)
docker-compose.yml         - 개발/로컬용 Compose
docker-compose.prod.yml    - production 배포용 Compose
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

## 배포

### 1. 서버 사전 준비 사항

서버에 아래 소프트웨어가 설치되어 있어야 한다.

- Docker Engine
- Docker Compose v2 (`docker compose` 명령)
- git
- curl

최초 1회 서버에서 레포를 clone한다. 이후 배포는 `git pull` 방식으로 동작한다.

```bash
git clone https://github.com/School-of-Company/Gwangju-talent-festival-ML.git <deploy-path>
cd <deploy-path>
cp .env.example .env
```

서버 준비 상태 확인:

```bash
ssh <user>@<host> -p <port> "docker --version && docker compose version && git --version"
```

### 2. .env 설정

`.env.example`을 복사한 뒤 서버 환경에 맞게 편집한다.

```bash
cp .env.example .env
```

| 변수 | 기본값 | 설명 |
|---|---|---|
| MODEL_PATH | /app/models/iforest-v1/model.joblib | 컨테이너 내부 경로 기준 |
| PORT | 8000 | host 외부 노출 포트 (컨테이너 내부 uvicorn은 8000 고정) |
| LOG_LEVEL | info | uvicorn 로그 레벨 (소문자 필수: info, debug, warning, error, critical) |

production에서는 `.env` 파일이 반드시 있어야 한다. 없으면 `deploy.sh`가 즉시 실패한다.

### 3. model.joblib 배치 방법

실제 운영 모델은 Git에 커밋하지 않는다. 서버 디렉토리에 직접 배치한다.

- 서버 호스트 경로: `<deploy-path>/models/iforest-v1/model.joblib`
- 컨테이너 내부 경로: `/app/models/iforest-v1/model.joblib` (volume mount로 연결)

모델 파일이 없어도 서버는 기동된다. `/health`가 `modelLoaded: false`를 반환하고 `/anomaly-score`는 503을 반환한다.

서버에서 trainer를 실행해 model.joblib을 생성하는 작업은 이번 PR 범위가 아니다. 모델 파일은 별도 절차로 서버에 배치한다.

### 4. 수동 배포 실행

서버에 SSH 접속 후 아래 명령으로 배포한다.

```bash
cd <deploy-path>
bash scripts/deploy.sh
```

`deploy.sh` 동작 순서:

1. `.env` 존재 여부 확인 (없으면 실패)
2. `git pull --ff-only origin main`
3. `docker compose build ml-api` 로 새 이미지 빌드
4. `docker compose -f docker-compose.prod.yml up -d --no-build` 로 컨테이너 교체
5. `/health` HTTP 200 확인 (최대 30s), 실패 시 이전 이미지로 자동 롤백
6. dangling 이미지 정리

### 5. GitHub Secrets 설정

레포 Settings -> Secrets and variables -> Actions 에서 아래 Secrets를 추가한다.

| Secret | 설명 | 필수 |
|---|---|---|
| ML_SERVER_HOST | 서버 IP 또는 도메인 | 필수 |
| ML_SERVER_USER | SSH 접속 사용자명 | 필수 |
| ML_SERVER_SSH_KEY | SSH 개인키 전체 내용 (-----BEGIN ... 포함) | 필수 |
| ML_SERVER_DEPLOY_PATH | 서버 내 프로젝트 절대경로 | 필수 |
| ML_SERVER_PORT | SSH 포트, 미설정 시 22 사용 | 선택 |

### 6. GitHub Actions 배포 흐름

main 브랜치에 push되면 `deploy.yml`이 자동 실행된다.

```
main push -> deploy.yml 트리거 -> SSH 접속 -> git pull --ff-only origin main
-> docker compose -f docker-compose.prod.yml up -d --build
-> /health HTTP 200 확인
```

전제: main 브랜치는 CI 통과 후 merge되는 branch protection 구조.

동시 배포 방지를 위해 `concurrency` 설정이 적용되어 있다. 이전 배포가 진행 중이면 취소하고 최신 push로 재시작한다.

### 7. 배포 후 health check

CD 검증 기준은 `/health` HTTP 200이다.

- `modelLoaded: true` — 모델이 로드된 정상 상태
- `modelLoaded: false` — 서버는 기동 중이나 model.joblib 미배치 상태. **배포 성공으로 간주한다.** deploy.sh가 WARNING 메시지를 출력한다.
- `/anomaly-score`는 model.joblib이 없으면 503을 반환한다.

```bash
curl http://localhost:8000/health
```

### 8. 장애 시 로그 확인

```bash
docker compose -f docker-compose.prod.yml logs ml-api
```

---

## 현재 제외 범위

- POST /anomaly-score/batch (배치 추론)
- Spring 서버 연동
- model registry
- 자동 재학습 파이프라인
- 서버에서 trainer 실행 (model.joblib 생성은 별도 절차)
