# Gwangju Talent Festival ML

Spring Boot 서버에서 export한 좌석/심사 이상 탐지 학습용 CSV 데이터셋을 기반으로
Isolation Forest 모델을 학습하는 Python ML 레포지토리.

## 프로젝트 목적

- Spring Boot 서버가 export한 domain, metricName, value 기반 시계열 지표에서 이상 패턴 탐지
- label == "normal" 데이터만으로 비지도 학습 (one-class classification)
- 학습된 모델은 FastAPI model-server에서 재사용

---

## 전체 구조

```text
app/
  main.py              - FastAPI health endpoint
trainer/
  train_iforest.py     - CLI 진입점
  preprocessing.py     - 데이터 검증, feature 생성
  metrics.py           - 이진 분류 지표
data/
  sample_dataset.csv   - 예시 데이터 (git 포함)
models/
  .gitkeep             - artifact 출력 디렉토리 (*.joblib, *.json, *.csv는 git 제외)
scripts/
  train.sh             - 편의 스크립트
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

## FastAPI health check 실행

```bash
uvicorn app.main:app --reload
```

```bash
curl http://localhost:8000/health
# {"status":"ok"}
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
    "model": IsolationForest,           # .predict(X) 가능
    "feature_columns": [...],           # one-hot 후 순서 보장 컬럼 목록
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

featureColumns, *Categories를 사용해 FastAPI model-server에서
동일한 one-hot 컬럼 순서를 재현한다.

---

## 현재 제외 범위

- FastAPI /anomaly-score API
- Spring 서버 연동
- GPU 서버 CI/CD
- model registry
