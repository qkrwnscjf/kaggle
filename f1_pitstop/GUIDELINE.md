# Kaggle F1 Pit Stop Prediction Project Roadmap

이 문서는 2026 Kaggle Playground Series (S6E5) 'Predicting F1 Pit Stops' 경쟁을 위한 단계별 시나리오 및 가이드라인입니다.

## 1. 프로젝트 시나리오 (Workflow)

### Phase 1: 탐색 및 기반 구축 (Research)
* **데이터 이해:** 제공된 train.csv, test.csv의 컬럼 의미 분석.
* **EDA:** 타이어 마모도(TyreLife, Degradation)와 피트인(PitNextLap) 간의 상관관계 시각화.
* **가설 설정:** "타이어 수명이 임계치에 도달하고 랩 타임이 느려지면 피트인 확률이 급격히 상승할 것이다."

### Phase 2: 특성 공학 (Feature Engineering)
* **상대적 지표:** TyreLife를 Compound별 최대 수명으로 나눈 비율 생성.
* **시계열 특성:** 최근 3~5개 랩의 LapTime_Delta 이동 평균(Moving Average).
* **경기 전략:** RaceProgress를 활용한 잔여 랩 수 계산 및 전략적 피트 타이밍 유추.

### Phase 3: 모델링 및 튜닝 (Modeling)
* **Base Model:** CatBoost (범주형 변수 자동 처리 및 높은 성능).
* **Validation:** Race와 Year 기준 GroupKFold (7:3 또는 5-Fold).
* **Optimization:** Optuna를 활용한 하이퍼파라미터 튜닝.

### Phase 4: 평가 및 제출 (Submission)
* **Metric:** AUC-ROC 점수 극대화.
* **Ensemble:** LightGBM, XGBoost와의 Soft Voting 앙상블 고려.

---

## 2. Kaggle Editor 활용 가이드라인

Kaggle Notebook 환경에서 효율적으로 작업하기 위한 팁입니다.

### 1 데이터 로드 및 경로 설정
Kaggle Notebook의 데이터 경로는 항상 ../input/ 아래에 위치합니다.
```python
import pandas as pd
import os

# 데이터 경로 확인
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 로드 예시
train = pd.read_csv('/kaggle/input/playground-series-s6e5/train.csv')
```

### 2 GPU/Internet 설정
* **Accelerator:** 대용량 데이터나 복잡한 모델 학습 시 우측 'Settings' 메뉴에서 GPU P100 또는 T4 x2를 활성화하세요.
* **Internet:** 외부 라이브러리를 pip install 하려면 Internet On 설정이 필요합니다.

### 3 메모리 관리
* 데이터셋이 큰 경우 pd.read_csv 시 dtype을 지정하여 메모리 사용량을 줄이세요.
* 불필요한 DataFrame은 del df 후 gc.collect()를 호출하여 메모리를 확보하세요.

### 4 버전 관리 및 제출
* **Save Version:** 'Save Version' 클릭 후 'Save & Run All (Commit)'을 선택하면 전체 코드가 실행되고 출력이 저장됩니다.
* **Submit:** 실행이 완료된 버전의 Output 탭에서 submission.csv를 선택하여 직접 제출할 수 있습니다.

---

## 3. 핵심 전략 요약
1. **Target Leakage 주의:** 미래의 정보를 사용하는 변수가 없는지 철저히 확인.
2. **시각화 기반 의사결정:** 상관관계 히트맵과 변수 중요도(Feature Importance)를 수시로 확인.
3. **일반화 성능 우선:** 리더보드 점수(Public Score)에만 집착하지 말고 검증 점수(Local CV)와의 괴리를 최소화.
