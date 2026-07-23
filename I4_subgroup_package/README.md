# I4_subgroup_package

I4를 **포함한** 모델(19 features: `I2`–`I20` + `I4`)의 학습 · 추론 · BMI/나이 서브그룹 평가용 **코드 패키지**입니다.

대상 서브그룹:

- `BMI25_under` / `BMI25_upper` (BMI 25 기준)
- `age60_under` / `age60_upper` (나이 60 기준)

> 데이터·학습 가중치·평가 산출물(CSV/이미지/`model.pkl`)은 포함하지 않습니다.  
> 로컬에 `traindataset.csv`, `testdataset.csv`, demographics Excel, `train_results/` 등이 있어야 재실행할 수 있습니다.

## 파일 구성

| 파일 | 역할 |
|------|------|
| `train.py` | 모델 학습 (CV, ROC, calibration, bootstrap CI) |
| `infer.py` | 테스트 CSV 추론·평가 |
| `ml_analysis_utils.py` | SHAP, permutation, calibration, bootstrap 공통 유틸 |
| `run_infer_subgroups.py` | 전체 + BMI/나이 서브그룹 일괄 추론 |
| `split_testdataset_by_bmi.py` | BMI 25 기준 분할 CSV 생성 |
| `split_testdataset_by_age.py` | 나이 60 기준 분할 CSV 생성 |
| `summarize_subgroup_tables.py` | 서브그룹 성능 표(논문용) 생성 |
| `summarize_subgroups.py` / `summarize_results.py` | 결과 요약 |
| `param_config.yaml` | **I4 포함** feature·하이퍼파라미터 설정 |
| `requirements.txt` | Python 의존성 |

## Features

- Inputs: `I2, I3, I4, I5, …, I20` (19개)
- Target: `abnormality` (0/1)
- Models: Random Forest, XGBoost, SVM, Logistic Regression, Neural Network
- Prediction: `predict_proba`, threshold `0.5`

## 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

프로젝트 루트에 데이터와(선택) 기존 `train_results/`를 두고, 이 폴더에서 실행합니다.

### 1) 학습

```bash
python train.py --config param_config.yaml
```

### 2) BMI / 나이 분할

```bash
python split_testdataset_by_bmi.py
python split_testdataset_by_age.py
```

### 3) 서브그룹 일괄 추론

I4 학습 run을 고정할 때 (`20260622_053*` 예시):

```bash
python run_infer_subgroups.py --weights_run_prefix 20260622_053 --eval_suffix _I4
```

결과는 `eval_results/<group>_I4/<algo>/<timestamp>/` 아래에 저장됩니다.

### 4) 성능 표 요약

```bash
python summarize_subgroup_tables.py
```

## 관련 저장소

이 패키지는 [Jeon-Insu/V3-gemma](https://github.com/Jeon-Insu/V3-gemma) 하위 폴더 `I4_subgroup_package/` 로 관리됩니다.
