# NLP 프로젝트 : Semantic Text Similaryity
## NLP-01조 아이즈원
박준성, 이재백, 강신욱, 홍성균, 백승우, 김정석

## **프로젝트 개요**
이 프로젝트는 **Semantic Text Similarity (STS)** 태스크를 통해 두 문장의 의미 유사도를 0.0에서 5.0 사이의 값으로 측정하는 AI 모델을 구축하는 것을 목표로 합니다. 
평가 지표로는 **피어슨 상관계수(Pearson Correlation Coefficient)**를 사용하였으며, 데이터 전처리, 증강, 하이퍼파라미터 최적화를 통해 모델 성능을 개선했습니다.

## **팀원**
- 박준성
- 이재백
- 강신욱
- 홍성균
- 백승우
- 김정석

### 개발 환경

| 항목 | 내용 |
|------|------|
| 서버 | AI Stages GPU (Tesla V100-SXM2) * 4EA |
| 기술 스택 | Python, Transformers, PyTorch, Pandas, WandB, Hugging Face, Matplotlib |
| 운영체제 | Linux |

### 협업 도구

- **Github**: 코드 공유 및 버전 관리, Issue로 진행 중인 Task 공유
- **Notion**: 회의 내용 공유, 프로젝트 일정 관리, 실험 기록
- **Slack**: Github 및 WandB 봇을 활용한 협업, 의견 공유, 회의
- **Zoom**: 실시간 소통을 통한 의견 공유 및 회의

## 팀 구성 및 역할

| 팀원 | 역할 |
|------|------|
| 박준성 |  |
| 이재백 |  |
| 강신욱 |  |
| 홍성균 |  |
| 백승우 |  |
| 김정석 |  |

## 프로젝트 파일 구조
```bash
├── notebooks
│   └── EDA.ipynb               # 탐색적 데이터 분석(EDA)
├── src
│   ├── data_pipeline
│   │   ├── augment_func
│   │   │   ├── AugFunction.py  # 데이터 증강 추상 클래스
│   │   │   ├── swap_sentences.py  # sentence_1과 sentence_2 순서 변경
│   │   │   ├── undersample_label_0.py  # 라벨 0의 데이터 언더샘플링
│   │   ├── augmentation.py      # 증강 실행 모듈
│   │   ├── dataloader.py        # 데이터 로딩, 전처리 및 토큰화
│   │   └── dataset.py           # 데이터셋 클래스
│   ├── eda
│   │   ├── exploration.py       # 스트림릿을 활용한 데이터 시각화
│   │   └── feature.py           # 문장 토큰 길이 특징 추가
│   ├── model
│   │   ├── MultiTaskLoss.py     # 회귀 및 분류 손실을 결합한 다중 학습 손실 함수
│   │   ├── loss.py              # 커스텀 손실 함수
│   │   ├── model.py             # PyTorch Lightning 기반 모델 학습 및 검증
│   │   └── optimizer.py         # 최적화 알고리즘 관리
│   ├── utils
│   │   ├── decorators.py        # 메타데이터 데코레이터
│   │   ├── config.py            # YAML 설정 로드
│   │   ├── ensemble.py          # 앙상블 로직
│   └── main.py                  # 모델 학습, 검증 및 로깅 (PyTorch Lightning)
└── config.yaml                  # 기본 학습 및 데이터 설정
```
- **notebooks**: 코드 셀 단위 확인이 필요한 EDA 결과 또는 임시 코드 작성
- **src**: 소스 코드 디렉토리
  - **data_pipeline**: 데이터 증강 및 전처리를 포함한 모델에 데이터를 공급하기 위한 소스 코드
  - **eda**: 데이터 EDA를 위한 시각화 함수 등
  - **model**: 모델 및 모델 훈련 과정에서 사용되는 손실 함수, 옵티마이저 등
  - **utils**: 여러 파일에서 사용될 가능성이 있는 유틸리티 함수들
- **main.py**: PyTorch Lightning과 WandB를 사용하여 모델을 학습 및 로깅하며 검증과 테스트를 수행
- **streamlit.py**: 데이터 분포 및 토큰 길이 분포를 streamlit 라이브러리를 활용하여 시각화 및 분석 
- **config.yaml**: 모델 기본 훈련 및 데이터 경로 설정

## **실행 방법**
1. **레포지토리 클론:**
   ```bash
   git clone https://github.com/your-repo-url.git
   ```
2. **필요한 패키지 설치:**
   ```bash
   pip install -r requirements.txt
   ```
3. **설정 파일 수정:**
   `config.yaml` 파일에서 데이터 경로 및 하이퍼파라미터를 설정합니다.

4. **모델 학습:**
   ```bash
   python src/main.py
   ```

## 프로젝트 수행 절차 및 방법

### 그라운드 룰

1. 팀 Notion에 있는 서버 현황판 활용하기
2. Git 관련
   - Github Issue 활용하여 수행 중인 작업 공유하기
   - commit convention 사용
   - branch naming convention: `{name}-issue-{issue 번호}`
3. 소통 관련
   - 상호 존중
   - 한 작업에 복수 인력이 투입될 때 따로 모여서 실시간으로 대화하며 협업하기
   - 일일 제출 횟수 제한을 감안하여 슬랙/줌에서 회의 후 최종 결과물 제출
   - 데일리 스크럼/피어 세션 때 남아 있는 PR 처리하기

### 전체 프로젝트 수행 과정 및 상세 설명

1. **프로젝트 기초 강의 수강** (09/10 ~ 09/13)
   - NLP 프로젝트 관련 기초 강의를 수강하며 프로젝트 적용점 도출
   - 기본 LINUX 명령어 이해
   - Streamlit을 이용한 빠른 배포 및 데이터 시각화 공유
   - huggingface 라이브러리 및 NLP Task(Text Classification, QA, STS, NER 등) 이해 

2. **베이스 라인 코드 분석 및 프로젝트 파일 구조 개편** (09/14 ~ 09/19)
   - 기본 제공되는 Pytorch 기반 베이스라인 코드 이해 및 파일 구조 개편
   - 모듈성, 응집도, 결합도를 고려하여 코드 모듈화
   - 모듈화된 코드 기반으로 모델 학습 파이프라인을 크게 data_pipeline, EDA, model, utils로 나눔
   - CLI에서 main.py를 실행시킴으로써 학습 진행
   - 학습 기본값은 config.yaml에 저장하여 argument parser로 입력시키던 하이퍼파라미터의 안정화

3. **데이터 EDA 및 학습 모델 분석** (09/20)
   - 학습 데이터 분석 후 실제로 여러 모델들을 테스트하여 성능 평가
   - EDA: 라벨 데이터 시각화, 길이에 따른 라벨 분포 확인과 이에 따른 max_token 지정
   - 다양한 모델 테스트: 여러 모델들을 테스트하고, 각 모델의 성능 분석 (주로 bert-base)
   - 성능 평가: 실제 예측값과 val_data 간에 피어슨 상관계수를 통한 모델들의 비교 분석

4. **데이터 전처리 및 증강** (09/21 ~ 09/23)
   - 데이터 전처리 및 증강의 실효성 검증
   - 학습 데이터 sentence_swap (😃 val_pearson 소폭 상승 확인)
   - label_0 데이터에 대해 sentence_1을 sentence_2로 복사하여 label_5 데이터로 증강 (😑 모델 별로 미미한 변화 또는 아주 작은 val_pearson의 상승)
   - 데이터 정규화를 통해 반복되는 음절 간소화 (😃 val_pearson 소폭 상승 확인)
   - 영문 데이터에 대하여 GoolgeTranslator 라이브러리를 활용하여 한글로 변환 (😃 val_pearson 소폭 상승 확인)
   - Dataset 모듈에 input_data로 input_ids만 들어가는 것을 확인. attention_mask 또한 input_data에 추가 (😃 val_pearson 소폭 상승 확인)
   - train, val 데이터를 합친 후 k-fold 방식으로 모델에 학습, 검증 데이터를 입력시켜 일반화 성능 향상 (😑 val_pearson 상승 없거나 소폭 하락)
   - mecab() 형태소 분석기를 활용하여 단순 띄어쓰기 외에 형태소 단위로 띄어쓰기를 진행하여 모델에 입력 (🥺 val_pearson 하락)
   - 학습에 continuous한 label데이터(0.0 ~5.0) 외에 binary_label데이터(0 or 1)를 포함시켜 예측시킴 (🥺 val_pearson 하락)

5. **하이퍼파라미터 튜닝** (09/24)
   - WandB 라이브러리를 통한 실험 로깅 및 sweep 옵션을 통해 하이퍼파라미터 튜닝
   - main.py에 wandb.login 및 wandb.init을 추가하여 학습 과정을 로깅할 수 있도록 함
   - argument parser로 model_path와 sweep 옵션을 설정하여 하이퍼파라미터 최적화 실행
   - wandb sweep의 경우 yaml 파일로 넘기기보다 main.py에서 직접 sweep_config를 설정할 수 있도록 dictionary를 구현함

6. **예측값 EDA를 통한 앙상블 대상 모델 선정** (09/25 ~ 9/26)
   - test_pearson 및 val_pearson 기반 모델 리스트 업 및 앙상블 결정
   - 학습 완료 모델들의 예측값들을 val_pearson 및 test_pearson 상관계수를 통해 앙상블의 재료로 리스트업
   - pearson 상관계수가 예측값들의 선형성을 중요시하는 바, 이에 따라 각 예측값의 상관계수가 서로 0.95 이상인 모델들 위주로 앙상블 진행 
   - 앙상블의 예측값을 시각화하여 실제 어떤 모델들을 넣고 뺄지 혹은 가중치를 얼마나 둘지를 결정하여 최종 제출

## 프로젝트 결과

| 분류 | 순위 | Pearson |
|------|------|---------|
| private(최종 순위) | 6 | 0.9391 |
| public(중간 순위) | 9 | 0.9341 |

### 최종 순위 상승 이유 분석:
1. 최종 앙상블 모델 선정에 있어서 과적합을 낮추기 위해 최대한 base가 다른 모델들을 선택해 반영함.
2. 앙상블 과정에서 가중치를 조정함에 있어 test 데이터셋에 대한 과적합을 낮추기 위해 ±0.3 단위로만 진행한 것이 유효했다고 판단됨.

## 자체 평가

### 잘한 점
- 협업: Slack, Notion, GitHub, Zoom을 효과적으로 활용

### 아쉬운 점
- 가설 검증 과정이 충분히 체계적이지 못했다.
