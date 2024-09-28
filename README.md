# NLP 프로젝트 : Semantic Text Similaryity

## 1. **프로젝트 개요**
이 프로젝트는 **Semantic Text Similarity (STS)** 태스크를 통해 두 문장의 의미 유사도를 0.0에서 5.0 사이의 값으로 측정하는 AI 모델을 구축하는 것을 목표로 합니다. 
평가 지표로는 **피어슨 상관계수(Pearson Correlation Coefficient)**를 사용하였으며, 데이터 전처리, 증강, 하이퍼파라미터 최적화를 통해 모델 성능을 개선했습니다.

### 1.1 개발 환경

| 항목 | 내용 |
|------|------|
| 서버 | AI Stages GPU (Tesla V100-SXM2) * 4EA |
| 기술 스택 | Python, Transformers, PyTorch, Pandas, WandB, Hugging Face, Matplotlib |
| 운영체제 | Linux |

### 1.2 협업 도구

- **Github**: 코드 공유 및 버전 관리, Issue로 진행 중인 Task 공유
- **Notion**: 회의 내용 공유, 프로젝트 일정 관리, 실험 기록
- **Slack**: Github 및 WandB 봇을 활용한 협업, 의견 공유, 회의
- **Zoom**: 실시간 소통을 통한 의견 공유 및 회의

### 1.3 팀 구성 및 역할

| 팀원 | 역할 |
|------|------|
| 박준성 |  |
| 이재백 |  |
| 강신욱 |  |
| 홍성균 |  |
| 백승우 |  |
| 김정석 |  |

## 2. 프로젝트 파일 구조
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

## 2.1 **실행 방법**
2.1.1 **레포지토리 클론:**
   ```bash
   git clone https://github.com/your-repo-url.git
   ```
2.1.2 **필요한 패키지 설치:**
   ```bash
   pip install -r requirements.txt
   ```
2.1.3 **설정 파일 수정:**
   `config.yaml` 파일에서 데이터 경로 및 하이퍼파라미터를 설정합니다.

2.1.4 **모델 학습:**
   ```bash
   python src/main.py --mode=train # config.yaml에 따른 학습
   python src/main.py --mode=train --sweep=True # main.py 내에 sweep_config에 따른 학습
   ```

## 3. 프로젝트 수행 과정 및 방법

### 3.1 그라운드 룰

- Notion 서버 현황판 활용
- Github Issue 및 commit/branch 컨벤션 준수
- 팀 소통 및 협업 규칙 준수

### 3.2 프로젝트 기초 강의 수강 (09/10 ~ 09/13)
- NLP 프로젝트 기초, LINUX, Streamlit, huggingface 학습

### 3.3 베이스라인 코드 분석 및 구조 개편 (09/14 ~ 09/19)
- 코드 모듈화 및 파이프라인 구축

### 3.4 데이터 EDA 및 모델 분석 (09/20)
- 데이터 시각화 및 다양한 모델 성능 평가

### 3.5 데이터 전처리 및 증강 (09/21 ~ 09/23)
- 다양한 전처리 및 증강 기법 적용 및 효과 검증

### 3.6 하이퍼파라미터 튜닝 (09/24)
- WandB를 활용한 실험 로깅 및 하이퍼파라미터 최적화

### 3.7 앙상블 모델 선정 (09/25 ~ 9/26)
- 예측값 EDA를 통한 최종 앙상블 모델 선정 및 가중치 결정

## 4. 프로젝트 결과

| 분류 | 순위 | Pearson |
|------|------|---------|
| private(최종 순위) | 6 | 0.9391 |
| public(중간 순위) | 9 | 0.9341 |

### 최종 순위 상승 이유 분석:
1. 최종 앙상블 모델 선정에 있어서 과적합을 낮추기 위해 최대한 base가 다른 모델들을 선택해 반영함.
2. 앙상블 과정에서 가중치를 조정함에 있어 test 데이터셋에 대한 과적합을 낮추기 위해 ±0.3 단위로만 진행한 것이 유효했다고 판단됨.

## 6. 자체 평가

### 6.1 잘한 점

#### 6.1.1 협업 툴 활용
- **Slack**: 실시간 의견 공유, GitHub 연동으로 알림 설정
- **Notion**: 프로젝트 일정 관리 및 회의 기록
- **GitHub**: 코드 버전 관리 및 이슈 트래킹
- **Zoom**: 실시간 화상 회의로 신속한 결정

#### 6.1.2 프로젝트 목표 설정
- 개인 및 팀 단위 목표 설정
- 목표 중심의 지속적인 프로젝트 참여

#### 6.1.3 원활한 소통
- 데일리 스크럼, 모각공, 피어세션 활용
- 실시간 문제 해결 및 아이디어 토론

### 6.2 아쉬운 점

#### 6.2.1 가설 검증 과정 미흡
- 기능 추가/제거에 따른 버전 관리 부족
- public score에 과도하게 집중

#### 6.2.2 데이터 EDA 부족
- 초기 EDA 이후 지속적인 분석 부족
- 예측값 분석을 통한 모델 성능 향상 기회 놓침
