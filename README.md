
# **NLP 문장 유사도 측정 프로젝트 (NLP-01조)**

## **프로젝트 개요**
이 프로젝트는 **Semantic Text Similarity (STS)** 태스크를 통해 두 문장의 의미 유사도를 0.0에서 5.0 사이의 값으로 측정하는 AI 모델을 구축하는 것을 목표로 합니다. 평가 지표로는 **피어슨 상관계수(Pearson Correlation Coefficient)**를 사용하였으며, 데이터 전처리, 증강, 하이퍼파라미터 최적화를 통해 모델 성능을 개선했습니다.

## **팀원**
- 박준성
- 이재백
- 강신욱
- 홍성균
- 백승우
- 김정석

## **프로젝트 구조**
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

## **개발 환경**
- **서버:** AI Stages GPU (Tesla V100-SXM2) x 4
- **언어:** Python
- **라이브러리:** Transformers, PyTorch, Pandas, WandB, Hugging Face, Matplotlib
- **운영체제:** Linux

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

## **데이터 증강 기법**
모델 성능을 향상시키기 위해 다양한 데이터 증강 기법을 적용했습니다:
- **문장 순서 변경:** sentence_1과 sentence_2를 서로 교환
- **언더샘플링:** 라벨 0 데이터를 줄여 데이터 균형 맞춤
- **텍스트 정규화:** 반복되는 음절 간소화
- **번역 증강:** Google Translator를 이용해 영문 데이터를 한국어로 번역

## **모델 평가**
- **평가지표:** 피어슨 상관계수(Pearson Correlation Coefficient, PCC)
- **검증 전략:** K-fold 교차 검증을 사용해 일반화 성능 향상
- **앙상블:** 가중 평균을 사용한 앙상블 적용

## **결과**
- **Private Leaderboard:** 6위 (PCC: 0.9391)
- **Public Leaderboard:** 9위 (PCC: 0.9341)

## **협업 도구**
- **버전 관리:** [GitHub](https://github.com/your-repo-url)
- **프로젝트 관리:** Notion
- **소통 도구:** Slack, Zoom

## **감사 인사**
프로젝트에 함께한 팀원들에게 감사를 전합니다.
