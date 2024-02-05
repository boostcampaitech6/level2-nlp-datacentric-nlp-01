 # 0. 프로젝트 요약
- **EDA**
    - 대회 분석(오라벨 체크)
        - G2P
        - Labeling Error
        - class의 모호함
    - Embedding Clustering
        - last hidden state CLS
        - Anomaly Detection
        - Gaussian Mixture Model

- **Data PreProcessing**
1)  Dataset 전처리
   
    - Cleanlab
    - G2P
    - Hand-craft

2)  Input token 전처리
   
    - 한자변환
    - 의미 없는 토큰 제거
    - 맞춤법 검사

- **Data Augmentation**
    - TTS 증강
    - Backtranslation
    - 네이버 뉴스 크롤링
    - [AI 허브, 기계독해 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=577)

# 1. 프로젝트 개요
### 1.1 프로젝트 주제
- 본 대회는 모델 구조의 변경 없이 Data-Centric 관점으로 텍스트 주제를 분류(Topic Classification)한다.
- Topic Classification은 뉴스 헤드라인과 Topic의 관계를 파악해서, 올바른 Topic을 예측하는 것을 목표로 한다.
- Data-centric의 취지에 맞춰, Train data에 대한 Augmentation, Quality Control 등은 가능하지만, Baseline 코드 모델 변경은 일체 허용되지 않는다.

### 1.2 프로젝트 구현 내용
- 주제분류 프로젝트를 통해 모델이 문장 주제를 어떻게 파악하는지 알아본다.
- Data-centric의 관점이 주목받는 요즘, 데이터만으로 얼마나 성능 개선이 가능한지 탐구하고자 한다.

### 1.3 활용 장비 및 재료(개발 환경, 협업 tool 등)
- VS Code + SSH 접속을 통해 AI stage 서버 GPU 활용
- Git을 통한 버전 관리, Github를 통한 코드 공유 및 모듈화
- Slack, Zoom, Notion을 통한 프로젝트 일정 및 문제 상황 공유 + 회의 진행

### 1.4 데이터셋 설명
- 기본제공 데이터셋 train.csv, test_data.csv
- 생활문화, 스포츠, 세계, 정치, 경제, IT과학, 사회 7개 토픽의 뉴스 기사 헤드라인로 구성
- train 총 7000 행, test_data 총 47785 행
- 이후 대회 진행 중 전처리 및 증강에 따라 학습 데이터를 변경하여 사용함.

# 2. 프로젝트 팀 구성 및 역할
- 김인수(팀장) : Hand-craft label, Embedding Clustering, g2p noise data cleaning
- 김동언(팀원) : Hand-craft label, 한자변환, TTS → STT 데이터 증강
- 오수종(팀원) : Hand-craft label, Cleanlab
- 이재형(팀원) : Hand-craft label, Embedding Clustering, Naver News Crawling
- 임은형(팀원) : Hand-craft label, Naver News Crawling, Backtranslation
- 이건하(팀원) : Hand-craft label, Embedding Clustering

# 3. 프로젝트 수행 절차 및 방법
### 3.1 팀 목표 설정
![Timeline](https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-01/assets/88371786/4690ce38-922b-436c-bc45-d9f5eedd7085)

### 3.2 프로젝트 사전기획
- 김인수: Hand-craft label, Embedding Clustering, g2p noise data cleaning
- 김동언: Hand-craft label, 한자변환, TTS → STT 데이터 증강
- 오수종: Hand-craft label, Cleanlab
- 이재형: Hand-craft label, Embedding Clustering, Naver News Crawling
- 임은형: Hand-craft label, Naver News Crawling, Backtranslation
- 이건하: Hand-craft label, Embedding Clustering
- 이외에 대회에 필요한 태스크 분배해서 진행
- Notion : 조사 및 실험 결과 공유, 기록 정리 및 보관
- Github: 공용 코드 정리, 모듈화 및 코드리뷰, 진행사항 공유
- Zoom : 화상 회의를 통한 실시간 상황 공유, 결과 피드백, 이슈 해결

### 3.3 프로젝트 수행
- Project Pipeline
    ![ProjectPipeline](https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-01/assets/88371786/c1d10eea-dd74-4884-ba1b-d7c7fadb9637)

1. Hand-Craft Labeling을 통해 실제 데이터 내에 존재하는 이상치를 검수함
2. 문장 내 단어 중 오히려 학습에 방해되는 요인을 제거함
3. Cleanlab을 통해 모델을 통한 이상치 탐지 후 이상치를 제거함
4. Embedding Clustering을 수행하고, T-SNE를 통해 라벨링 된 군집 내 다른 라벨링이 된 문장이 있는 경우 이상치로 판단하고 제거함
5. Hanja 패키지를 활용하여 문장 내 한자가 [UNK]가 되는 것을 방지함
6. Hanspell을 통해 맞춤법을 교정하여 Tokenize가 잘 수행되도록 처리함
7. G2P 패키지를 활용하여 음성학 적으로 잘못 표기된 단어를 생성함 (Robustness)
8. Data Crawling을 수행하여, 라벨별 데이터를 추가 수집함
9. AI hub 내에 존재하는 데이터를 이용하여 데이터를 추가 수집함
10. Backtranslation을 수행하여 데이터를 증강함
11. gTTS 모델을 사용하여 Text-to-Speech로 변환한 후 Whisper 모델을 사용하여 Speech-to-Text 로 다시 변환하여 데이터를 증강함

# 4. 프로젝트 수행 결과
### 4.1 EDA
- 대회 분석(noise 확인)
    - 학습 및 평가 데이터에 존재하는 **G2P**, **Labeling error**, **class의 모호함**(세계-사회, IT과학-생활문화 등) 문제를 해결하는 것이 과제라고 판단 
        ![DataNoise](https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-01/assets/88371786/d0936156-c0ee-4b21-843d-6d9a2bd6a012)
        
    - 학습 데이터의 input_text와 평가 데이터의 input_text의 길이 분포는 전체적으로 유사함
        <img width="800" alt="length" src="https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-01/assets/88371786/9369f3a1-a45c-4a4d-a04a-000556dae52c">
- Embedding Clustering
    - **Labeling error**, **class의 모호함**을 확인하기 위해 훈련데이터를 이용해 klue/bert-base(Baseline Model)을 fine-tuning 시킨 뒤, Input sentence의 임베딩 분포를 확인
        - klue/bert-base 모델의 last hidden state의 CLS 토큰 임베딩 값(768차원)
    - 시각화-군집화를 위해 768차원의 임베딩 값을 t-sne, PCA 등으로 dimension extraction 진행
    - 2D 매핑시 각자의 군집을 잘 형성 → Labeling error로 추정되는 Annomaly target로 확인
        ![EmbeddingClustering1](https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-01/assets/88371786/56c7104b-a575-46e0-b402-a4657d014f5e)

    - Gaussian Mixture Model을 이용한 Clustering을 진행해 Annomaly target을 유사한 임베딩 값을 가지는 target으로 재분류 후 학습 진행하였으나, baseline 보다 성능이 하락하였기에, 다른 전처리 방법을 사용하기로 함
    
    ![EmbeddingClustering2](https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-01/assets/88371786/51468d38-46e9-4ef8-8d3d-d6e4f1a7e496)
    - 오라벨 데이터를 그대로 훈련데이터로 사용하였기에 Embedding이 오염되었을 가능성, 다른 target과 경계상에 존재하는 일부 데이터셋의 target이 변경되어 예측성능이 하락했다는 결론 하에 다른 방법 사용.

### 4.2 Preprocessing
1) Dataset 전처리
- Cleanlab
    - 먼저 모델을 학습 시키고, 그 모델로 train의 예측값을 추론해서 그 안에서 특이값을 찾음
    ![Cleanlab1](https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-01/assets/88371786/fe629e81-7ae3-4d45-8ebf-c71549bed56e)
    
    - Cleanlab을 통해 7000개의 train data 중 533개가 이상치 데이터라고 판단
    - 예시를 뽑아보면 ‘출근길 눈 내리는 경기북부… 한파는 다시 풀려' 생활문화로 보이는 input이 정치로 오분류 되어있는 모습. 아래 예시들도 동일
    ![Cleanlab2](https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-01/assets/88371786/718b4e63-e41f-4245-bd9a-7661bc5478b9)
    
    - 그 이상치의 라벨에 따른 분포이며 맨 오른쪽이 라벨의 퀄리티 스코어
    - F1 score는 0.8384 →0.8409, Accuracy는 0.8421 → 0.8444 로 상승
    - clean lab을 할때 , 기본모델로 학습 하되, 여러 다양한 epoch 을 적용
        - epoch = 20 일때 만든 모델을 활용 기본모델 대비
        - F1 score은 0.8409 → 0.8421 Acuracy는 0.8444 → 0.8460 로 상승

- Graphemes to Phoneme(G2P)
    <img width="1264" alt="DataNoise_" src="https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-01/assets/88371786/ac724897-c930-44ea-b72d-254afdf76d08">

    - train, test 모두에 존재하는 g2p noise 데이터를 제거하면 test 데이터셋에서 noise 데이터가 아닌 경우의 모델 성능 향상 가정
    - prescriptive, descriptive 방법으로 변환을 했을 경우 본래 text와 차이가 없다면 g2p noise 데이터로 간주. 두 방법의 합집합인 경우만 해당.기존 train 데이터 기준 818개.
    - 결과: 기존 baseline model보다 성능 하락 → noise 데이터를 삭제하는 것이 아닌 주입을 통해 train 데이터셋을 test 데이터셋과 유사하게 만들지 못한 이유라고 판단.
- Hand-craft label
    - train dataset의 라벨링이 과연 모델 학습에 적합할지 의문 가짐
        - “인터뷰 판후이 0 대 8 상황에서 인간 첫승…정말 좋았다” → 제목만 봐선 스포츠 같지만, IT 과학으로 분류됨
    - 기계적인 분류가 아니라, 모델이 정확하게 학습할 수 있게, 오해의 소지가 있는 라벨들을 모든 팀원들이 1200개씩 수작업 라벨링 진행
    - 오히려 성능이 baseline 보다 하락하는 결과 → 팀 내부적으로 카테고리 별 분류 기준을 세부적으로 정했으나, 결과적으로 애매한 데이터가 삭제되어 성능이 하락한 것 같다.

2) Input token 전처리
- 한자변환(hanja)
    - 한자를 한글로 변환하여 학습을 시도했으나, 결과적으로 성능 향상에 거의 영향을 미치지 않았다. 데이터가 짧은 문장이기 때문에 한자를 모두 한글로 변환하는 것은 오히려 inference 시에 더 혼란을 가져다 줄 것이라고 판단하여, 한자와 변환한 한글 모두를 사용하여 학습을 시도하였지만, 성능은 거의 동일했다.
    - 뉴스 기사 제목에 비슷한 한자들이 반복해서 나오고, 또한 Vocab에 이미 해당 한자들이 추가되어 있어 굳이 변환 해줄 필요가 없었을 것이라고 생각한다.
- 의미없는 토큰 제거
    - "..."과 같은 큰 의미가 없는 토큰을 제거한 후 학습을 진행하니 성능이 소폭 상승하였다. 모델의 혼란을 줄이고, 불필요한 정보를 배제하여 이러한 결과가 나왔다고 생각한다.
- 맞춤법 검사(HanSpell)
    - 기존 학습 데이터에는 맞춤법 오류가 일부 존재했기 때문에, 맞춤법 교정 라이브러리인 hanspell을 활용하여 이를 제거하려고 하였다.
    - 테스트 데이터에도 맞춤법 오류가 존재하는 데이터가 있어서 이러한 노이즈를 완전히 제거하는 것이 성능 향상에 큰 도움이 되지 않았다고 생각한다.

### 4.3 Augmentation
- TTS 증강
- BackTranslation
    - BackTranslation을 사용하여 데이터 증강을 시도하였다. T5 모델을 기반으로 한 ko-en 및 en-ko로 fine-tuning된 모델을 사용하여 train 데이터를 증강했으나, 결과는 좋지 않았다. 더 나은 성능을 위해 한국어로 fine-tuning된 모델을 선택하였지만 문장이 반복되거나 역번역이 올바르게 이루어지지 않는 문제가 발생하였고, 예상대로 성능이 하락하였다.
    - T5 대신 deepl, papago와 같은 API를 사용한 결과, T5 모델보다 나은 성능을 얻을 수 있었다.
    - 그러나 약 6000개의 데이터셋에서 일부(처음에는 라벨별 20%, 최대 50%까지 실험)를 증강하면서 학습한 결과  train f1 score는 계속 상승하였지만 리더보드 결과는 개선되지 않았다. 이는 학습 데이터에 과적합 되어 결과가 개선되지 않았을 것이라고 생각한다.
    - 역번역으로 비슷한 내용을 여러 번 학습 함에 따라 더 다양한 데이터를 활용해야 한다는 결론을 얻었다. 이러한 경험으로 인해 다양한 데이터 소스를 활용하고 적절한 데이터 증강 방법을 선택하는 것이 모델의 성능 향상에 중요하다고 판단하였다.
- Naver News Crawling
    - 2016년에서 2022년까지의 IT과학, 경제, 사회, 생활문화, 세계, 스포츠, 정치 분야의 뉴스 데이터 10954개 수집
    - 결과: 원래 train dataset과 혼합해서 학습한 결과, baseline 보다 성능 하락
- AI Hub 기계독해 데이터셋
    - IT과학, 경제, 사회, 생활문화, 세계, 스포츠, 정치 분야의 뉴스 데이터 22350개 수집
    - 문화 → 생활문화, 지역 → 사회 등과 같이 dataset에 맞게 뉴스 카테고리 변경
    - 결과: 모델 성능 하락 → 신문사별로 신문을 분류하는 기준이 다른 경향. 이러한 경향이 모델이 뉴스 주제를 분류하는데 악영향을 끼쳤다고 판단.

### 4.4 Final Submission
- **실험 결과**
    - 다양한 시도를 접목해 보았으나, 단일 기법 적용으로 직접적인 성능향상으로 이어진 결과가 많지 않았기에 최종 점수만 정리했으며, 관련 분석 내용에 대해서 5.3에 자세히 서술하였다.
- **Result**
    - Public LeaderBoard
        <img width="1072" alt="PublicScore" src="https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-01/assets/88371786/bced0634-5667-40fd-8bb4-1f1b98de64ba">
        
    - Private LeaderBoard
        <img width="1085" alt="PrivateScore" src="https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-01/assets/88371786/7ae34893-2ffb-4bcb-bc14-d57635db139c">

# 5. 자체 평가 의견
### 5.1 잘한 점
- Github, Notion 등 협업이 원활하게 이루어져서 실험 단계에서 다양한 요소를 실험해볼 수 있었다.

### 5.2 시도 했으나 잘 되지 않았던 것
- 임베딩 클러스터링 결과가 잘 나와 이를 활용하고 싶었지만, 이후 데이터분석에서 다음 로직으로 이어질 수 있는 논리를 추출하지 못했다.
- 라벨 오류를 handcraft 방식으로 직접 수정했다. 사람이 직접 확인하는 것이 가장 효과적인 라벨 오류 해결 방법이라고 생각하였지만 이번 대회에서는 각자가 일부 데이터의 오류를 수정하였고, 이로 인해 기준이 명확하지 않았고 test 데이터에도 라벨 오류가 존재하여 성능이 하락했다고 판단하였다. 더 많은 시간이 주어졌다면 각자가 전체 학습 데이터를 직접 수정하고 다수결 방식을 통해 오류를 수정했다면 더 나은 성능을 얻을 수 있지 않았을까 하는 아쉬움이 남았다.

### 5.3 아쉬웠던 점 → 개선 방안
- 대회 규칙상 ‘Inference 시 전처리작업을 적용할 수 없다’는 점, Train dataset뿐만 아니라 ‘Test dataset에도 noise가 존재’한다는 점, Dataset의 ‘target(class)가 모호’했다는 점과 같은 이유로, 대회의 본질은 Noise Cleaning이 아니라 Noise 주입해야 한다는 사실을 너무 늦게 깨달았다.
- 크롤링을 초반부터 하지 않아, 많은 데이터를 수집하지 못했다. 또한 DB 환경이 아니라는 점을 고려하지 못해 데이터를 중간에 저장하는 로직을 만들지 못했다. → 빠른 액션과 깊은 고민 사이의 절충이 필요하다. 경험을 통해 부족한 부분을 쌓아나가야겠다.

### 5.4 프로젝트를 통해 배운 점 또는 시사점
- 앞으로 데이터를 살펴볼 때 생각해보아야 하는 다양한 이슈에 대해 알게 되었다.
- 어떤 케이스에서는 노이즈를 주입하는 것이 오히려 모델의 강건성을 키울 수 있다는 것을 알게 되었다.

## Reference
- py-hanspell
    - https://github.com/ssut/py-hanspell
- g2p
    - https://github.com/Kyubyong/g2pK
- cleanlab
    - https://docs.cleanlab.ai/stable/index.html
- AI Hub
    - https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=577
- Naver News Crawler(BS4)
    - https://bigdata-doctrine.tistory.com/34
- BackTranslation
    - https://huggingface.co/KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-ko2en
    - https://huggingface.co/KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-en2ko
