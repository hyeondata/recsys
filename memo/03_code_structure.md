# 코드 구조 분석

## src/data/ 폴더

### 1. movie_preprocessor.py
**클래스**: `MoviePreprocessor`

**주요 기능**:
- 영화 데이터 로드 및 전처리
- 영화 제목 정제 (년도 추출, 특수문자 제거, 소문자 변환)
- 단어 사전(vocabulary) 구축
- 제목을 숫자 시퀀스로 인코딩
- 장르 one-hot 특성 추출

**주요 메서드**:
- `load_movie_data()`: u.item 파일 로드
- `preprocess_title()`: 제목 정제 (년도 분리, 특수문자 제거)
- `build_vocab()`: 단어 빈도 기반 사전 구축 (특수 토큰: `<PAD>`, `<UNK>`)
- `encode_titles()`: 제목을 정수 시퀀스로 변환 (max_length=10, 패딩 포함)
- `process_movie_data()`: 전체 파이프라인 실행
- `get_genre_features()`: 장르 특성 추출

**출력**:
- movie_df: 전처리된 영화 데이터프레임
- vocab: 단어 사전 딕셔너리
- encoded_titles: 인코딩된 제목 numpy 배열

---

### 2. user_preprocessor.py
**클래스**: `UserPreprocessor`

**주요 기능**:
- 사용자 데이터 로드 및 전처리
- 나이를 8개 그룹으로 분류 (0:<10, 1:10-19, ..., 7:70+)
- 성별 인코딩 (M:0, F:1)
- 직업 라벨 인코딩 (LabelEncoder 사용)

**주요 메서드**:
- `load_user_data()`: u.user 파일 로드
- `categorize_age()`: 나이를 8개 그룹으로 분류
- `gender_to_int()`: 성별을 정수로 변환
- `process_user_data()`: 전체 파이프라인 실행
- `get_user_features()`: 사용자 특성 벡터 생성

**출력**:
- user_df: 전처리된 사용자 데이터프레임
  - 컬럼: user_id, age_group_label, gender_int, occupation_encoded
- metadata: 메타데이터 딕셔너리
  - num_users, num_age_groups, num_occupations, occupation_mapping, original_ages

---

### 3. dataset.py
**클래스**: `MovieRatingDataset` (PyTorch Dataset 상속)

**주요 기능**:
- 영화 제목과 평점 데이터셋
- u.data (평점)와 영화 정보 결합

**초기화 파라미터**:
- movie_data: 전처리된 영화 데이터
- rating_path: u.data 파일 경로
- vocab: 단어 사전
- max_length: 제목 최대 길이 (기본값: 10)

**반환 데이터** (`__getitem__`):
```python
{
    'encoded_title': torch.Tensor,  # 인코딩된 제목 [max_length]
    'rating': torch.Tensor,         # 정규화된 평점 (0-1)
    'user_id': torch.Tensor,        # 사용자 ID (0부터 시작)
    'movie_id': torch.Tensor        # 영화 ID (0부터 시작)
}
```

**특징**:
- 평점 정규화: (rating - 1) / 4.0 → [0, 1] 범위
- ID 조정: 원본은 1부터 시작 → 0부터 시작으로 변환

---

### 4. enhanced_dataset.py
**클래스**: `EnhancedMovieRatingDataset` (PyTorch Dataset 상속)

**주요 기능**:
- MovieRatingDataset의 확장 버전
- 사용자 특성(나이, 성별, 직업) 추가 포함

**초기화 파라미터**:
- movie_data: 전처리된 영화 데이터
- user_data: 전처리된 사용자 데이터
- rating_path: u.data 파일 경로
- vocab: 단어 사전
- max_length: 제목 최대 길이 (기본값: 10)

**반환 데이터** (`__getitem__`):
```python
{
    'encoded_title': torch.Tensor,  # 인코딩된 제목
    'rating': torch.Tensor,         # 정규화된 평점
    'user_id': torch.Tensor,        # 사용자 ID
    'movie_id': torch.Tensor,       # 영화 ID
    'age_group': torch.Tensor,      # 나이 그룹 (0-7)
    'gender': torch.Tensor,         # 성별 (0:M, 1:F)
    'occupation': torch.Tensor      # 직업 인코딩
}
```

**특징**:
- 기본 데이터셋의 모든 기능 포함
- 사용자 특성 3가지 추가 (age_group, gender, occupation)

---

## src/models/ 폴더

**현재 상태**: 비어있음

**예상 구현 내용**:
1. 기본 모델: 영화 제목 기반 평점 예측
2. 향상된 모델: 영화 제목 + 사용자 특성 기반 예측
3. 가능한 아키텍처:
   - Embedding layer (제목 단어, 사용자 ID, 영화 ID)
   - LSTM/GRU (제목 시퀀스 처리)
   - Fully connected layers (특성 결합 및 평점 예측)

---

## 데이터 흐름

### 기본 워크플로우
```
1. MoviePreprocessor.process_movie_data()
   └─> movie_df, vocab, encoded_titles

2. UserPreprocessor.process_user_data()
   └─> user_df, metadata

3. MovieRatingDataset 또는 EnhancedMovieRatingDataset
   └─> PyTorch DataLoader로 배치 생성

4. 모델 학습 (아직 구현 필요)
```

### 주요 특징
- 모든 전처리 로직이 명확히 분리됨
- PyTorch Dataset 표준 인터페이스 준수
- 기본 데이터셋과 향상된 데이터셋 분리로 실험 용이
