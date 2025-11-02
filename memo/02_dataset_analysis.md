# ml-100k 데이터셋 분석

## 데이터셋 개요
- **이름**: MovieLens 100k Dataset
- **출처**: GroupLens Research Project (University of Minnesota)
- **수집 기간**: 1997년 9월 19일 ~ 1998년 4월 22일

## 데이터 규모
- **평점 수**: 100,000개
- **사용자 수**: 943명
- **영화 수**: 1,682개
- **평점 범위**: 1-5점
- **조건**: 각 사용자는 최소 20개 영화 평가

## 주요 파일

### 1. u.data (평점 데이터)
- **형식**: TSV (Tab-Separated Values)
- **컬럼**: user_id | movie_id | rating | timestamp
- **특징**: 무작위 순서로 정렬됨

### 2. u.item (영화 정보)
- **형식**: TSV
- **컬럼**:
  - movie_id
  - title (제목)
  - release_date
  - video_release_date
  - IMDb_URL
  - 19개 장르 컬럼 (one-hot encoding)
- **장르 종류**: unknown, Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western

### 3. u.user (사용자 정보)
- **형식**: TSV
- **컬럼**: user_id | age | gender | occupation | zip_code
- **특징**: 인구통계학적 정보 포함

### 4. u.genre (장르 목록)
- 전체 19개 장르 목록

### 5. u.occupation (직업 목록)
- 사용자 직업 카테고리

## 데이터 분할 파일
- **u1.base ~ u5.base/test**: 80%/20% 5-fold cross validation
- **ua.base/test, ub.base/test**: 사용자당 정확히 10개 평점을 테스트셋에 포함

## 인용 정보
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent
Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
DOI=http://dx.doi.org/10.1145/2827872
