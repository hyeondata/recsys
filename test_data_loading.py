#!/usr/bin/env python3
"""
간단한 데이터 로딩 테스트
"""
import sys
import torch
from torch.utils.data import DataLoader

# src 디렉토리를 path에 추가
sys.path.append('src')

from data.movie_preprocessor import MoviePreprocessor
from data.user_preprocessor import UserPreprocessor
from data.enhanced_dataset import EnhancedMovieRatingDataset

def test_data_loading():
    print("=" * 50)
    print("데이터 로딩 테스트 시작")
    print("=" * 50)

    # 1. 영화 데이터 전처리
    print("\n[1] 영화 데이터 전처리...")
    movie_preprocessor = MoviePreprocessor("ml-100k")
    movie_df, vocab, encoded_titles = movie_preprocessor.process_movie_data()
    print(f"  - 영화 개수: {len(movie_df)}")
    print(f"  - 단어 사전 크기: {len(vocab)}")
    print(f"  - 인코딩된 제목 shape: {encoded_titles.shape}")

    # 2. 사용자 데이터 전처리
    print("\n[2] 사용자 데이터 전처리...")
    user_preprocessor = UserPreprocessor("ml-100k")
    user_df, user_metadata = user_preprocessor.process_user_data()
    print(f"  - 사용자 개수: {len(user_df)}")
    print(f"  - 나이 그룹 개수: {user_metadata['num_age_groups']}")
    print(f"  - 직업 개수: {user_metadata['num_occupations']}")

    # 3. 데이터셋 생성
    print("\n[3] 학습 데이터셋 생성...")
    train_dataset = EnhancedMovieRatingDataset(
        movie_data=movie_df,
        user_data=user_df,
        rating_path="ml-100k/u1.base",
        vocab=vocab
    )
    print(f"  - 학습 데이터 크기: {len(train_dataset)}")

    print("\n[4] 검증 데이터셋 생성...")
    val_dataset = EnhancedMovieRatingDataset(
        movie_data=movie_df,
        user_data=user_df,
        rating_path="ml-100k/u1.test",
        vocab=vocab
    )
    print(f"  - 검증 데이터 크기: {len(val_dataset)}")

    # 4. DataLoader 테스트
    print("\n[5] DataLoader 테스트...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0
    )

    # 첫 배치 로드
    batch = next(iter(train_loader))
    print(f"  - Batch keys: {list(batch.keys())}")
    print(f"  - user_id shape: {batch['user_id'].shape}")
    print(f"  - movie_id shape: {batch['movie_id'].shape}")
    print(f"  - age_group shape: {batch['age_group'].shape}")
    print(f"  - gender shape: {batch['gender'].shape}")
    print(f"  - occupation shape: {batch['occupation'].shape}")
    print(f"  - rating shape: {batch['rating'].shape}")

    # 5. GPU 확인
    print("\n[6] GPU 확인...")
    if torch.cuda.is_available():
        print(f"  - GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("  - GPU 사용 불가능 (CPU 사용)")

    print("\n" + "=" * 50)
    print("데이터 로딩 테스트 완료 ✓")
    print("=" * 50)

    return {
        'num_users': user_metadata['num_users'],
        'num_movies': len(movie_df),
        'num_age_groups': user_metadata['num_age_groups'],
        'num_occupations': user_metadata['num_occupations'],
        'vocab_size': len(vocab),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
    }

if __name__ == "__main__":
    try:
        metadata = test_data_loading()
        print("\n반환된 메타데이터:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
