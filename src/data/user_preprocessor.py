"""
사용자 데이터 전처리 모듈
- 나이 그룹 분류
- 성별 인코딩
- 직업 인코딩
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any

class UserPreprocessor:
    """사용자 데이터 전처리 클래스"""
    
    def __init__(self, data_dir: str):
        # data_dir이 디렉토리면 u.user를 찾고, 파일이면 그대로 사용
        import os
        if os.path.isdir(data_dir):
            self.user_path = os.path.join(data_dir, 'u.user')
        else:
            self.user_path = data_dir
        self.le_occupation = LabelEncoder()
        
    def categorize_age(self, age: int) -> int:
        """나이를 그룹별로 분류"""
        if age < 10:
            return 0
        elif age < 20:
            return 1
        elif age < 30:
            return 2
        elif age < 40:
            return 3
        elif age < 50:
            return 4
        elif age < 60:
            return 5
        elif age < 70:
            return 6
        else:
            return 7
    
    def gender_to_int(self, gender: str) -> int:
        """성별을 정수로 변환"""
        if gender == 'M':
            return 0
        else:
            return 1
    
    def load_user_data(self) -> pd.DataFrame:
        """사용자 데이터 로드"""
        columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        user_df = pd.read_csv(self.user_path, sep="|", names=columns)
        
        # zip_code 제거 (필요없음)
        user_df.drop('zip_code', axis=1, inplace=True)
        
        return user_df
    
    def process_user_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """전체 사용자 데이터 전처리 파이프라인"""
        print("사용자 데이터 전처리 시작...")
        
        # 1. 데이터 로드
        user_df = self.load_user_data()
        print(f"  로드된 사용자 수: {len(user_df)}")
        
        # 2. 나이 그룹 분류
        user_df['age_group_label'] = user_df['age'].apply(self.categorize_age)
        original_ages = user_df['age'].copy()
        user_df.drop('age', axis=1, inplace=True)
        
        # 3. 성별 인코딩
        user_df['gender_int'] = user_df['gender'].apply(self.gender_to_int)
        user_df.drop('gender', axis=1, inplace=True)
        
        # 4. 직업 인코딩
        user_df['occupation_encoded'] = self.le_occupation.fit_transform(user_df['occupation'])
        occupation_mapping = dict(zip(self.le_occupation.classes_, self.le_occupation.transform(self.le_occupation.classes_)))
        user_df.drop('occupation', axis=1, inplace=True)
        
        print("사용자 데이터 전처리 완료!")
        print(f"  최종 컬럼: {list(user_df.columns)}")
        print(f"  나이 그룹 수: {user_df['age_group_label'].nunique()}")
        print(f"  직업 수: {len(occupation_mapping)}")
        
        # 메타데이터 반환
        metadata = {
            'num_users': len(user_df),
            'num_age_groups': user_df['age_group_label'].nunique(),
            'num_occupations': len(occupation_mapping),
            'occupation_mapping': occupation_mapping,
            'original_ages': original_ages
        }
        
        return user_df, metadata
    
    def get_user_features(self, user_df: pd.DataFrame) -> np.ndarray:
        """사용자 특성 벡터 생성"""
        feature_columns = ['age_group_label', 'gender_int', 'occupation_encoded']
        return user_df[feature_columns].values.astype(np.float32)