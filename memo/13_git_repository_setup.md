# Git Repository 설정 및 GitHub 업로드

## 작성 일자
2025년 11월 3일

---

## 작업 개요

claudeMoE 프로젝트를 GitHub 저장소 https://github.com/hyeondata/recsys.git의 새로운 브랜치 'moe'로 업로드했습니다.

---

## Git 설정 과정

### 1. Git 저장소 초기화

```bash
# 현재 상태 확인 - Git 저장소가 아님
git status
# 출력: fatal: (현재 폴더 또는 상위 폴더 중 일부가) 깃 저장소가 아닙니다

# Git 저장소 초기화
git init
# 출력: /home/gpu/claudeMoE/.git/ 안의 빈 깃 저장소를 다시 초기화했습니다
```

### 2. .gitignore 파일 생성

프로젝트에 불필요한 파일들을 제외하기 위해 .gitignore 생성:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual Environment
.venv/
venv/
ENV/
env/

# Data
ml-100k/
*.csv
*.dat

# Checkpoints & Models
checkpoints/
*.pt
*.pth
*.ckpt

# Results & Logs
results/
runs/
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Claude Code
.claude/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Misc
*.bak
*.tmp
.cache/
```

**제외된 항목들**:
- **ml-100k/**: 데이터셋 (100MB 크기, GitHub에 불필요)
- **checkpoints/**: 학습된 모델 파일 (.pt, .pth)
- **results/**: 실험 결과 파일
- **.venv/**: Python 가상환경
- **.claude/**: Claude Code 설정

**포함된 항목들**:
- 소스 코드 (src/)
- 문서 (memo/, memo_compression/, README.md)
- 의존성 관리 (pyproject.toml, uv.lock)

> ⚠️ **참고**: v2.0부터 configs/ 디렉토리는 제거되었습니다 (모든 설정은 CLI 인자로 제공)

### 3. Git 사용자 설정

```bash
# Git 사용자 이름 설정
git config --global user.name "hyeondata"

# Git 이메일 설정
git config --global user.email "kimyonghyeondata@gmail.com"

# 설정 확인
git config --global --list | grep user
# 출력:
# user.name=hyeondata
# user.email=kimyonghyeondata@gmail.com
```

### 4. 원격 저장소 연결

```bash
# 원격 저장소 추가
git remote add origin https://github.com/hyeondata/recsys.git

# 원격 저장소 확인
git remote -v
# 출력:
# origin	https://github.com/hyeondata/recsys.git (fetch)
# origin	https://github.com/hyeondata/recsys.git (push)
```

### 5. 'moe' 브랜치 생성 및 전환

```bash
# 새로운 브랜치 'moe' 생성 및 전환
git checkout -b moe
# 출력: 새로 만든 'moe' 브랜치로 전환합니다

# 현재 브랜치 확인
git branch
# 출력: * moe
```

### 6. 파일 스테이징

```bash
# 모든 파일 추가
git add .

# 스테이징된 파일 확인
git status
```

**스테이징된 파일 목록** (v1.0 초기 커밋):

> ⚠️ **참고**: 아래 목록은 v1.0 초기 커밋 내용입니다. v2.0에서 일부 파일이 제거되거나 추가되었습니다.
> - **제거됨**: configs/, test_data_loading.py, compare_models.py
> - **추가됨**: train_ppo_moe_standard.py, train_grpo_moe_standard.py, paper_metrics_comparison.py, memo_compression/

```
새 파일:       .gitignore
새 파일:       README.md
새 파일:       configs/dense_moe.yaml (v2.0에서 제거됨)
새 파일:       configs/grpo_moe.yaml (v2.0에서 제거됨)
새 파일:       configs/ppo_moe.yaml (v2.0에서 제거됨)
새 파일:       memo/01_project_overview.md
새 파일:       memo/02_dataset_analysis.md
새 파일:       memo/03_code_structure.md
새 파일:       memo/04_model_implementation.md
새 파일:       memo/05_training_implementation.md
새 파일:       memo/06_current_work_status.md
새 파일:       memo/07_experiment_results.md
새 파일:       memo/08_final_results_and_summary.md
새 파일:       memo/09_model_architectures_visualization.md
새 파일:       memo/10_visualization_guide.md
새 파일:       memo/11_batch_size_optimization.md
새 파일:       memo/12_code_refactoring_tqdm.md
새 파일:       memo/moe_research.code-workspace
새 파일:       requirements.txt
새 파일:       src/data/__init__.py
새 파일:       src/data/dataset.py
새 파일:       src/data/enhanced_dataset.py
새 파일:       src/data/movie_preprocessor.py
새 파일:       src/data/user_preprocessor.py
새 파일:       src/models/__init__.py
새 파일:       src/models/base_moe.py
새 파일:       src/models/dense_moe.py
새 파일:       src/models/expert_network.py
새 파일:       src/models/grpo_moe.py
새 파일:       src/models/ppo_moe.py
새 파일:       src/training/__init__.py
새 파일:       src/training/evaluate.py
새 파일:       src/training/train_dense_moe.py
새 파일:       src/training/train_grpo_moe.py
새 파일:       src/training/train_ppo_moe.py
새 파일:       src/utils/__init__.py
새 파일:       src/utils/metrics.py
새 파일:       src/utils/trainer_utils.py
새 파일:       src/visualization/compare_models.py (v2.0에서 제거됨)
새 파일:       test_data_loading.py (v2.0에서 제거됨)
```

**v2.0에서 추가된 파일들**:
- src/training/train_ppo_moe_standard.py
- src/training/train_grpo_moe_standard.py
- src/visualization/paper_metrics_comparison.py
- memo/14_batch_loading_fix.md
- memo/15_publication_visualization.md
- memo/16_standard_rl_implementation.md
- memo_compression/ (전체 폴더)

### 7. 커밋 생성

```bash
git commit -m "feat: Implement MoE-based Movie Recommendation System

Implemented three Mixture of Experts (MoE) models for movie recommendation:
- Dense MoE: Baseline model with FC-based gating network
- PPO-MoE: Reinforcement learning with Proximal Policy Optimization
- GRPO-MoE: Group Relative Policy Optimization for expert selection

Features:
- Complete data preprocessing pipeline for MovieLens 100k
- Expert network architecture with 8 specialized experts
- Training scripts with tqdm progress bars
- Comprehensive evaluation and visualization tools
- Extensive documentation in memo/ folder

Experimental Results:
- Dense MoE: RMSE 0.9803 (best performance)
- GRPO-MoE: RMSE 1.0412
- PPO-MoE: RMSE 1.0659"
```

**커밋 결과**:
```
[moe (최상위-커밋) a8ac611] feat: Implement MoE-based Movie Recommendation System
 40 files changed, 9012 insertions(+)
```

**통계**:
- **커밋 해시**: a8ac611
- **파일 수**: 40개
- **총 추가된 라인**: 9,012줄

### 8. GitHub에 푸시

```bash
# 원격 저장소에 푸시 (upstream 설정 포함)
git push -u origin moe
```

**푸시 결과**:
```
branch 'moe' set up to track 'origin/moe'.
remote:
remote: Create a pull request for 'moe' on GitHub by visiting:
remote:      https://github.com/hyeondata/recsys/pull/new/moe
remote:
To https://github.com/hyeondata/recsys.git
 * [new branch]      moe -> moe
```

---

## 최종 Git 구조

### 브랜치 정보
- **로컬 브랜치**: moe
- **원격 브랜치**: origin/moe
- **트래킹 설정**: moe → origin/moe

### 커밋 정보
```bash
git log --oneline -1
# a8ac611 feat: Implement MoE-based Movie Recommendation System
```

### 원격 저장소
- **URL**: https://github.com/hyeondata/recsys.git
- **브랜치**: moe
- **커밋 수**: 1개
- **파일 수**: 40개

---

## GitHub에서 확인 가능한 내용

### 1. 소스 코드
```
src/
├── data/           # 데이터 전처리
├── models/         # MoE 모델 (Dense, PPO, GRPO)
├── training/       # 학습 스크립트
├── utils/          # 유틸리티
└── visualization/  # 시각화
```

### 2. 시각화 도구
```
src/visualization/
├── publication_plots.py
├── paper_metrics_comparison.py
├── training_curves.py
└── README.md
```

### 3. 문서
```
memo/
├── 01_project_overview.md
├── 02_dataset_analysis.md
├── 03_code_structure.md
├── 04_model_implementation.md
├── 05_training_implementation.md
├── 06_current_work_status.md
├── 07_experiment_results.md
├── 08_final_results_and_summary.md
├── 09_model_architectures_visualization.md
├── 10_visualization_guide.md
├── 11_batch_size_optimization.md
└── 12_code_refactoring_tqdm.md
```

### 4. 기타
- README.md
- requirements.txt
- .gitignore

---

## Pull Request 생성 (선택 사항)

GitHub에서 제공하는 링크를 통해 Pull Request 생성 가능:

```
https://github.com/hyeondata/recsys/pull/new/moe
```

### PR 제목 (제안)
```
[Feature] Implement MoE-based Movie Recommendation System
```

### PR 설명 (제안)
```markdown
## 개요
MovieLens 100k 데이터셋을 활용한 Mixture of Experts (MoE) 기반 영화 추천 시스템 구현

## 주요 기능
- 3가지 MoE 모델 구현
  - Dense MoE: FC Layer 기반 Gating (베이스라인)
  - PPO-MoE: Proximal Policy Optimization
  - GRPO-MoE: Group Relative Policy Optimization

- 완전한 학습 파이프라인
  - 데이터 전처리
  - 모델 학습 및 평가
  - 결과 시각화

## 실험 결과
| 모델 | RMSE | MAE |
|------|------|-----|
| Dense MoE | 0.9803 | 0.7756 |
| GRPO-MoE | 1.0412 | 0.8276 |
| PPO-MoE | 1.0659 | 0.8575 |

## 문서화
- 12개의 상세 문서 (memo/ 폴더)
- 코드 주석 및 docstring
- README.md 사용 가이드
```

---

## 향후 Git 작업 가이드

### 새로운 기능 추가 시

```bash
# 최신 상태로 업데이트
git pull origin moe

# 변경사항 확인
git status

# 파일 추가
git add <파일명>

# 커밋
git commit -m "feat: <기능 설명>"

# 푸시
git push origin moe
```

### 실험 결과 업데이트 시

```bash
# memo 파일 수정 후
git add memo/<파일명>.md

# 커밋
git commit -m "docs: Update experiment results"

# 푸시
git push origin moe
```

### 버그 수정 시

```bash
git add <수정된_파일>
git commit -m "fix: <버그 설명>"
git push origin moe
```

---

## Git 커밋 메시지 컨벤션

### 타입
- **feat**: 새로운 기능 추가
- **fix**: 버그 수정
- **docs**: 문서 수정
- **refactor**: 코드 리팩토링
- **test**: 테스트 코드
- **style**: 코드 포맷팅
- **chore**: 빌드, 설정 파일 수정

### 예시
```bash
feat: Add curriculum learning to PPO-MoE
fix: Fix expert selection bug in GRPO-MoE
docs: Update README with installation guide
refactor: Simplify data preprocessing pipeline
test: Add unit tests for expert network
```

---

## 데이터 및 결과물 관리

### GitHub에 업로드하지 않은 항목
이런 파일들은 로컬에서만 관리:

```bash
# 데이터셋 (100MB)
ml-100k/

# 학습된 모델 (각 8-9MB)
checkpoints/
├── dense_moe/dense_moe_best.pt
├── ppo_moe/ppo_moe_best.pt
└── grpo_moe/grpo_moe_best.pt

# 실험 결과
results/
└── all_models_evaluation.json
```

### 대용량 파일 공유 방법 (필요 시)
1. **Google Drive**: 체크포인트 공유
2. **Hugging Face Hub**: 모델 배포
3. **Weights & Biases**: 실험 로그
4. **Git LFS**: 대용량 파일 버전 관리 (선택)

---

## 저장소 통계

### 코드 통계
```bash
# 총 라인 수 확인
find src -name "*.py" | xargs wc -l
# 출력: 약 9,000+ 줄
```

### 파일 구성
- **Python 소스**: 21개
- **YAML 설정**: 3개
- **Markdown 문서**: 13개 (README + memo)
- **기타**: 3개 (requirements.txt, .gitignore, test)

---

## 백업 및 복구

### 로컬 백업
```bash
# 전체 프로젝트 복사 (데이터 포함)
cp -r ~/claudeMoE ~/claudeMoE_backup_20251103
```

### GitHub에서 복구
```bash
# 새로운 위치에 클론
git clone -b moe https://github.com/hyeondata/recsys.git claudeMoE_recovered

# 데이터셋은 별도로 다운로드
cd claudeMoE_recovered
# MovieLens 100k 다운로드 및 압축 해제
```

---

## 협업 가이드 (필요 시)

### Fork & Pull Request 방식
1. 저장소 Fork
2. 기능 브랜치 생성: `git checkout -b feature/new-feature`
3. 변경사항 커밋
4. Fork된 저장소에 푸시
5. Pull Request 생성

### 코드 리뷰 체크리스트
- [ ] 코드 스타일 일관성
- [ ] Docstring 작성
- [ ] 테스트 통과
- [ ] 문서 업데이트
- [ ] 실험 결과 검증

---

## 성공 체크리스트

✅ Git 저장소 초기화 완료
✅ .gitignore 설정 완료
✅ Git 사용자 설정 완료
✅ 원격 저장소 연결 완료
✅ 'moe' 브랜치 생성 완료
✅ 40개 파일 커밋 완료 (9,012줄)
✅ GitHub에 푸시 완료
✅ 문서화 완료

---

## 결론

claudeMoE 프로젝트를 성공적으로 GitHub에 업로드했습니다!

### 저장소 정보
- **URL**: https://github.com/hyeondata/recsys
- **브랜치**: moe
- **커밋**: a8ac611
- **파일**: 40개
- **라인**: 9,012줄

### 다음 단계
1. 필요 시 Pull Request 생성
2. 추가 실험 진행
3. 결과 업데이트
4. 코드 개선 및 최적화

---

*작성자: hyeondata (kimyonghyeondata@gmail.com)*
*작성일: 2025년 11월 3일*
*프로젝트: claudeMoE - Git Repository Setup*
