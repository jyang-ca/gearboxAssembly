# Long Trajectory Gear Assembly - 강화학습 접근 방법 전체 컨텍스트

## 목차
1. [현상 (Current Situation)](#현상-current-situation)
2. [방법 (Approach)](#방법-approach)
3. [결과 (Results)](#결과-results)

---

## 현상 (Current Situation)

### 1.1 프로젝트 개요

**과제**: 이중 팔 로봇(Galaxea R1)을 사용하여 행성 기어박스(Planetary Gearbox) 조립 작업을 강화학습으로 학습

**조립 시퀀스**: 6개의 기어를 순차적으로 조립
1. Sun Planetary Gear 1
2. Sun Planetary Gear 2
3. Sun Planetary Gear 3
4. Sun Planetary Gear 4 (중앙)
5. Planetary Carrier (링 기어 위에 배치)
6. Planetary Reducer (기어 4 위에 배치)

**정밀도 요구사항**: 1cm 이내의 정밀도로 조립 완료

### 1.2 환경 설정

**시뮬레이션 환경**:
- **프레임워크**: Isaac Lab (NVIDIA Omniverse 기반)
- **물리 엔진**: PhysX GPU
- **시뮬레이션 시간 간격**: 0.01초 (sim_dt)
- **디시메이션**: 5 (실제 제어 주기는 0.05초)
- **에피소드 길이**: 최대 120초
- **병렬 환경 수**: 4096개 (기본값)

**로봇 설정**:
- **로봇**: Galaxea R1 Challenge (이중 팔 로봇)
- **액션 공간**: 14차원
  - 왼쪽 팔: 6 DOF
  - 오른쪽 팔: 6 DOF
  - 왼쪽 그리퍼: 1 DOF
  - 오른쪽 그리퍼: 1 DOF
- **관찰 공간**: 69차원
  - 관절 위치: 14차원 (6+6+1+1)
  - 관절 속도: 14차원 (6+6+1+1)
  - 엔드 이펙터 포즈: 14차원 (3+4+3+4)
  - 기어 관찰: 18차원 (위치, 방향 등)
  - 인코딩: 9차원 (서브태스크/기어 타입 인코딩)

**물리 설정**:
- 테이블 마찰 계수: 0.4
- 기어 마찰 계수: 0.01
- 그리퍼 마찰 계수: 2.0
- GPU 충돌 스택 크기: 2^30 (4096 환경 처리용)

### 1.3 현재 학습 상태

**체크포인트 정보**:
- **위치**: `logs/rl_games/Galaxea-LongTrajectoryAssembly-Direct-v0/LongTrajectoryAssembly/nn/`
- **파일명**: `last_Galaxea-LongTrajectoryAssembly-Direct-v0_ep_100_rew_229.06622.pth`
- **크기**: 2.4 MB
- **에피소드**: 100번째
- **평균 보상**: 229.07

**학습 로그**:
- 최근 학습 완료: 2026-01-04 09:05:57
- Policy_Approach (Shared): FAILED (초기 학습 실패)

**Hugging Face 업로드**:
- Repository: `yjsm1203/gearboxAssembly-checkpoints`
- 업로드된 파일:
  - Checkpoint 파일 2개 (.pth)
  - 설정 파일 4개 (.yaml, .pkl)

---

## 방법 (Approach)

### 2.1 아키텍처 설계: 8-Policy 구조 (Phase 2)

전체 조립 작업을 8개의 정책으로 분해:

**공유 정책 (Shared Policies)**:
1. **Policy_Approach**: 모든 기어에 대해 접근 동작 학습
2. **Policy_Grasp**: 모든 기어에 대해 파지 동작 학습

**기어별 정책 (Gear-Specific Policies)**:
3. **Policy_Transport_Gear1**: 기어 1을 핀 위치로 운반
4. **Policy_Transport_Gear2**: 기어 2를 핀 위치로 운반
5. **Policy_Transport_Gear3**: 기어 3을 핀 위치로 운반
6. **Policy_Transport_Gear4**: 기어 4를 중앙 위치로 운반
7. **Policy_Transport_Carrier**: 캐리어를 링 기어 위에 배치
8. **Policy_Transport_Reducer**: 리듀서를 기어 4 위에 배치

**설계 이유**:
- 접근과 파지는 모든 기어에 공통적이므로 공유 정책 사용
- 운반은 각 기어의 목표 위치가 다르므로 기어별 정책 필요
- 모듈화를 통해 학습 안정성 향상 및 디버깅 용이

### 2.2 환경 기반 상태 전이 (Environment-Based State Transitions)

**서브태스크 구조**:
각 기어 조립은 3단계로 구성:
1. **Approach**: 엔드 이펙터를 기어 위의 pre-grasp 위치로 이동
2. **Grasp**: 그리퍼를 닫아 기어를 파지하고 들어올림
3. **Transport**: 기어를 목표 위치로 운반하여 조립

**규칙 기반 전이 (Rule-Based Transitions)**:
- 환경이 자동으로 서브태스크 완료를 감지하고 다음 단계로 전이
- 정책은 현재 서브태스크에만 집중
- 전이 보너스: 5.0 (서브태스크 완료 시)

**전이 조건**:
- **Approach → Grasp**: 
  - 수평 거리 < 3cm
  - 높이 차이 < 2cm
  - 그리퍼 방향 정렬 (quaternion dot > 0.95)
  - 그리퍼 열림 상태
- **Grasp → Transport**:
  - 그리퍼 닫힘 (normalized position > 0.8)
  - 접촉 힘 > 2.0N
  - 기어가 테이블에서 10cm 이상 들어올림
- **Transport → Approach (다음 기어)**:
  - 위치 정밀도 < 1cm
  - 방향 정렬 (각도 < 0.1 rad)
  - 안정성 (속도 < 0.01 m/s)

### 2.3 보상 함수 설계

#### 2.3.1 Approach 보상
```python
reward_approach = (
    exp(-horizontal_distance / 0.05) * 0.1 +      # 수평 정렬
    exp(-height_diff / 0.05) * 0.1 +                # 높이 정렬
    quaternion_dot * 0.1 +                          # 방향 정렬
    gripper_open * 0.05 +                           # 그리퍼 열림
    completion_bonus * 1.0                          # 완료 보너스
)
```

#### 2.3.2 Grasp 보상
```python
reward_grasp = (
    gripper_closed * 0.1 +                          # 그리퍼 닫힘
    contact_force_reward * 0.1 +                    # 접촉 힘
    lift_height / 0.1 * 0.1 +                       # 들어올림
    completion_bonus * 2.0                          # 완료 보너스
)
```

#### 2.3.3 Transport 보상
```python
reward_transport = (
    exp(-horizontal_distance / 0.01) * 0.2 +        # 수평 정렬
    exp(-height_diff / 0.01) * 0.2 +                 # 높이 정렬
    quaternion_dot * 0.1 +                          # 방향 정렬
    exp(-velocity / 0.01) * 0.1 +                   # 안정성
    completion_bonus * 10.0                         # 완료 보너스 (가장 높음)
)
```

**보상 설계 원칙**:
- 각 서브태스크의 목표에 맞춘 보상 구조
- 완료 보너스는 Transport가 가장 높음 (가장 어려운 작업)
- 지수 함수를 사용하여 목표에 가까울수록 높은 보상
- 시간 페널티: -0.001 (에피소드당)

### 2.4 강화학습 알고리즘 및 하이퍼파라미터

**알고리즘**: A2C (Advantage Actor-Critic) Continuous
- **변형**: PPO 스타일 클리핑 사용 (`ppo: True`)
- **정책 타입**: Continuous A2C with fixed logstd

**네트워크 구조**:
- **타입**: Actor-Critic (공유 네트워크, `separate: False`)
- **Hidden Layers**: [512, 256, 128]
- **활성화 함수**: ELU
- **입력 정규화**: True (clip: 5.0)
- **액션 정규화**: True (clip: 1.0)

**학습 하이퍼파라미터**:
```yaml
learning_rate: 3e-4
lr_schedule: adaptive
gamma: 0.99                    # 할인 계수
tau: 0.95                       # GAE 람다
kl_threshold: 0.008            # KL 발산 임계값
e_clip: 0.2                    # PPO 클리핑 범위
horizon_length: 32             # 롤아웃 길이
minibatch_size: 16384          # 미니배치 크기
mini_epochs: 8                 # 미니배치 에폭 수
grad_norm: 1.0                 # 그래디언트 클리핑
entropy_coef: 0.001            # 엔트로피 계수
critic_coef: 2                 # 크리틱 손실 가중치
normalize_advantage: True      # 어드밴티지 정규화
value_bootstrap: True          # 값 부트스트랩
```

**학습 설정**:
- **최대 에폭**: 5000 (기본값)
- **체크포인트 저장 주기**: 100 에폭마다
- **시드**: 42

### 2.5 학습 전략

#### 2.5.1 순차 학습 (Sequential Training)
`train_all_subtask_policies.py` 스크립트를 사용하여 8개 정책을 순차적으로 학습:

1. Policy_Approach 학습 완료
2. Policy_Grasp 학습 완료
3. Policy_Transport_Gear1~4, Carrier, Reducer 순차 학습

**장점**:
- 각 정책이 독립적으로 학습되어 디버깅 용이
- 이전 정책을 고정하고 다음 정책만 학습 가능
- 실패한 정책만 재학습 가능

#### 2.5.2 전체 시퀀스 학습 (Full Sequence Training)
`train_long_trajectory_assembly.py` 스크립트를 사용하여 전체 시퀀스를 한 번에 학습:

- 모든 서브태스크를 하나의 정책으로 학습
- 환경이 자동으로 서브태스크 전이 관리
- 더 복잡하지만 end-to-end 학습 가능

**현재 사용 중**: Full sequence training (subtask="full")

### 2.6 관찰 공간 설계

**69차원 관찰 벡터 구성**:
1. **로봇 상태 (28차원)**:
   - 관절 위치: 14차원
   - 관절 속도: 14차원

2. **엔드 이펙터 상태 (14차원)**:
   - 왼쪽 EE 위치: 3차원
   - 왼쪽 EE 방향 (quaternion): 4차원
   - 오른쪽 EE 위치: 3차원
   - 오른쪽 EE 방향 (quaternion): 4차원

3. **기어 상태 (18차원)**:
   - 현재 조립 중인 기어의 위치, 방향, 속도 등

4. **인코딩 (9차원)**:
   - 현재 서브태스크 타입 인코딩: 3차원
   - 현재 기어 타입 인코딩: 6차원

**설계 원칙**:
- 각 서브태스크에 필요한 정보만 포함
- 정규화된 관찰값 사용 (clip: 5.0)
- 서브태스크/기어 타입을 원-핫 인코딩으로 제공

### 2.7 액션 공간 설계

**14차원 액션 벡터**:
- 각 관절에 대한 위치 명령 (normalized to [-1, 1])
- IK 컨트롤러를 통해 엔드 이펙터 포즈로 변환

**제어 방식**:
- Differential IK Controller 사용
- 명령 타입: Pose (위치 + 방향)
- IK 방법: DLS (Damped Least Squares)

---

## 결과 (Results)

### 3.1 중간 학습 결과

#### 3.1.1 현재 체크포인트 성능
- **에피소드**: 100
- **평균 보상**: 229.07
- **모델 크기**: 2.4 MB
  - 모델 가중치: ~0.77 MB (약 202,000 파라미터)
  - 메타데이터: ~1.63 MB

**모델 파라미터 분석**:
- Observation space: 69차원
- Action space: 14차원
- Hidden layers: [512, 256, 128]
- 총 파라미터: 201,999개
- 예상 모델 크기: 0.77 MB (정상 범위)

#### 3.1.2 학습 진행 상황
- **학습 시작**: 2026-01-04
- **최근 체크포인트**: 100 에폭
- **보상 추세**: 229.07 (초기 단계)

**보상 해석**:
- Approach 완료 보너스: +1.0
- Grasp 완료 보너스: +2.0
- Transport 완료 보너스: +10.0
- 전이 보너스: +5.0
- 평균 보상 229는 여러 서브태스크 완료를 의미

### 3.2 디버깅 및 이슈

#### 3.2.1 초기 학습 실패
**문제**: Policy_Approach (Shared) 학습 실패
- 로그: `training_log_20260104_090425.txt`에 "FAILED" 기록

**가능한 원인**:
1. 보상 스케일 불균형
2. 전이 조건이 너무 엄격함
3. 초기 정책이 목표에 도달하기 어려움

**해결 방안**:
- 보상 가중치 조정
- 전이 임계값 완화
- 커리큘럼 학습 도입 (curriculum_start_gear_idx 사용)

#### 3.2.2 체크포인트 크기 이슈
**질문**: 왜 체크포인트가 3MB밖에 안 되는가?

**답변**: 정상입니다
- 모델 구조가 작음 (69차원 입력, [512,256,128] hidden)
- RL-Games는 모델 가중치만 저장 (optimizer state 제외)
- 2.4MB는 모델 가중치(0.77MB) + 메타데이터(1.63MB)

**확인 방법**:
```python
# 파라미터 계산
obs_dim = 69
action_dim = 14
hidden = [512, 256, 128]
# 총 파라미터: ~202,000
# float32 크기: 202,000 * 4 bytes = 0.77 MB
```

#### 3.2.3 Git 설정 이슈
**문제**: Git user.name/user.email 미설정
**해결**: 
- `git config --global user.name "User"`
- `git config --global user.email "user@example.com"`

#### 3.2.4 GitHub 인증 이슈
**문제**: HTTPS 인증 실패
**해결**: 
- SSH 키 생성 및 등록
- 원격 저장소 URL을 SSH로 변경

### 3.3 예상 결과

#### 3.3.1 학습 완료 기준
**각 서브태스크 완료 조건**:

1. **Approach 완료**:
   - 수평 거리 < 3cm
   - 높이 차이 < 2cm
   - 그리퍼 방향 정렬 (dot > 0.95)
   - 그리퍼 열림

2. **Grasp 완료**:
   - 그리퍼 닫힘 > 0.8
   - 접촉 힘 > 2.0N
   - 기어 들어올림 > 10cm

3. **Transport 완료**:
   - 위치 정밀도 < 1cm
   - 방향 정렬 < 0.1 rad
   - 속도 < 0.01 m/s

**전체 조립 완료**:
- 6개 기어 모두 조립 완료
- 각 기어가 정밀도 1cm 이내로 배치
- 에피소드 성공률 > 80%

#### 3.3.2 성능 지표
**학습 중 모니터링**:
- 평균 에피소드 보상
- 서브태스크 완료율
- 에피소드 길이
- 성공률 (전체 조립 완료율)

**예상 학습 곡선**:
1. **초기 (0-500 에폭)**: 랜덤 행동, 낮은 보상
2. **중기 (500-2000 에폭)**: 서브태스크별 학습, 보상 증가
3. **후기 (2000-5000 에폭)**: 전체 시퀀스 최적화, 높은 성공률

#### 3.3.3 최종 목표
- **성공률**: > 80% (전체 조립 완료)
- **정밀도**: < 1cm 위치 오차
- **안정성**: 낮은 속도로 조립 (충돌 최소화)
- **일반화**: 다양한 초기 위치에서 성공

### 3.4 향후 개선 방향

#### 3.4.1 하이퍼파라미터 튜닝
- 학습률 조정 (3e-4 → 1e-4 또는 5e-4)
- 네트워크 크기 증가 ([1024, 512, 256])
- 보상 가중치 최적화

#### 3.4.2 커리큘럼 학습
- 쉬운 기어부터 시작 (gear_1)
- 점진적으로 어려운 기어 추가
- `curriculum_start_gear_idx` 활용

#### 3.4.3 정책 구조 개선
- 서브태스크별 정책 분리 (현재는 full sequence)
- 공유 정책과 기어별 정책 조합
- Hierarchical RL 접근

#### 3.4.4 보상 함수 개선
- Sparse reward → Dense reward
- Shaped reward 함수 최적화
- Success rate 기반 보상

---

## 기술 스택 및 의존성

### 프레임워크
- **Isaac Lab**: 시뮬레이션 환경
- **RL-Games**: 강화학습 알고리즘 구현
- **PyTorch**: 딥러닝 프레임워크
- **NVIDIA Omniverse**: 물리 시뮬레이션

### 주요 라이브러리
- `isaaclab`: Isaac Lab 코어 라이브러리
- `rl_games`: RL 알고리즘
- `wandb`: 실험 추적 (선택적)

### 하드웨어 요구사항
- **GPU**: NVIDIA GPU (PhysX GPU 시뮬레이션)
- **메모리**: 4096 환경 × 환경당 메모리
- **저장공간**: 체크포인트 및 로그 저장용

---

## 파일 구조

```
gearboxAssembly/
├── scripts/
│   ├── train_long_trajectory_assembly.py    # 전체 시퀀스 학습
│   ├── train_all_subtask_policies.py        # 순차 정책 학습
│   └── rl_games/
│       ├── train.py
│       └── play.py
├── source/Galaxea_Lab_External/
│   └── Galaxea_Lab_External/tasks/direct/long_trajectory_assembly/
│       ├── long_trajectory_assembly_env.py      # 환경 구현
│       └── long_trajectory_assembly_env_cfg.py  # 환경 설정
├── logs/rl_games/
│   └── Galaxea-LongTrajectoryAssembly-Direct-v0/
│       └── LongTrajectoryAssembly/
│           ├── nn/                            # 체크포인트
│           └── params/                         # 설정 파일
└── checkpoints/
    └── long_trajectory_assembly/              # 정책별 체크포인트
```

---

## 실행 방법

### 전체 시퀀스 학습
```bash
python scripts/train_long_trajectory_assembly.py \
    --task Galaxea-LongTrajectoryAssembly-Direct-v0 \
    --subtask full \
    --num_envs 4096 \
    --max_iterations 5000 \
    --track  # W&B 추적 활성화
```

### 특정 서브태스크 학습
```bash
python scripts/train_long_trajectory_assembly.py \
    --task Galaxea-LongTrajectoryAssembly-Direct-v0 \
    --subtask approach \
    --num_envs 4096 \
    --max_iterations 2000
```

### 체크포인트에서 재개
```bash
python scripts/train_long_trajectory_assembly.py \
    --task Galaxea-LongTrajectoryAssembly-Direct-v0 \
    --checkpoint logs/rl_games/.../nn/checkpoint.pth \
    --max_iterations 5000
```

---

## 참고 자료

### 주요 설정 파일
- 환경 설정: `long_trajectory_assembly_env_cfg.py`
- 환경 구현: `long_trajectory_assembly_env.py`
- 학습 스크립트: `train_long_trajectory_assembly.py`

### 체크포인트 위치
- 로컬: `logs/rl_games/Galaxea-LongTrajectoryAssembly-Direct-v0/.../nn/`
- Hugging Face: `yjsm1203/gearboxAssembly-checkpoints`

### 로그 파일
- 학습 로그: `checkpoints/long_trajectory_assembly/training_log_*.txt`
- W&B: `wandb/run-*/`
- TensorBoard: `logs/rl_games/.../summaries/`

---

## 결론

현재 프로젝트는 **8-Policy 구조의 Hierarchical RL 접근**을 사용하여 복잡한 기어 조립 작업을 학습하고 있습니다. 환경 기반 규칙 전이를 통해 각 서브태스크를 독립적으로 학습하면서도 전체 시퀀스를 연결할 수 있는 구조를 가지고 있습니다.

**핵심 특징**:
1. 모듈화된 정책 구조 (8개 정책)
2. 환경 기반 자동 전이
3. 서브태스크별 맞춤 보상 함수
4. 높은 병렬성 (4096 환경)

**현재 상태**: 초기 학습 단계 (100 에폭, 평균 보상 229.07)

**다음 단계**: 
- 학습 지속 (목표: 5000 에폭)
- 성능 모니터링 및 하이퍼파라미터 튜닝
- 실패한 서브태스크 재학습

---

*문서 작성일: 2026-01-04*
*마지막 업데이트: 2026-01-04*

