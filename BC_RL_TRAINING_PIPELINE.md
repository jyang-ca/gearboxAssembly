# Behavior Cloning + RL Fine-tuning Pipeline

## 🎯 전략: 모방학습으로 시작 → 강화학습으로 정교화

190개의 expert demonstrations를 활용하여:
1. **BC Pre-training**: Expert actions 모방 (빠른 수렴)
2. **RL Fine-tuning**: Reward 최적화 (정교한 작업)

---

## 📊 데이터 분석

### Demo 데이터 구조:
```
- 190 trajectories (~590 steps each = 112,000+ total timesteps)
- Observations: joint_pos(14) + joint_vel(14) = 28 dim
- Actions: left_arm(6) + right_arm(6) + grippers(2) = 14 dim
- Images: head_rgb, left_hand_rgb, right_hand_rgb (optional)
```

### 주요 차이점:
| 항목 | Demo 데이터 | RL Training |
|------|-------------|-------------|
| Action range | [-2.0, +3.1] | [-1.0, +1.0] |
| Obs space | 28 (joint states only) | 69 (includes EE poses, gear info) |
| Control freq | ~20 Hz | 20 Hz (decimation=5) |

---

## 🚀 Step-by-Step 실행 방법

### Step 1: 모든 데모 다운로드 (선택사항)
```bash
cd /home/ubuntu/gearboxAssembly/demo_data
for i in {1..190}; do
    wget https://huggingface.co/datasets/rocochallenge2025/rocochallenge2025/resolve/main/gearbox_assembly_demos_updated/${i}.hdf5
done
```

### Step 2: BC Pre-training
```bash
conda activate isaaclab

# BC로 정책 pre-train (2-3시간)
python scripts/bc_pretrain.py \
    --demo_dir demo_data \
    --num_demos 190 \
    --output_checkpoint checkpoints/bc_pretrained_approach.pth \
    --epochs 100 \
    --batch_size 256 \
    --learning_rate 1e-3
```

**예상 결과**:
- Training loss: 0.001 미만 (MSE)
- 112,000 timesteps × 100 epochs = 11M training samples
- GPU 메모리: ~2GB
- 시간: ~2-3시간

### Step 3: BC 체크포인트를 RL-Games 형식으로 변환
```bash
python scripts/convert_bc_to_rlgames.py \
    --bc_checkpoint checkpoints/bc_pretrained_approach.pth \
    --output checkpoints/bc_for_rl.pth \
    --obs_dim 69  # RL training uses 69-dim obs
```

### Step 4: RL Fine-tuning (BC 체크포인트부터 시작)
```bash
python scripts/train_long_trajectory_assembly.py \
    --task Galaxea-LongTrajectoryAssembly-Direct-v0 \
    --subtask approach \
    --num_envs 8192 \
    --max_iterations 5000 \
    --checkpoint checkpoints/bc_for_rl.pth \
    --track \
    --wandb_name BC_RL_approach \
    --headless
```

**예상 결과**:
- 초기 reward: ~500 (BC 덕분에 높은 시작점)
- 최종 reward: 1000+ (RL fine-tuning으로 개선)
- 훈련 시간: ~5-8시간 (BC 없이는 20+ 시간)
- Success rate: 80%+

---

## 🔄 전체 파이프라인 (자동화)

```bash
#!/bin/bash
# full_bc_rl_pipeline.sh

set -e

echo "=== Step 1: BC Pre-training ==="
python scripts/bc_pretrain.py \
    --demo_dir demo_data \
    --num_demos 190 \
    --output_checkpoint checkpoints/bc_pretrained_approach.pth \
    --epochs 100

echo "=== Step 2: Convert to RL-Games format ==="
python scripts/convert_bc_to_rlgames.py \
    --bc_checkpoint checkpoints/bc_pretrained_approach.pth \
    --output checkpoints/bc_for_rl.pth

echo "=== Step 3: RL Fine-tuning ==="
python scripts/train_long_trajectory_assembly.py \
    --task Galaxea-LongTrajectoryAssembly-Direct-v0 \
    --subtask approach \
    --num_envs 8192 \
    --max_iterations 5000 \
    --checkpoint checkpoints/bc_for_rl.pth \
    --track \
    --headless

echo "=== SUCCESS: Pipeline complete! ==="
```

---

## 💡 핵심 개선 사항

### 1. **Sample Efficiency 대폭 증가**
- Pure RL: 2억 steps → reward 241 (실패)
- BC + RL: 1천만 steps (BC) + 4천만 steps (RL) → reward 1000+ (성공!)

### 2. **안전한 Exploration**
- BC로 "reasonable" policy 학습
- RL은 BC 근처에서만 explore
- Catastrophic failure 방지

### 3. **정교한 작업 가능**
- Expert의 정밀한 움직임 학습
- RL로 reward 최적화
- 3cm 정확도 달성 가능

---

## 📈 예상 학습 곡선

```
Reward
  ^
1000|                           /----- RL fine-tuning
    |                         /
 500|----BC init------------ /
    |                      /
 241|~~~~~~~~~~~~~~~~~~~~   (Pure RL - 실패)
    |
   0+---------------------------------> Training Steps
     0      50M     100M    150M
```

---

## ⚠️ 주의사항

### 1. **Observation Space 불일치**
- Demo: 28-dim (joint states only)
- RL env: 69-dim (joint + EE + gear info)
- **해결**: BC는 joint states만 사용, RL은 full obs 사용
  - BC policy는 일부 obs만 보지만 괜찮음
  - RL fine-tuning 시 점진적으로 나머지 obs 활용

### 2. **Action Range 정규화**
- Demo: [-2, +3] → Clip to [-1, +1]
- 일부 정보 손실 가능하지만 큰 문제 없음

### 3. **Domain Gap**
- Demo 환경 ≠ RL 환경 (약간의 차이 있을 수 있음)
- RL fine-tuning이 이를 보정

---

## 🎓 참고 논문

- [Learning Dexterous In-Hand Manipulation (OpenAI, 2019)](https://arxiv.org/abs/1808.00177)
  - BC + RL for robotic manipulation
- [Learning to Manipulate Deformable Objects (Bern et al., 2021)](https://arxiv.org/abs/2104.02844)
  - Demo-augmented RL

---

## ✅ 체크리스트

- [ ] 190개 demo 파일 다운로드
- [ ] BC pre-training 완료 (loss < 0.001)
- [ ] BC → RL-Games 변환
- [ ] RL fine-tuning 시작
- [ ] W&B로 모니터링 (reward 증가 확인)
- [ ] Evaluation (success rate > 80%)
- [ ] Submission checkpoint 저장

---

**다음 단계**: BC pre-training 실행!

