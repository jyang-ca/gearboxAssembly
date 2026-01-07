# Approach Subtask 훈련 실패 원인 및 해결 방안

## 문제 진단
1. ✅ 체크포인트 정상 (2.4MB, 236M frames 훈련)
2. ❌ Reward 241로 정체 (학습 실패)
3. ❌ Reward 설계 문제
4. ❌ 체크포인트 체이닝 비활성화

---

## 해결 방안

### 1. Reward Weights 대폭 증가
```python
# env_cfg.py 수정
reward_approach_distance_weight = 2.0  # 0.1 → 2.0 (20배)
reward_approach_height_weight = 1.0    # 0.1 → 1.0 (10배)
reward_approach_orientation_weight = 1.0  # 0.1 → 1.0 (10배)
reward_approach_gripper_open_weight = 0.5  # 0.05 → 0.5 (10배)
reward_approach_complete_bonus = 100.0  # 1.0 → 100.0 (100배!)
reward_time_penalty = 0.0001  # 0.001 → 0.0001 (10배 감소)
```

### 2. Threshold 완화 (Curriculum Learning)
```python
# 초기 훈련 시
approach_horizontal_threshold = 0.10  # 3cm → 10cm (쉽게)
approach_height_threshold = 0.05  # 2cm → 5cm
approach_orientation_dot_threshold = 0.85  # 0.95 → 0.85 (덜 엄격)

# 점진적으로 엄격하게 조정
```

### 3. 더 긴 훈련
```bash
python scripts/train_optimized.py \
  --num_envs 8192 \
  --max_iterations 5000 \  # 1000 → 5000
  --only_subtask approach \
  --track
```

### 4. Imitation Learning (Optional)
- 성공적인 trajectory 몇 개를 수동으로 생성
- Behavior cloning으로 pre-train
- RL fine-tuning

---

## 예상 결과
- Reward: 241 → 1000+ (완료 보너스 포함)
- Success rate: 0% → 80%+
- 훈련 시간: ~3-5시간 (8192 envs, 5000 iter)

---

## 실행 순서
1. `env_cfg.py` reward weights 수정
2. Threshold 완화
3. 재훈련 시작
4. W&B로 모니터링 (reward 증가 확인)
5. Success rate 80% 도달 시 threshold 점진적 강화

