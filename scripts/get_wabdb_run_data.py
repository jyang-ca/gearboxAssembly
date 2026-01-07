import wandb
import pandas as pd
import os

# (1) W&B 로그인 (이미 로그인돼 있으면 생략 가능)
# 터미널에서: wandb login
# 또는
# wandb.login()

api = wandb.Api()

run = api.run("didwlgh111-notion/LongTrajectoryAssembly/runs/ou304lz1")

# (2) 모든 history 가져오기
history = run.history(keys=None, pandas=True)

# (3) 저장 경로 확인
save_path = os.path.abspath("run_full_history.csv")
history.to_csv(save_path, index=False)

print(f"CSV 저장 완료: {save_path}")
