# RoCo Challenge 2026 - Gearbox Assembly 실행 가이드

## 📋 프로젝트 목적

이 프로젝트는 3단계로 구성됩니다:

1. **데이터 수집**: Isaac Sim 환경에서 기어와 부품들의 RGB 이미지 + YOLO 라벨 데이터 수집
2. **Vision Model 훈련**: 수집된 데이터로 객체 검출 모델(YOLO) 훈련
3. **조립 실행**: 훈련된 vision model로 부품 위치를 찾아 rule-based 정책으로 조립 수행

---

## 🚀 주요 실행 커맨드

### 1️⃣ GT + Rule-Based Agent (Ground Truth 위치 사용)

```bash
python scripts/rule_based_agent.py \
  --task Template-Galaxea-Lab-External-Direct-v0 \
  --num_envs 1 \
  --enable_cameras \
  --headless \
  --video \
  --video_length 200
```

**동작 방식:**
- **GT (Ground Truth)**: 시뮬레이션에서 직접 제공하는 정확한 객체 위치(`obj.data.root_state_w`)를 사용
- **Rule-Based Policy**: `GalaxeaRulePolicy` 클래스가 사전 정의된 규칙으로 로봇 동작 생성
  - 기어를 순서대로 집고 → 이동하고 → 조립하는 state machine 방식
- **카메라**: `--enable_cameras`로 활성화
- **비디오 녹화**: `--video` 플래그로 저장

**생성 데이터:**
- 📁 `videos/rule_based_agent/YYYY-MM-DD_HH-MM-SS-rl-video-episode-0.mp4`

---

### 2️⃣ Vision + Rule-Based Agent (Vision 모델 사용)

```bash
python scripts/rule_based_agent.py \
  --task Template-Galaxea-Lab-External-Direct-v0 \
  --num_envs 1 \
  --enable_cameras \
  --use_vision \
  --headless \
  --video \
  --video_length 200
```

**동작 방식:**
- **Vision System**: `VisionPoseEstimator` 클래스가 카메라 이미지를 분석
  - RGB 이미지에서 YOLO로 2D bounding box 검출
  - Depth 이미지와 결합하여 3D 위치 추정
  - 10 step warmup으로 vision history buffer 초기화
- **Rule-Based Policy**: GT 대신 vision에서 추정한 위치를 사용하여 동작 생성
- **차이점**: `--use_vision` 플래그가 `vision_estimator`를 초기화하고 policy에 전달

**생성 데이터:**
- 📁 `videos/rule_based_agent/YYYY-MM-DD_HH-MM-SS-rl-video-episode-0.mp4`

---

### 3️⃣ Data Collection (YOLO 훈련 데이터 수집)

```bash
./run_collect.sh --collect_steps 50
```

또는 직접:

```bash
python scripts/collect_data.py \
  --task Template-Galaxea-Lab-External-Direct-v0 \
  --num_envs 1 \
  --headless \
  --enable_cameras \
  --dataset_dir dataset_yolo \
  --collect_steps 50
```

**동작 방식:**
- **Random Actions**: 로봇이 무작위로 움직이며 다양한 각도에서 부품 관찰
- **카메라 렌더링**: `front_camera` (고정 글로벌 뷰)에서 이미지 캡처
- **3D → 2D Projection**: 
  - 각 객체의 3D bounding box 8개 코너를 카메라 좌표계로 변환
  - Intrinsic matrix로 2D 픽셀 좌표 투영
  - OpenGL → OpenCV 좌표계 변환 (`y, z` 반전)
- **Occlusion Check**: Depth map으로 가려진 객체 필터링
- **YOLO Format**: `class_id x_center y_center width height` (normalized 0-1)

**클래스 매핑:**
- `0`: sun_planetary_gear (4개 모두 동일 클래스)
- `1`: ring_gear
- `2`: planetary_reducer

**생성 데이터:**
```
📁 dataset_yolo/
├── 📁 images/
│   ├── 000000_front.png
│   ├── 000001_front.png
│   └── ...
├── 📁 labels/
│   ├── 000000_front.txt  # YOLO format labels
│   ├── 000001_front.txt
│   └── ...
└── 📁 debug_images/
    ├── 000000_front_debug.png  # Bounding box 시각화
    └── ...
```

**Label 파일 예시 (`000000_front.txt`):**
```
0 0.512345 0.678901 0.123456 0.234567
1 0.345678 0.456789 0.234567 0.345678
2 0.789012 0.890123 0.098765 0.087654
```

---

## 📂 데이터 생성 디렉토리 요약

| 커맨드 | 생성 위치 | 내용 |
|--------|----------|------|
| **GT + Rule-Based** | `videos/rule_based_agent/` | 조립 과정 비디오 (MP4) |
| **Vision + Rule-Based** | `videos/rule_based_agent/` | Vision 기반 조립 비디오 |
| **Data Collection** | `dataset_yolo/images/`<br>`dataset_yolo/labels/`<br>`dataset_yolo/debug_images/` | RGB 이미지 (PNG)<br>YOLO 라벨 (TXT)<br>디버그 이미지 (PNG) |

---

## 🔄 전체 워크플로우

```
1. Data Collection
   ↓
   dataset_yolo/ 생성
   
2. Vision Model Training
   ↓
   train_vision_model.py 실행
   ↓
   gearbox_training/runs/ 에 YOLO 모델 저장
   
3. Deployment
   ↓
   Vision + Rule-Based Agent 실행
   ↓
   videos/ 에 결과 저장
```

---

## 📝 주요 파라미터 설명

### 공통 파라미터

- `--task`: 실행할 환경 이름
- `--num_envs`: 병렬 실행할 환경 개수 (기본값: 1)
- `--enable_cameras`: 카메라 활성화
- `--headless`: GUI 없이 백그라운드 실행
- `--disable_fabric`: Fabric 대신 USD I/O 사용 (디버깅용)

### rule_based_agent.py 전용

- `--video`: 비디오 녹화 활성화
- `--video_length`: 녹화할 스텝 수 (기본값: 200)
- `--use_vision`: Vision 시스템 사용 (기본값: GT 사용)
- `--no_action`: 동작 비활성화 (환경 확인용)

### collect_data.py 전용

- `--dataset_dir`: 데이터셋 저장 디렉토리 (기본값: `dataset_yolo`)
- `--collect_steps`: 수집할 프레임 수 (기본값: 50)

---

## 🔍 참고 문서

- [RoCo Challenge 2026 공식 문서](https://rocochallenge.github.io/RoCo2026/doc.html)
- [DATA_COLLECTION_REPORT.md](DATA_COLLECTION_REPORT.md) - 데이터 수집 구현 상세 내역
- [README.md](README.md) - 프로젝트 설치 및 기본 사용법
