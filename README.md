# ETRI 비전 프로젝트

로컬 Inception v3 기반 CIFAR-10 이미지 분류, Inception-FPN Faster R-CNN 객체 탐지, 검출별 Grad-CAM 시각화 프로젝트

Python 3.12와 `uv`를 사용하는 구성

## 현재 데이터와 체크포인트

- `data/cifar-10-batches-py/`: CIFAR-10 데이터
- `data/VOCdevkit/VOC2012/`: Pascal VOC 2012 데이터와 XML 바운딩 박스 주석
- `checkpoints/inception_v3_cifar10_best.pth`: CIFAR-10 분류 가중치
- `checkpoints/inception_fpn_fasterrcnn.pth`: 객체 탐지 가중치. 저장 메타데이터 기준 epoch 1, VOC `val`, IoU 0.5, mAP 0.06277
- `data/`와 `checkpoints/`: `.gitignore` 대상 경로. Git 추적 제외

CIFAR-10 분류 가중치만으로 객체 탐지 학습은 불가능한 구조. 탐지 학습에는 Pascal VOC처럼 객체별 바운딩 박스 주석이 필요. 탐지 모델 초기화에는 CIFAR-10 체크포인트의 호환 가능한 Inception 특징 가중치만 사용하며, FPN과 Faster R-CNN 탐지 헤드는 VOC 주석으로 학습하는 방식

## 설치

저장소 루트에서 `uv` 의존성 동기화

```bash
uv sync
```

## CIFAR-10 분류 학습

직접 실행 또는 셸 스크립트를 통한 단계별 실행

```bash
uv run python -m training.TLInception --device auto
bash training/train_cifar10.sh --head-epochs 1 --finetune-epochs 5 --device auto
```

기본 데이터 저장 경로 `data/`, 최고 검증 정확도 체크포인트 경로 `checkpoints/inception_v3_cifar10_best.pth`. 분류층 단독 학습 후 전체 모델 미세 조정, 최고 검증 정확도 가중치 저장, 최종 테스트 정확도 출력의 학습 흐름

## 객체 탐지 학습 및 mAP

현재 프로젝트의 VOC 데이터 경로 `data/VOCdevkit/VOC2012/`. 데이터가 없는 환경에서 사용하는 다운로드 스크립트

```bash
bash detection/scripts/download_voc2012.sh
```

VOC 탐지 학습 예시

```bash
uv run python -m detection.train_detector \
  --data data/VOCdevkit/VOC2012 \
  --classes aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor \
  --classification-checkpoint checkpoints/inception_v3_cifar10_best.pth \
  --epochs 10 \
  --checkpoint checkpoints/inception_fpn_fasterrcnn.pth
```

각 epoch 학습 후 VOC `val` 분할을 통한 클래스별 AP와 VOC 방식 mAP 평가. 기본 지표 `mAP@0.50`, IoU 기준 설정 옵션 `--iou-threshold`. 최고 검증 mAP 가중치와 평가 메타데이터 저장. 현재 체크포인트에 기록된 mAP `0.06277`. 데이터 형식과 평가 정의는 [객체 탐지 안내](OBJECT_DETECTION.md) 참고

## 탐지 결과와 Grad-CAM

VOC 검증 이미지의 heatmap 생성 예시

```bash
uv run python -m detection.detect_gradcam \
  --image data/VOCdevkit/VOC2012/JPEGImages/2008_000026.jpg \
  --checkpoint checkpoints/inception_fpn_fasterrcnn.pth \
  --output-dir data/gradcam_results \
  --score-threshold 0.5 \
  --device auto
```

검출별 결과 이미지 저장. 선택된 검출 점수에 대한 heatmap, 예측 바운딩 박스, 클래스명과 신뢰도를 함께 표시. Grad-CAM은 모델 점수 설명이며, heatmap과 정답 위치의 일치 또는 탐지 박스 정확도를 보증하지 않는 결과. 재사용 API와 계산 단계는 [Grad-CAM 안내](gradcam/README.md) 참고

## 소스 파일과 함수

### 실행 및 학습

| 파일 | 역할과 함수 |
|---|---|
| `training/TLInception.py` | CIFAR-10 전이 학습 실행 파일. `choose_device`: CUDA/MPS/CPU 선택. `build_datasets`: 데이터 증강과 train/validation/test 데이터셋 구성. `make_loaders`: 데이터로더 구성. `set_head_only_training`: 분류층 단독 학습 설정. `make_finetune_optimizer`: backbone과 분류층별 학습률을 적용한 optimizer 구성. `run_epoch`: 학습 또는 평가 epoch와 진행률 처리. `save_checkpoint`: 모델 가중치와 학습 메타데이터 저장. `main`: 분류 학습 전체 흐름 실행. |
| `training/train_cifar10.sh` | CIFAR-10 학습 실행 도우미. 루트 디렉터리로 이동해 인자를 `training.TLInception` 모듈에 전달 |
| `detection/train_detector.py` | VOC 객체 탐지 학습과 epoch별 검증 실행 파일. `parse_classes`: 클래스 이름 검증 및 파싱. `collate_detection_batch`: 가변 개수 상자 데이터를 배치로 묶는 함수. `main`: 데이터셋·모델·optimizer 구성, 학습, mAP 평가, 최고 성능 체크포인트 저장. |
| `detection/detect_gradcam.py` | 탐지 모델 추론과 검출별 Grad-CAM 저장 CLI. `main`: 체크포인트와 입력 이미지 로드, 검출 점수별 heatmap 생성, 예측 박스 합성. |
| `detection/scripts/download_voc2012.sh` | Pascal VOC 2012 데이터 다운로드와 `data/VOCdevkit/VOC2012/` 압축 해제 스크립트 |

### 모델

| 파일 | 역할과 함수·클래스 |
|---|---|
| `models/inception.py` | 프로젝트의 Inception v3 구현. `_load_pretrained_state_dict`: 호환 파라미터 로드. `inception_v3`: Inception v3 생성. `Inception3`: 전체 네트워크 구성과 순전파를 담당하는 `__init__`, `forward`. `InceptionA`, `InceptionB`, `InceptionC`, `InceptionD`, `InceptionE`: 각 Inception 블록을 정의하는 `__init__`, `forward`. `InceptionAux`: 보조 분류기 구성과 순전파. `BasicConv2d`: convolution·정규화·활성화 블록 구성과 순전파. |
| `models/detection.py` | Inception backbone과 Faster R-CNN 구성. `InceptionFPNBackbone` 클래스의 `__init__`: 특징 추출기와 FPN 구성. `load_classification_checkpoint`: 호환 Inception 특징 가중치 로드. `_transform_input`: Inception 입력 정규화. `forward`: 다단계 FPN 특징 계산. `build_inception_faster_rcnn`: RPN·ROI pooling·탐지 헤드 구성. `choose_device`: 실행 장치 선택. |
| `models/__init__.py` | `Inception3`, `inception_v3` 공개 API 내보내기 모듈 |

### 데이터와 평가

| 파일 | 역할과 함수·클래스 |
|---|---|
| `detection/dataset.py` | Pascal VOC XML 데이터 로더. `VOCDetectionDataset.__init__`: 이미지·주석·split 확인과 클래스 ID 구성. `__len__`: 이미지 개수 반환. `__getitem__`: 이미지, 상자, 클래스, difficult 플래그를 타깃으로 구성. |
| `detection/metrics.py` | VOC AP/mAP 계산. `_box_iou`: 상자 간 IoU 계산. `_average_precision`: VOC 적분 방식 AP 계산. `calculate_voc_map`: 클래스별 AP와 mAP 집계. `evaluate_detector`: 검증 데이터 추론과 지표 계산 연결. |
| `detection/progress.py` | `report_progress`: 진행률, 처리 이미지 수, 경과 시간과 예상 시간 출력 |
| `detection/__init__.py` | `VOCDetectionDataset`, `calculate_voc_map`, `evaluate_detector`, `report_progress` 공개 API 내보내기 모듈 |

### Grad-CAM

| 파일 | 역할과 함수·클래스 |
|---|---|
| `gradcam/core.py` | 모델 독립적인 Grad-CAM 계산. `GradCAM.__init__`: 대상 레이어 forward hook 등록. `_select_tensor`, `_capture_activation`: 활성화 텐서 선택과 저장. `generate`: 대상 점수 gradient로 heatmap 계산. `close`, `__enter__`, `__exit__`: hook 등록 해제와 컨텍스트 관리. |
| `gradcam/visualization.py` | `overlay_detection_cam`: heatmap 합성, 예측 상자·레이블·점수 표시, 결과 이미지 저장 |
| `gradcam/__init__.py` | `GradCAM`, `overlay_detection_cam` 공개 API 내보내기 모듈 |

### 디렉터리 구성

- `training/`: CIFAR-10 학습 모듈과 학습 셸 스크립트
- `detection/`: VOC 데이터 처리·평가 모듈, 객체 탐지 학습 및 Grad-CAM 실행 모듈
- `detection/scripts/`: 객체 탐지 데이터셋 준비 스크립트
- `models/`: Inception 분류·탐지 모델 구현
- `gradcam/`: 재사용 가능한 Grad-CAM 계산·시각화 API

Python 실행 모듈과 셸 스크립트는 저장소 루트에서 실행. Python 모듈은 `uv run python -m <패키지>.<모듈>` 형식 사용

### 문서와 설정

| 파일 또는 경로 | 역할 |
|---|---|
| `OBJECT_DETECTION.md` | VOC 데이터 구조, 클래스 매핑, 탐지 학습·mAP 평가·Grad-CAM 실행 안내 |
| `gradcam/README.md` | Grad-CAM API와 계산·시각화 흐름 안내 |
| `pyproject.toml` | 프로젝트 메타데이터, Python 버전 범위, `uv` 의존성 선언 |
| `uv.lock` | 의존성 버전 잠금 파일 |
| `requirements.txt` | Python 의존성 보조 목록. 기본 설치 방법은 `uv sync` |
| `.python-version` | 프로젝트 Python 주 버전 지정 |
| `.gitignore` | 가상환경, 캐시, 데이터, 체크포인트 등 Git 제외 대상 지정 |
| `cam.jpg` | Grad-CAM 예시 이미지 자산 |
| `docs/` | 발표 자료와 참고 문서. 모델 실행 코드와 직접 연결되지 않는 자료 모음 |
