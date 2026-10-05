# Inception 기반 객체 탐지

## 데이터 요구사항

현재 `checkpoints/inception_v3_cifar10_best.pth`는 CIFAR-10 **이미지 분류** 체크포인트입니다. CIFAR-10에는 객체별 바운딩 박스가 없으므로 이 체크포인트만으로 객체 탐지기를 학습할 수 없습니다. 아래 Pascal VOC 형식의 이미지와 XML 박스 주석을 준비해야 합니다.

```text
my_detection_data/
├── JPEGImages/
│   ├── image_001.jpg
│   └── image_002.jpg
└── Annotations/
    ├── image_001.xml
    └── image_002.xml
```

각 XML은 이미지 파일과 같은 이름을 사용하고, `<object><name>class-name</name><bndbox>...` 필드에 `xmin`, `ymin`, `xmax`, `ymax`를 포함해야 합니다. 클래스 순서는 `--classes`에 쉼표로 지정하며, XML의 클래스명과 대소문자 무시 기준으로 일치해야 합니다. 모델은 클래스 ID 0을 배경으로 예약합니다.

PASCAL VOC 2012 학습·검증 데이터는 프로젝트 루트에서 아래 스크립트로 받을 수 있습니다. 약 2GB의 압축 파일을 다운로드하며, 중단된 다운로드는 다시 실행해 이어받을 수 있습니다.

```bash
bash detection/scripts/download_voc2012.sh
```

압축 해제 후 데이터 경로는 `data/VOCdevkit/VOC2012`입니다. 이 스크립트는 기존 CIFAR-10 데이터를 유지하고, VOC 데이터가 이미 있으면 다시 풀지 않습니다. 내려받은 압축 파일은 `data/VOCtrainval_11-May-2012.tar`에 남겨 재사용할 수 있습니다.

## 학습

프로젝트 루트에서 실행합니다.

```bash
uv run python -m detection.train_detector \
  --data data/VOCdevkit/VOC2012 \
  --classes aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor \
  --classification-checkpoint checkpoints/inception_v3_cifar10_best.pth \
  --epochs 10 \
  --log-interval 50 \
  --checkpoint checkpoints/inception_fpn_fasterrcnn.pth
```

기본적으로 `ImageSets/Main/train.txt`(5,717장)로 학습하고 `ImageSets/Main/val.txt`(5,823장)로 매 epoch 검증합니다. 두 이미지 ID 목록은 중복될 수 없으며, 데이터셋은 해당 분할 파일의 이미지만 읽습니다. `--train-split`과 `--validation-split`으로 `ImageSets/Main/` 아래의 다른 분할 이름도 지정할 수 있습니다.

터미널에서 실행하면 학습 중 매 배치마다 현재 손실, 누적 평균 손실, 손실 항목별 값, 진행률, 경과 시간과 예상 남은 시간을 확인할 수 있습니다. 검증 중에도 처리한 이미지 수와 진행률 및 예상 남은 시간이 표시되고, 검증이 모두 끝난 뒤 mAP가 출력됩니다. 터미널 출력을 파일이나 파이프로 보낼 때는 기본 50배치마다 출력하며, `--log-interval 1`을 지정하면 매 배치 기록합니다.

검증은 클래스별 AP와 VOC 방식의 mAP를 계산합니다. 기본 IoU 임계값은 0.5이며 `--iou-threshold`로 바꿀 수 있습니다. mAP은 검증 분할에 일반 정답 상자가 하나 이상 있는 클래스의 AP 평균입니다. VOC `difficult` 객체는 AP 정답 수에서 제외하고, 해당 객체에만 매칭된 예측은 점수 계산에서 무시합니다. `detection.metrics.calculate_voc_map`은 예측과 정답 목록으로 지표를 계산하고, `evaluate_detector`는 모델 추론과 지표 계산을 함께 수행합니다. 가장 높은 validation mAP를 얻은 에폭의 모델을 체크포인트 경로에 저장하며, 체크포인트에 클래스별 AP도 기록합니다.

학습기는 Inception의 분류층을 버리고 CIFAR-10 체크포인트에서 호환되는 Inception 특징 가중치만 가져옵니다. FPN, RPN, 객체 분류·박스 회귀층은 새로 초기화해 VOC 바운딩 박스 주석으로 학습합니다. Pascal VOC detection의 AP는 IoU 0.5를 기준으로 하며, 상세 정의는 [VOC 2012 개발 키트](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html)를 참고하세요.

## 추론 및 Grad-CAM

학습된 detector 체크포인트로 실행합니다.

```bash
uv run python -m detection.detect_gradcam \
  --image /path/to/image.jpg \
  --checkpoint checkpoints/inception_fpn_fasterrcnn.pth \
  --output-dir outputs/gradcam \
  --score-threshold 0.5
```

검출별로 별도의 결과 이미지가 생성됩니다. 각 이미지에는 선택된 검출 점수에 대한 Grad-CAM 열지도와 해당 검출기의 바운딩 박스가 함께 그려집니다. 히트맵은 모델 설명 시각화이며, 검출 박스의 정확도나 객체 위치의 정답성을 보증하지 않습니다.
