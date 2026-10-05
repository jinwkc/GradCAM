# Inception 기반 객체 탐지

## 데이터 요구사항

현재 `checkpoints/inception_v3_cifar10_best.pth`는 CIFAR-10 **이미지 분류** 체크포인트. CIFAR-10에는 객체별 바운딩 박스가 없어 이 체크포인트만으로 객체 탐지기 학습 불가. Pascal VOC 형식의 이미지와 XML 박스 주석 준비 필요.

```text
my_detection_data/
├── JPEGImages/
│   ├── image_001.jpg
│   └── image_002.jpg
└── Annotations/
    ├── image_001.xml
    └── image_002.xml
```

각 XML은 이미지 파일과 같은 이름 사용. `<object><name>class-name</name><bndbox>...` 필드에 `xmin`, `ymin`, `xmax`, `ymax` 포함 필요. 클래스 순서는 `--classes`에 쉼표로 지정하고, XML 클래스명과 대소문자 무시 기준으로 일치 필요. 클래스 ID 0은 배경으로 예약.

PASCAL VOC 2012 학습·검증 데이터는 프로젝트 루트에서 아래 스크립트로 다운로드. 압축 파일은 약 2GB이며, 중단 후 스크립트 재실행으로 이어받기 가능.

```bash
bash detection/scripts/download_voc2012.sh
```

압축 해제 후 데이터 경로는 `data/VOCdevkit/VOC2012`. 이 스크립트는 기존 CIFAR-10 데이터를 유지하고 VOC 데이터가 이미 있으면 재압축 해제 생략. 내려받은 압축 파일은 `data/VOCtrainval_11-May-2012.tar`에 보존해 재사용 가능.

## 학습

프로젝트 루트에서 실행.

```bash
uv run python -m detection.train_detector \
  --data data/VOCdevkit/VOC2012 \
  --classes aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor \
  --classification-checkpoint checkpoints/inception_v3_cifar10_best.pth \
  --epochs 10 \
  --log-interval 50 \
  --checkpoint checkpoints/inception_fpn_fasterrcnn.pth
```

기본 학습 분할은 `ImageSets/Main/train.txt`(5,717장), 매 epoch 검증 분할은 `ImageSets/Main/val.txt`(5,823장). 두 이미지 ID 목록의 중복은 불가하며, 데이터셋은 각 분할 파일에 적힌 이미지만 읽음. `--train-split`과 `--validation-split`으로 `ImageSets/Main/` 아래의 다른 분할 이름 지정 가능.

터미널 실행 시 학습 중 매 배치의 현재 손실, 누적 평균 손실, 손실 항목별 값, 진행률, 경과 시간, 예상 남은 시간 표시. 검증 중에도 처리 이미지 수, 진행률, 예상 남은 시간 표시, 완료 후 mAP 출력. 터미널 출력을 파일이나 파이프로 보낼 때는 기본 50배치마다 기록하며, `--log-interval 1` 지정 시 매 배치 기록.

검증 단계에서 클래스별 AP와 VOC 방식 mAP 계산. 기본 IoU 임계값은 0.5이며 `--iou-threshold`로 변경 가능. mAP은 검증 분할에서 일반 정답 상자가 하나 이상인 클래스 AP의 평균. VOC `difficult` 객체는 AP 정답 수에서 제외하며, 해당 객체에만 매칭된 예측은 점수 계산에서 무시. `detection.metrics.calculate_voc_map`은 예측·정답 목록으로 지표 계산, `evaluate_detector`는 모델 추론과 지표 계산 수행. 가장 높은 validation mAP를 기록한 에폭의 모델을 체크포인트 경로에 저장하고 클래스별 AP도 함께 기록.

학습기는 Inception 분류층을 제외하고 CIFAR-10 체크포인트에서 호환되는 Inception 특징 가중치만 로드. FPN, RPN, 객체 분류·박스 회귀층은 새로 초기화해 VOC 바운딩 박스 주석으로 학습. Pascal VOC detection의 AP 기준은 IoU 0.5. 상세 정의는 [VOC 2012 개발 키트](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html) 참고.

## 추론 및 Grad-CAM

학습된 detector 체크포인트로 실행.

```bash
uv run python -m detection.detect_gradcam \
  --image /path/to/image.jpg \
  --checkpoint checkpoints/inception_fpn_fasterrcnn.pth \
  --output-dir outputs/gradcam \
  --score-threshold 0.5
```

검출별로 별도의 결과 이미지 생성. 각 이미지에 선택한 검출 점수의 Grad-CAM 열지도와 해당 검출의 바운딩 박스 표시. 히트맵은 모델 설명 시각화이며, 검출 박스 정확도나 객체 위치 정답성의 보증 자료가 아님.
