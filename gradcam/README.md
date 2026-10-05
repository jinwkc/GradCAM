# Grad-CAM 모듈

`gradcam/`은 객체 탐지 모델의 한 검출 점수를 기준으로 Grad-CAM을 계산하고, 열지도와 모델 예측 바운딩 박스를 원본 이미지에 그리는 패키지입니다.

## 현재 프로젝트 코드와 이전 파일

현재 작업 디렉터리에는 `gradCam.py`, `gradCAM2.py` 원본 파일이 없습니다. 따라서 아래 내용은 예전 파일의 코드를 줄 단위로 옮겼다는 뜻이 아니라, 이 프로젝트에서 확인 가능한 동작 원리와 시각화 방식을 현재 API에 맞춰 정리한 것입니다.

| 현재 파일 | 역할 | 이전 코드와의 관계 |
|---|---|---|
| `core.py` | activation과 gradient로 Grad-CAM 맵 계산 | Grad-CAM의 채널별 평균 기울기 가중 원리를 사용합니다. 이전 파일을 직접 가져와 재사용했는지는 현재 원본 부재로 검증할 수 없습니다. |
| `visualization.py` | JET 색상 열지도를 원본 이미지와 합성하고 상자·레이블·점수를 그림 | CAM 표시와 이미지 합성이라는 시각화 목적을 제공합니다. 상자는 CAM contour에서 찾지 않고 탐지 모델의 예측값을 사용합니다. |
| `__init__.py` | 패키지 공개 API 제공 | `from gradcam import ...` 형식으로 가져오게 합니다. |
| `../detection/detect_gradcam.py` | 탐지 모델을 실행하고 각 검출의 결과 이미지 저장 | 이 패키지 API를 연결하는 실행 모듈입니다. |

Guided Backpropagation, CAM contour에서 사각형 추정, 예전 특정 모델 계층 번호에 의존하는 처리는 현재 패키지에 포함되지 않습니다.

## 계산 흐름

객체 탐지 실행은 `model.eval()`을 사용하지만, gradient가 필요하므로 모델 순전파를 `torch.no_grad()` 또는 `torch.inference_mode()`로 감싸지 않습니다.

1. `GradCAM`이 `model.backbone.fpn`에 forward hook을 등록합니다.
2. FPN이 반환한 특징 딕셔너리에서 현재 실행 스크립트는 `stage2`를 선택합니다. Inception `Mixed_5d`에서 나온 공간 해상도가 높은 특징입니다.
3. Faster R-CNN 출력 중 임계값을 넘은 검출 점수 하나를 CAM의 목표로 선택합니다.
4. 목표 점수와 선택한 활성값의 gradient로 각 채널의 공간 평균 가중치를 계산합니다.
5. 채널 가중 특징들을 합산하고 ReLU를 적용해 CAM을 만든 다음 원본 이미지 크기로 확대하고 `[0, 1]`로 정규화합니다.
6. 시각화 함수가 열지도를 이미지에 합성하고, 해당 검출의 예측 상자·클래스·점수를 그려 파일로 저장합니다.

```text
채널 가중치[k] = 평균(목표 점수의 활성값[k]에 대한 기울기)
CAM = ReLU(채널 축 합계(채널 가중치 × 활성값))
```

각 검출 점수에 대해 별도 결과 이미지를 만듭니다. 박스는 탐지기 예측값이고, CAM은 그 점수에 기여한 특징의 시각화입니다. CAM은 정답 상자나 localization 정확도의 검증 자료가 아닙니다.

## 공개 API

```python
from gradcam import GradCAM, overlay_detection_cam

target_layer = detector.backbone.fpn
select_stage2 = lambda output: output["stage2"]

with GradCAM(target_layer, select_stage2) as cam_builder:
    detections = detector([image_tensor])[0]  # gradient가 필요하므로 no_grad를 사용하지 않습니다.
    index = int(detections["scores"].argmax())
    cam = cam_builder.generate(
        detections["scores"][index],
        output_size=(image.height, image.width),
    )
```

| 함수 | 주요 인자 | 반환값 |
|---|---|---|
| `GradCAM(target_layer, activation_selector=None)` | `target_layer`: 활성값을 얻을 `nn.Module`. `activation_selector`: 레이어 출력에서 NCHW 활성 텐서를 선택하는 함수. | forward hook이 등록된 `GradCAM` 객체. `with` 블록을 벗어나면 hook을 제거합니다. |
| `GradCAM.generate(target_score, output_size, retain_graph=False)` | `target_score`: 설명할 스칼라 점수. `output_size`: `(height, width)`. `retain_graph`: 같은 forward의 다른 점수로 CAM을 더 계산할 때 그래프 유지 여부. | CPU에 있는 `[0, 1]` 범위의 2차원 CAM 텐서. |
| `GradCAM.close()` | 인자 없음. | `None`; hook을 제거합니다. |
| `overlay_detection_cam(image, cam, box, label, score, output_path, alpha=0.38)` | RGB 이미지, CAM, 이미지 좌표의 `[x1, y1, x2, y2]`, 클래스 이름, 점수, 저장 경로, 합성 투명도. | 저장한 결과의 `Path`. |

`activation_selector`는 FPN처럼 모듈 출력이 딕셔너리인 경우 설명 대상 텐서를 선택합니다. `generate` 대상 점수는 autograd 그래프에 연결되어 있어야 합니다. 한 순전파에서 여러 점수의 CAM을 구할 때는 마지막 호출 전까지 `retain_graph=True`를 사용합니다.

## 실행

바운딩 박스 주석으로 학습한 탐지 체크포인트가 있어야 합니다. CIFAR-10 체크포인트는 Inception 특징 추출기만 초기화하며, 탐지 계층을 학습한 체크포인트가 아닙니다. 현재 PASCAL VOC 데이터와 객체 탐지 체크포인트는 프로젝트에 없습니다.

```bash
uv run python -m detection.detect_gradcam \
  --image /path/to/image.jpg \
  --checkpoint checkpoints/inception_fpn_fasterrcnn.pth \
  --output-dir outputs/gradcam \
  --score-threshold 0.5
```

더 많은 옵션은 다음으로 확인합니다.

```bash
uv run python -m detection.detect_gradcam --help
```

탐지 데이터 형식과 detector 학습 과정은 [OBJECT_DETECTION.md](../OBJECT_DETECTION.md)를 참고하세요.
