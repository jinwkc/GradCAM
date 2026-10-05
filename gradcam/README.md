# Grad-CAM 모듈

`gradcam/`은 객체 탐지 모델의 검출 점수를 기준으로 Grad-CAM을 계산하고, 열지도와 모델 예측 바운딩 박스를 원본 이미지에 그리는 패키지.

## 현재 프로젝트 코드와 이전 파일

현재 작업 디렉터리에는 `gradCam.py`, `gradCAM2.py` 원본 파일이 없음. 아래 내용은 예전 파일을 줄 단위로 옮긴 결과가 아니라, 프로젝트에서 확인 가능한 동작 원리와 시각화 방식을 현재 API 기준으로 정리한 내용.

| 현재 파일 | 역할 | 이전 코드와의 관계 |
|---|---|---|
| `core.py` | activation과 gradient로 Grad-CAM 맵 계산 | Grad-CAM의 채널별 평균 기울기 가중 원리 채택. 이전 파일의 직접 재사용 여부는 원본 부재로 검증 불가. |
| `visualization.py` | JET 색상 열지도를 원본 이미지와 합성하고 상자·레이블·점수를 그림 | CAM 표시와 이미지 합성 기능. 상자는 CAM contour가 아니라 탐지 모델의 예측값 사용. |
| `__init__.py` | 패키지 공개 API 제공 | `from gradcam import ...` 형식의 가져오기 경로 제공. |
| `../detection/detect_gradcam.py` | 탐지 모델을 실행하고 각 검출의 결과 이미지 저장 | 패키지 API 연결 실행 모듈. |

Guided Backpropagation, CAM contour 기반 사각형 추정, 예전 특정 모델 계층 번호 의존 처리는 현재 패키지에 미포함.

## 계산 흐름

객체 탐지 실행은 `model.eval()`을 사용. Gradient 계산을 위해 모델 순전파에는 `torch.no_grad()` 또는 `torch.inference_mode()` 미적용.

1. `GradCAM`의 `model.backbone.fpn` forward hook 등록.
2. 실행 스크립트가 FPN 특징 딕셔너리에서 `stage2` 선택. Inception `Mixed_5d`에서 나온 고해상도 공간 특징.
3. Faster R-CNN 출력에서 임계값을 넘은 검출 점수 하나를 CAM 목표로 선택.
4. 목표 점수와 선택 활성값의 gradient로 채널별 공간 평균 가중치 계산.
5. 채널 가중 특징 합산과 ReLU 적용으로 CAM 생성. 원본 이미지 크기로 확대하고 `[0, 1]` 범위로 정규화.
6. 시각화 함수가 열지도를 이미지에 합성하고 검출의 예측 상자·클래스·점수를 표시해 파일로 저장.

```text
채널 가중치[k] = 평균(목표 점수의 활성값[k]에 대한 기울기)
CAM = ReLU(채널 축 합계(채널 가중치 × 활성값))
```

검출 점수별 결과 이미지 생성. 박스는 탐지기 예측값, CAM은 해당 점수에 기여한 특징의 시각화. CAM은 정답 상자나 localization 정확도 검증 자료가 아님.

## 공개 API

```python
from gradcam import GradCAM, overlay_detection_cam

target_layer = detector.backbone.fpn
select_stage2 = lambda output: output["stage2"]

with GradCAM(target_layer, select_stage2) as cam_builder:
    detections = detector([image_tensor])[0]  # gradient 계산을 위해 no_grad 미사용
    index = int(detections["scores"].argmax())
    cam = cam_builder.generate(
        detections["scores"][index],
        output_size=(image.height, image.width),
    )
```

| 함수 | 주요 인자 | 반환값 |
|---|---|---|
| `GradCAM(target_layer, activation_selector=None)` | `target_layer`: 활성값을 얻을 `nn.Module`. `activation_selector`: 레이어 출력에서 NCHW 활성 텐서를 선택하는 함수. | forward hook이 등록된 `GradCAM` 객체. `with` 블록 종료 시 hook 제거. |
| `GradCAM.generate(target_score, output_size, retain_graph=False)` | `target_score`: 설명할 스칼라 점수. `output_size`: `(height, width)`. `retain_graph`: 같은 forward의 다른 점수로 CAM을 더 계산할 때 그래프 유지 여부. | CPU의 `[0, 1]` 범위 2차원 CAM 텐서. |
| `GradCAM.close()` | 인자 없음. | `None`; hook 제거. |
| `overlay_detection_cam(image, cam, box, label, score, output_path, alpha=0.38)` | RGB 이미지, CAM, 이미지 좌표의 `[x1, y1, x2, y2]`, 클래스 이름, 점수, 저장 경로, 합성 투명도. | 저장한 결과의 `Path`. |

`activation_selector`는 FPN처럼 모듈 출력이 딕셔너리일 때 설명 대상 텐서 선택. `generate`의 대상 점수는 autograd 그래프 연결 필요. 한 순전파에서 여러 점수의 CAM 계산 시 마지막 호출 전까지 `retain_graph=True` 사용.

## 실행

바운딩 박스 주석으로 학습한 탐지 체크포인트 필요. CIFAR-10 체크포인트는 Inception 특징 추출기 초기화용이며 탐지 계층 학습 체크포인트가 아님. 현재 프로젝트에는 PASCAL VOC 데이터와 객체 탐지 체크포인트가 없음.

```bash
uv run python -m detection.detect_gradcam \
  --image /path/to/image.jpg \
  --checkpoint checkpoints/inception_fpn_fasterrcnn.pth \
  --output-dir outputs/gradcam \
  --score-threshold 0.5
```

추가 옵션 확인 명령:

```bash
uv run python -m detection.detect_gradcam --help
```

탐지 데이터 형식과 detector 학습 과정은 [OBJECT_DETECTION.md](../OBJECT_DETECTION.md) 참고.
