"""히트맵 및 바운딩 박스 시각화 도우미."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


def overlay_detection_cam(
    image: Image.Image,
    cam: torch.Tensor | np.ndarray,
    box: torch.Tensor | np.ndarray,
    label: str,
    score: float,
    output_path: str | Path,
    *,
    alpha: float = 0.38,
) -> Path:
    """탐지 결과의 히트맵을 합성하고 모델이 예측한 사각형을 그립니다.

    JET 히트맵 합성은 기존
    ``gradCAM2.py``의 시각화 방식을 따릅니다. 사각형은 CAM 임계값에서 근사 추출하지 않고
    객체 탐지 모델의 출력에서 가져옵니다.

    Args:
        image: 원본 이미지(RGB 또는 RGB로 변환 가능한 이미지).
        cam: 정규화된 2차원 CAM(텐서 또는 NumPy 배열).
        box: 원본 이미지 픽셀 기준 예측 상자 ``[x1, y1, x2, y2]``.
        label: 탐지된 객체 클래스 이름.
        score: 사각형 옆에 표시할 탐지 신뢰도.
        output_path: 결과 이미지 경로. 상위 디렉터리가 없으면 생성합니다.
        alpha: 히트맵 투명도(0은 표시 안 함, 1은 완전히 불투명).

    Returns:
        결과 경로를 나타내는 ``Path``.

    Raises:
        ValueError: alpha가 허용 범위를 벗어나거나 box 값이 네 개가 아닌 경우.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    rgb = np.asarray(image.convert("RGB"))
    cam_array = cam.detach().cpu().numpy() if isinstance(cam, torch.Tensor) else np.asarray(cam)
    cam_array = np.nan_to_num(cam_array, nan=0.0, posinf=1.0, neginf=0.0)
    cam_array = np.clip(cam_array, 0.0, 1.0)
    if cam_array.shape != rgb.shape[:2]:
        cam_array = cv2.resize(
            cam_array.astype(np.float32),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    heatmap_bgr = cv2.applyColorMap(np.uint8(cam_array * 255), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    canvas = Image.fromarray(blended)
    draw = ImageDraw.Draw(canvas)

    box_values = box.detach().cpu().tolist() if isinstance(box, torch.Tensor) else list(box)
    if len(box_values) != 4:
        raise ValueError("box must contain [x1, y1, x2, y2]")
    x1, y1, x2, y2 = box_values
    width, height = canvas.size
    x1 = int(np.clip(round(x1), 0, width - 1))
    y1 = int(np.clip(round(y1), 0, height - 1))
    x2 = int(np.clip(round(x2), 0, width - 1))
    y2 = int(np.clip(round(y2), 0, height - 1))
    draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=max(2, width // 300))
    draw.text((x1, max(0, y1 - 16)), f"{label} {score:.2f}", fill=(0, 255, 0))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


__all__ = ["overlay_detection_cam"]
