"""학습된 탐지 모델을 실행하고 탐지 결과별 Grad-CAM 히트박스 이미지를 저장합니다."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from gradcam import GradCAM, overlay_detection_cam
from models.detection import build_inception_faster_rcnn, choose_device


def main() -> None:
    """탐지 추론을 수행하고 선택된 결과마다 오버레이 이미지를 저장합니다.

    명령줄 인자:
        --image: 입력 이미지 경로.
        --checkpoint: 클래스 이름이 포함된 탐지 모델 체크포인트.
        --output-dir: 탐지 결과별 시각화 이미지 저장 디렉터리.
        --score-threshold: 시각화할 최소 탐지 신뢰도.
        --max-detections: 설명할 최대 탐지 수.
        --cam-alpha: 히트맵 합성 투명도.
        --device: auto, cpu, mps 또는 cuda.

    반환값:
        반환값 없음. 저장한 시각화 이미지 경로를 출력합니다.
    """
    parser = argparse.ArgumentParser(
        description="Detect objects and visualize each detection with Grad-CAM"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/inception_fpn_fasterrcnn.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gradcam"))
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--cam-alpha", type=float, default=0.38)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    if not args.checkpoint.is_file():
        parser.error(
            f"detector checkpoint not found: {args.checkpoint}. "
            "Train it first with detection.train_detector and a box-annotated dataset."
        )
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0 and 1")
    if args.max_detections < 1:
        parser.error("--max-detections must be positive")

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    class_names = checkpoint.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        parser.error(f"checkpoint has no class_names metadata: {args.checkpoint}")

    model = build_inception_faster_rcnn(class_names).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (KeyError, RuntimeError) as exc:
        parser.error(f"incompatible detector checkpoint: {exc}")
    model.eval()

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    image_tensor = to_tensor(image).to(device)
    target_layer = model.backbone.fpn
    activation_selector = lambda output: output["stage2"]

    with GradCAM(target_layer, activation_selector) as gradcam:
        # autograd를 활성화한 상태로 유지합니다. 선택한 탐지 점수가 CAM의 대상입니다.
        detections = model([image_tensor])[0]
        selected = torch.where(detections["scores"] >= args.score_threshold)[0]
        selected = selected[:args.max_detections]
        if selected.numel() == 0:
            print("No detections passed the score threshold.")
            return

        for position, detection_index in enumerate(selected.tolist()):
            class_id = int(detections["labels"][detection_index].item())
            if not 1 <= class_id <= len(class_names):
                continue
            label = class_names[class_id - 1]
            score = float(detections["scores"][detection_index].detach().cpu())
            cam = gradcam.generate(
                detections["scores"][detection_index],
                output_size=(image.height, image.width),
                retain_graph=position < selected.numel() - 1,
            )
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._") or "object"
            output_path = args.output_dir / (
                f"{args.image.stem}_{position:02d}_{safe_label}_{score:.2f}_gradcam.jpg"
            )
            overlay_detection_cam(
                image,
                cam,
                detections["boxes"][detection_index],
                label,
                score,
                output_path,
                alpha=args.cam_alpha,
            )
            print(f"{label} {score:.3f}: {output_path}")


if __name__ == "__main__":
    main()
