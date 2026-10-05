"""프로젝트의 Inception v3 체크포인트를 사용하는 Faster R-CNN 객체 탐지 모델."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from .inception import inception_v3


class InceptionFPNBackbone(nn.Module):
    """Inception의 세 단계를 Faster R-CNN용 특징 피라미드로 제공합니다."""

    feature_names = ("stage2", "stage3", "stage4", "pool")

    def __init__(self, classification_checkpoint: Path | None = None) -> None:
        super().__init__()
        self.inception = inception_v3(
            weights=None,
            num_classes=1000,
            aux_logits=False,
            transform_input=True,
            init_weights=False,
        )
        if classification_checkpoint is not None:
            self.load_classification_checkpoint(classification_checkpoint)

        # CIFAR-10 분류기(fc)는 영역 제안이나 상자 클래스 예측에 사용하지 않습니다.
        self.inception.fc = nn.Identity()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[288, 768, 2048],
            out_channels=256,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = 256

    def load_classification_checkpoint(self, path: Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Inception classification checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint has no model_state_dict: {path}")

        # Inception 특징 추출기 가중치만 가져옵니다. 원본 분류기는 CIFAR-10 클래스를
        # 분류하므로 의도적으로 제외합니다.
        backbone_state = {
            key: value
            for key, value in state.items()
            if not key.startswith(("fc.", "AuxLogits."))
        }
        incompatible = self.inception.load_state_dict(backbone_state, strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing != {"fc.weight", "fc.bias"} or unexpected:
            raise ValueError(
                "Checkpoint does not match this Inception v3 backbone. "
                f"Missing keys: {sorted(missing)}; unexpected keys: {sorted(unexpected)}"
            )

    @staticmethod
    def _transform_input(images: torch.Tensor) -> torch.Tensor:
        """프로젝트에서 학습한 Inception과 동일한 입력 변환을 적용합니다."""
        images = images.clone()
        images[:, 0] = images[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        images[:, 1] = images[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        images[:, 2] = images[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        return images

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        x = self._transform_input(images) if self.inception.transform_input else images
        body = self.inception

        x = body.Conv2d_1a_3x3(x)
        x = body.Conv2d_2a_3x3(x)
        x = body.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = body.Conv2d_3b_1x1(x)
        x = body.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = body.Mixed_5b(x)
        x = body.Mixed_5c(x)
        stage2 = body.Mixed_5d(x)  # 채널 288개, 스트라이드 8

        x = body.Mixed_6a(stage2)
        x = body.Mixed_6b(x)
        x = body.Mixed_6c(x)
        x = body.Mixed_6d(x)
        stage3 = body.Mixed_6e(x)  # 채널 768개, 스트라이드 16

        x = body.Mixed_7a(stage3)
        x = body.Mixed_7b(x)
        stage4 = body.Mixed_7c(x)  # 채널 2048개, 스트라이드 32

        stages = OrderedDict(
            (("stage2", stage2), ("stage3", stage3), ("stage4", stage4))
        )
        return self.fpn(stages)


def build_inception_faster_rcnn(
    class_names: Sequence[str],
    classification_checkpoint: Path | None = None,
) -> FasterRCNN:
    """객체 탐지기를 구성합니다. 레이블 0은 배경, 1부터 N까지는 객체 클래스입니다."""
    if not class_names:
        raise ValueError("At least one object class is required")
    if len({name.casefold() for name in class_names}) != len(class_names):
        raise ValueError("Object class names must be unique")

    backbone = InceptionFPNBackbone(classification_checkpoint)
    anchors = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )
    roi_pool = MultiScaleRoIAlign(
        featmap_names=list(backbone.feature_names), output_size=7, sampling_ratio=2
    )
    return FasterRCNN(
        backbone,
        num_classes=len(class_names) + 1,
        min_size=299,
        max_size=1333,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        rpn_anchor_generator=anchors,
        box_roi_pool=roi_pool,
        box_score_thresh=0.05,
        box_detections_per_img=100,
    )


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
