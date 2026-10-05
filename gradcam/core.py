"""모델 종류에 종속되지 않는 Grad-CAM 기본 기능."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """스칼라 점수와 모델 레이어 하나를 사용해 Grad-CAM 맵을 생성합니다.

    기존
    ``gradCam.py``의 활성화값·기울기 가중 방식에 기반하며, 모든 ``nn.Module`` 레이어에 적용할 수 있습니다.
    컨텍스트 관리자를 사용하면 임시 순전파 훅이 자동으로 제거됩니다.

    인자:
        target_layer: 설명할 NCHW 활성화값을 출력하는 모듈.
        activation_selector: 레이어 출력에서 활성화 텐서를 선택하는 함수(선택 사항).
            예를 들어 레이어가 FPN 특징 딕셔너리를 반환할 때 ``output["stage2"]``를 선택합니다.


    반환값:
        target_layer에 순전파 훅을 등록한 ``GradCAM`` 인스턴스.
    """

    def __init__(
        self,
        target_layer: nn.Module,
        activation_selector: Callable[[object], torch.Tensor] | None = None,
    ) -> None:
        self.target_layer = target_layer
        self.activation_selector = activation_selector or self._select_tensor
        self.activations: torch.Tensor | None = None
        self._hook = target_layer.register_forward_hook(self._capture_activation)

    @staticmethod
    def _select_tensor(output: object) -> torch.Tensor:
        """텐서 출력을 그대로 반환합니다.

        인자:
            output: 대상 모듈에서 가져온 출력.

        반환값:
            출력 텐서.

        예외:
            TypeError: 출력이 구조화된 값인데 선택 함수가 지정되지 않은 경우.
        """
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "Target layer output is not a tensor; provide activation_selector "
                "for structured outputs"
            )
        return output

    def _capture_activation(self, _module, _inputs, output) -> None:
        """순전파 훅에서 선택한 NCHW 활성화값을 저장합니다.

        인자:
            _module: 훅이 등록된 모듈(콜백에서는 사용하지 않음).
            _inputs: 모듈 입력 튜플(콜백에서는 사용하지 않음).
            output: ``activation_selector``에 전달할 모듈 출력.

        반환값:
            반환값 없음. 활성화값은 ``generate``에서 사용하도록 인스턴스에 저장됩니다.
        """
        activation = self.activation_selector(output)
        if activation.ndim != 4:
            raise ValueError(f"Grad-CAM expects NCHW activations, got {activation.shape}")
        self.activations = activation

    def generate(
        self,
        target_score: torch.Tensor,
        output_size: tuple[int, int],
        *,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        """스칼라 대상 하나에 대한 정규화된 CAM을 계산합니다.

        인자:
            target_score: 설명할 스칼라 점수(예: 탐지 결과 하나의 신뢰도 점수).
                autograd 계산 그래프가 유지되어야 합니다.
            output_size: 결과 맵 크기 ``(높이, 너비)``.
            retain_graph: 같은 모델 순전파 결과에서 다른 CAM을 계산하도록 그래프를 유지할지 여부.
                마지막 대상에서는 False로 설정합니다.

        반환값:
            ``(높이, 너비)`` 모양의 분리된 실수형 텐서. 값 범위는 [0, 1]이며
            CPU에 반환됩니다.

        예외:
            RuntimeError: 순전파 활성화값이 없거나 대상 점수에
                autograd 그래프가 연결되지 않은 경우.
            ValueError: 대상 점수가 스칼라가 아니거나 활성화값이 올바르지 않은 경우.
        """
        if self.activations is None:
            raise RuntimeError("Run the model forward pass before generating Grad-CAM")
        if target_score.numel() != 1:
            raise ValueError("target_score must be a scalar tensor")
        if not target_score.requires_grad:
            raise RuntimeError(
                "The target score has no gradient graph; do not wrap detection in no_grad()"
            )

        gradient = torch.autograd.grad(
            target_score,
            self.activations,
            retain_graph=retain_graph,
            allow_unused=False,
        )[0]
        channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((channel_weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam_min = cam.min()
        cam_max = cam.max()
        span = cam_max - cam_min
        if span > torch.finfo(cam.dtype).eps:
            cam = (cam - cam_min) / span
        else:
            cam = torch.zeros_like(cam)
        return cam.detach().cpu()

    def close(self) -> None:
        """순전파 훅을 제거합니다.

        인자:
            없음.

        반환값:
            없음.
        """
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self) -> "GradCAM":
        """``with`` 구문에서 사용할 현재 객체를 반환합니다.

        인자:
            없음.

        반환값:
            현재 ``GradCAM`` 인스턴스.
        """
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        """``with`` 구문을 벗어날 때 훅을 제거합니다.

        인자:
            _exc_type: 블록에서 예외가 발생한 경우 그 예외 형식.
            _exc: 블록에서 예외가 발생한 경우 그 예외 객체.
            _traceback: 예외가 발생한 경우 그 추적 정보.

        반환값:
            없음.
        """
        self.close()


__all__ = ["GradCAM"]
