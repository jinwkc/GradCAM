"""PASCAL VOC 객체 탐지 평가 지표."""

from __future__ import annotations

import time
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .progress import report_progress


def _box_iou(box: Tensor, boxes: Tensor) -> Tensor:
    """하나의 상자와 여러 상자 사이의 IoU를 CPU 텐서로 계산합니다."""
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection_size = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    box_area = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    boxes_area = (
        (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    )
    union = box_area + boxes_area - intersection
    return intersection / union.clamp(min=torch.finfo(torch.float32).eps)


def _average_precision(recall: list[float], precision: list[float]) -> float:
    """VOC 2010+ 방식의 precision envelope 적분 AP를 계산합니다."""
    recall = [0.0, *recall, 1.0]
    precision = [0.0, *precision, 0.0]
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    return sum(
        (recall[index + 1] - recall[index]) * precision[index + 1]
        for index in range(len(recall) - 1)
        if recall[index + 1] != recall[index]
    )


def calculate_voc_map(
    predictions: Sequence[dict[str, Tensor]],
    targets: Sequence[dict[str, Tensor]],
    class_names: Sequence[str],
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """VOC AP/mAP를 계산합니다.

    Args:
        predictions: 이미지별 ``boxes``, ``labels``, ``scores`` 결과.
        targets: 예측과 같은 순서의 정답 ``boxes``, ``labels``, 선택적 ``difficult``.
        class_names: 배경을 제외하고 label 1부터 매칭되는 클래스 이름.
        iou_threshold: 정답과 검출 상자를 매칭하는 최소 IoU.

    Returns:
        ``map``, 클래스별 ``per_class_ap`` 및 클래스별 일반 정답 수 딕셔너리.
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same image count")
    if not class_names:
        raise ValueError("class_names cannot be empty")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in the interval (0, 1]")

    ground_truth: dict[int, list[dict[str, Tensor]]] = {
        class_id: [] for class_id in range(1, len(class_names) + 1)
    }
    detections: dict[int, list[tuple[float, int, Tensor]]] = {
        class_id: [] for class_id in range(1, len(class_names) + 1)
    }

    for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
        target_boxes = target["boxes"].detach().cpu().to(torch.float32)
        target_labels = target["labels"].detach().cpu().to(torch.int64)
        difficult = target.get("difficult", torch.zeros_like(target_labels))
        difficult = difficult.detach().cpu().to(torch.bool)
        prediction_boxes = prediction["boxes"].detach().cpu().to(torch.float32)
        prediction_labels = prediction["labels"].detach().cpu().to(torch.int64)
        prediction_scores = prediction["scores"].detach().cpu().to(torch.float32)

        if not (len(target_boxes) == len(target_labels) == len(difficult)):
            raise ValueError(f"Mismatched target fields for validation image {image_index}")
        if not (len(prediction_boxes) == len(prediction_labels) == len(prediction_scores)):
            raise ValueError(f"Mismatched prediction fields for validation image {image_index}")

        for class_id in ground_truth:
            class_mask = target_labels == class_id
            ground_truth[class_id].append(
                {
                    "boxes": target_boxes[class_mask],
                    "difficult": difficult[class_mask],
                    "matched": torch.zeros(int(class_mask.sum()), dtype=torch.bool),
                }
            )

        for box, class_id, score in zip(
            prediction_boxes, prediction_labels, prediction_scores
        ):
            numeric_class_id = int(class_id)
            if numeric_class_id in detections:
                detections[numeric_class_id].append(
                    (float(score), image_index, box)
                )

    per_class_ap: dict[str, float] = {}
    num_ground_truth: dict[str, int] = {}
    scored_aps: list[float] = []
    for class_id, class_name in enumerate(class_names, start=1):
        records = ground_truth[class_id]
        positive_count = sum(int((~record["difficult"]).sum()) for record in records)
        num_ground_truth[class_name] = positive_count
        ordered_detections = sorted(
            detections[class_id], key=lambda detection: detection[0], reverse=True
        )
        true_positives: list[float] = []
        false_positives: list[float] = []

        for _, image_index, predicted_box in ordered_detections:
            record = records[image_index]
            overlaps = _box_iou(predicted_box, record["boxes"])
            if not len(overlaps):
                true_positives.append(0.0)
                false_positives.append(1.0)
                continue

            best_index = int(overlaps.argmax())
            if float(overlaps[best_index]) < iou_threshold:
                true_positives.append(0.0)
                false_positives.append(1.0)
            elif bool(record["difficult"][best_index]):
                # VOC 평가 규칙은 difficult 정답과 일치하는 검출을 점수 집계에서 제외합니다.
                continue
            elif bool(record["matched"][best_index]):
                true_positives.append(0.0)
                false_positives.append(1.0)
            else:
                record["matched"][best_index] = True
                true_positives.append(1.0)
                false_positives.append(0.0)

        if positive_count == 0:
            per_class_ap[class_name] = 0.0
            continue
        cumulative_true: list[float] = []
        cumulative_false: list[float] = []
        true_sum = false_sum = 0.0
        for true_positive, false_positive in zip(true_positives, false_positives):
            true_sum += true_positive
            false_sum += false_positive
            cumulative_true.append(true_sum)
            cumulative_false.append(false_sum)
        recall = [value / positive_count for value in cumulative_true]
        precision = [
            true_value / (true_value + false_value)
            for true_value, false_value in zip(cumulative_true, cumulative_false)
        ]
        class_ap = _average_precision(recall, precision)
        per_class_ap[class_name] = class_ap
        scored_aps.append(class_ap)

    if not scored_aps:
        raise ValueError("validation data contains no non-difficult ground-truth objects")
    mean_ap = sum(scored_aps) / len(scored_aps)
    return {
        "map": mean_ap,
        "per_class_ap": per_class_ap,
        "num_ground_truth": num_ground_truth,
        "iou_threshold": iou_threshold,
    }


@torch.inference_mode()
def evaluate_detector(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
    iou_threshold: float = 0.5,
    progress_label: str = "validation",
    log_interval: int = 50,
) -> dict[str, object]:
    """검증 로더 전체에 추론을 수행하고 VOC mAP 결과를 반환합니다."""
    model.eval()
    predictions: list[dict[str, Tensor]] = []
    targets: list[dict[str, Tensor]] = []
    total_batches = len(loader)
    started_at = time.monotonic()
    for batch_index, (images, batch_targets) in enumerate(loader, start=1):
        batch_predictions = model([image.to(device) for image in images])
        predictions.extend(
            {key: value.detach().cpu() for key, value in prediction.items()}
            for prediction in batch_predictions
        )
        targets.extend(
            {key: value.detach().cpu() for key, value in target.items()}
            for target in batch_targets
        )
        report_progress(
            progress_label,
            batch_index,
            total_batches,
            started_at,
            detail=f"images={len(targets)}/{len(loader.dataset)}",
            log_interval=log_interval,
        )
    return calculate_voc_map(predictions, targets, class_names, iou_threshold)
