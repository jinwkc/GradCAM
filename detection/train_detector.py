"""Pascal VOC 형식 데이터셋으로 Inception-FPN Faster R-CNN을 학습합니다."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from detection import VOCDetectionDataset, report_progress
from detection.metrics import evaluate_detector
from models.detection import build_inception_faster_rcnn, choose_device


def parse_classes(value: str) -> list[str]:
    classes = [part.strip() for part in value.split(",") if part.strip()]
    if not classes:
        raise argparse.ArgumentTypeError("provide comma-separated class names")
    if len({name.casefold() for name in classes}) != len(classes):
        raise argparse.ArgumentTypeError("class names must be unique")
    return classes


def collate_detection_batch(batch):
    return tuple(zip(*batch))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Faster R-CNN with the project's CIFAR-10 Inception checkpoint "
            "and evaluate Pascal VOC mAP"
        )
    )
    parser.add_argument("--data", type=Path, required=True,
                        help="VOC-style root containing JPEGImages/ and Annotations/")
    parser.add_argument("--classes", type=parse_classes, required=True,
                        help="comma-separated object classes in XML spelling")
    parser.add_argument("--classification-checkpoint", type=Path,
                        default=Path("checkpoints/inception_v3_cifar10_best.pth"),
                        help="CIFAR-10 Inception checkpoint used to initialize the backbone")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/inception_fpn_fasterrcnn.pth"),
                        help="best validation-mAP detection checkpoint")
    parser.add_argument("--train-split", default="train",
                        help="name under ImageSets/Main for training (default: train)")
    parser.add_argument("--validation-split", default="val",
                        help="name under ImageSets/Main for validation (default: val)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for VOC AP/mAP (default: 0.5)")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="print redirected batch progress every N batches (default: 50)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs and batch-size must be positive; num-workers cannot be negative")
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("lr must be positive and weight-decay cannot be negative")
    if args.train_split == args.validation_split:
        parser.error("training and validation splits must be different")
    if not 0.0 < args.iou_threshold <= 1.0:
        parser.error("iou-threshold must be in the interval (0, 1]")
    if args.log_interval < 1:
        parser.error("log-interval must be positive")
    if not args.classification_checkpoint.is_file():
        parser.error(f"classification checkpoint not found: {args.classification_checkpoint}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    train_dataset = VOCDetectionDataset(args.data, args.classes, split=args.train_split)
    validation_dataset = VOCDetectionDataset(
        args.data, args.classes, split=args.validation_split
    )
    overlap = set(train_dataset.image_ids) & set(validation_dataset.image_ids)
    if overlap:
        parser.error(
            f"training and validation splits overlap ({len(overlap)} image IDs); "
            f"example: {sorted(overlap)[0]}"
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_detection_batch,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_detection_batch,
    )
    model = build_inception_faster_rcnn(
        args.classes, classification_checkpoint=args.classification_checkpoint
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(f"Device: {device}", flush=True)
    print(
        f"Training images: {len(train_dataset)} ({args.train_split}); "
        f"validation images: {len(validation_dataset)} ({args.validation_split}); "
        f"classes: {args.classes}",
        flush=True,
    )
    best_map = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        component_loss_sums: dict[str, float] = {}
        epoch_started_at = time.monotonic()
        total_batches = len(train_loader)
        for batch_index, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite detection loss at epoch {epoch}: {loss.item()}")
            loss.backward()
            optimizer.step()
            batch_loss = float(loss.detach())
            total_loss += batch_loss
            for name, value in loss_dict.items():
                component_loss_sums[name] = (
                    component_loss_sums.get(name, 0.0) + float(value.detach())
                )
            component_text = " ".join(
                f"{name}={value / batch_index:.3f}"
                for name, value in component_loss_sums.items()
            )
            report_progress(
                f"Epoch {epoch}/{args.epochs} train",
                batch_index,
                total_batches,
                epoch_started_at,
                detail=(
                    f"loss={total_loss / batch_index:.4f} {component_text}"
                ),
                log_interval=args.log_interval,
            )

        mean_loss = total_loss / max(1, len(train_loader))
        validation_metrics = evaluate_detector(
            model,
            validation_loader,
            device,
            args.classes,
            iou_threshold=args.iou_threshold,
            progress_label=f"Epoch {epoch}/{args.epochs} validation",
            log_interval=args.log_interval,
        )
        validation_map = float(validation_metrics["map"])
        per_class_ap = validation_metrics["per_class_ap"]
        is_best = validation_map > best_map
        if is_best:
            best_map = validation_map
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "class_names": args.classes,
                    "epoch": epoch,
                    "mean_training_loss": mean_loss,
                    "validation_map": validation_map,
                    "per_class_ap": per_class_ap,
                    "iou_threshold": args.iou_threshold,
                    "validation_split": args.validation_split,
                    "backbone_checkpoint": str(args.classification_checkpoint),
                },
                args.checkpoint,
            )
        class_ap_text = ", ".join(
            f"{name}={float(ap):.3f}" for name, ap in per_class_ap.items()
        )
        print(
            f"Epoch {epoch}/{args.epochs} - loss: {mean_loss:.4f} - "
            f"val_mAP@{args.iou_threshold:.2f}: {validation_map:.4f} - "
            f"best: {best_map:.4f} - "
            f"checkpoint {'saved' if is_best else 'unchanged'}: {args.checkpoint}",
            flush=True,
        )
        print(f"  validation AP: {class_ap_text}", flush=True)


if __name__ == "__main__":
    main()
