"""Fine-tune torchvision's pretrained Inception v3 on CIFAR-10."""

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import Inception_V3_Weights

from models.inception import inception_v3


NUM_CLASSES = 10
IMAGE_SIZE = 299


def choose_device(requested):
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


def build_datasets(data_dir, validation_fraction, seed, download):
    weights = Inception_V3_Weights.DEFAULT
    weight_transforms = weights.transforms()
    normalize = transforms.Normalize(
        mean=weight_transforms.mean,
        std=weight_transforms.std,
    )
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        normalize,
    ])

    train_source = datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=train_transform
    )
    eval_source = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=eval_transform
    )
    test_source = datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=eval_transform
    )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_source), generator=generator).tolist()
    validation_size = max(1, int(len(indices) * validation_fraction))
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    return (
        Subset(train_source, training_indices),
        Subset(eval_source, validation_indices),
        test_source,
        train_source.classes,
    )


def make_loaders(datasets_, batch_size, num_workers, device, seed):
    train_dataset, validation_dataset, test_dataset, _ = datasets_
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        common["persistent_workers"] = True

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, validation_loader, test_loader


def set_head_only_training(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def make_finetune_optimizer(model, backbone_lr, head_lr, weight_decay):
    head_parameter_ids = {id(parameter) for parameter in model.fc.parameters()}
    backbone_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in head_parameter_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": model.fc.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    freeze_batch_norm=False,
    progress_label=None,
    log_interval=50,
):
    training = optimizer is not None
    model.train(training)
    if training and freeze_batch_norm:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(loader)
    started_at = time.monotonic()
    interactive = sys.stdout.isatty()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            batch_loss = loss.item()
            batch_correct = (logits.argmax(dim=1) == labels).sum().item()
            total_loss += batch_loss * batch_size
            correct += batch_correct
            total += batch_size

            should_report = (
                progress_label is not None
                and (
                    interactive
                    or batch_index % log_interval == 0
                    or batch_index == total_batches
                )
            )
            if should_report:
                elapsed = time.monotonic() - started_at
                eta = elapsed / batch_index * (total_batches - batch_index)
                bar_width = 20
                filled = int(bar_width * batch_index / total_batches)
                bar = "=" * filled + "." * (bar_width - filled)
                message = (
                    f"{progress_label} {batch_index}/{total_batches} [{bar}] "
                    f"batch_loss={batch_loss:.4f} "
                    f"batch_acc={batch_correct / batch_size:.4f} "
                    f"loss={total_loss / total:.4f} acc={correct / total:.4f} "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
                )
                if interactive:
                    end = "\n" if batch_index == total_batches else ""
                    sys.stdout.write(f"\r{message}{end}")
                    sys.stdout.flush()
                else:
                    print(message, flush=True)

    return total_loss / total, correct / total


def save_checkpoint(path, model, classes, epoch, validation_accuracy, stage):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_state_dict": state,
            "classes": classes,
            "num_classes": NUM_CLASSES,
            "epoch": epoch,
            "stage": stage,
            "validation_accuracy": validation_accuracy,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Transfer-learn pretrained Inception v3 on CIFAR-10"
    )

    # 실행 인자 요약:
    # --data-dir: CIFAR-10 저장 위치 (기본값: data; 없으면 다운로드)
    # --checkpoint: 검증 정확도가 가장 높은 모델 저장 경로
    # --device: auto( CUDA → MPS → CPU 자동 선택 ), cpu, mps 또는 cuda
    # --batch-size / --num-workers: 배치 크기(최소 2) / 데이터 로더 프로세스 수
    # --head-epochs: backbone을 고정하고 새 분류층만 학습하는 epoch 수
    # --finetune-epochs: 전체 모델을 미세 조정하는 epoch 수
    # --head-lr / --backbone-lr: 분류층 / backbone 학습률
    # --weight-decay: AdamW weight decay 값
    # --validation-fraction: 학습 세트에서 검증 세트로 나눌 비율
    # --seed: 데이터 분할과 학습 재현성 시드
    # --no-download: CIFAR-10 자동 다운로드 비활성화; 사전학습 가중치 다운로드에는 영향 없음
    # --log-interval: 터미널이 아닌 출력에서 진행 상황을 출력할 배치 간격
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/inception_v3_cifar10_best.pth"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--head-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2 for Inception BatchNorm")
    if args.num_workers < 0:
        parser.error("--num-workers must not be negative")
    if args.log_interval < 1:
        parser.error("--log-interval must be positive")
    if args.head_epochs < 0 or args.finetune_epochs < 0:
        parser.error("epoch counts must not be negative")
    if args.head_epochs + args.finetune_epochs == 0:
        parser.error("at least one training epoch is required")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"Device: {device}")

    dataset_bundle = build_datasets(
        args.data_dir,
        args.validation_fraction,
        args.seed,
        download=not args.no_download,
    )
    train_loader, validation_loader, test_loader = make_loaders(
        dataset_bundle,
        args.batch_size,
        args.num_workers,
        device,
        args.seed,
    )
    classes = dataset_bundle[3]

    # This obtains the official ImageNet weights on first run.
    model = inception_v3(
        weights=Inception_V3_Weights.DEFAULT,
        num_classes=NUM_CLASSES,
        aux_logits=False,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    best_validation_accuracy = -1.0
    epoch_number = 0
    total_epochs = args.head_epochs + args.finetune_epochs

    if args.head_epochs:
        set_head_only_training(model)
        optimizer = torch.optim.AdamW(
            model.fc.parameters(), lr=args.head_lr, weight_decay=args.weight_decay
        )
        for _ in range(args.head_epochs):
            epoch_number += 1
            train_loss, train_accuracy = run_epoch(
                model, train_loader, criterion, device,
                optimizer=optimizer, freeze_batch_norm=True,
                progress_label=f"Epoch {epoch_number}/{total_epochs} [head/train]",
                log_interval=args.log_interval,
            )
            validation_loss, validation_accuracy = run_epoch(
                model,
                validation_loader,
                criterion,
                device,
                progress_label=f"Epoch {epoch_number}/{total_epochs} [val]",
                log_interval=args.log_interval
            )
            print(
                f"head epoch {epoch_number}: train_loss={train_loss:.4f} "
                f"train_acc={train_accuracy:.4f} val_loss={validation_loss:.4f} "
                f"val_acc={validation_accuracy:.4f}"
            )
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                save_checkpoint(
                    args.checkpoint, model, classes, epoch_number,
                    validation_accuracy, "head",
                )

    if args.finetune_epochs:
        for parameter in model.parameters():
            parameter.requires_grad = True
        optimizer = make_finetune_optimizer(
            model, args.backbone_lr, args.head_lr, args.weight_decay
        )
        for _ in range(args.finetune_epochs):
            epoch_number += 1
            train_loss, train_accuracy = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                progress_label=f"Epoch {epoch_number}/{total_epochs} [finetune/train]",
                log_interval=args.log_interval
            )
            validation_loss, validation_accuracy = run_epoch(
                model,
                validation_loader,
                criterion,
                device,
                progress_label=f"Epoch {epoch_number}/{total_epochs} [val]",
                log_interval=args.log_interval
            )
            print(
                f"finetune epoch {epoch_number}: train_loss={train_loss:.4f} "
                f"train_acc={train_accuracy:.4f} val_loss={validation_loss:.4f} "
                f"val_acc={validation_accuracy:.4f}"
            )
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                save_checkpoint(
                    args.checkpoint, model, classes, epoch_number,
                    validation_accuracy, "finetune",
                )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
    print(
        f"best checkpoint: {args.checkpoint} "
        f"(epoch={checkpoint['epoch']}, val_acc={checkpoint['validation_accuracy']:.4f})"
    )
    print(f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}")


if __name__ == "__main__":
    main()
