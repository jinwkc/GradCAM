"""Pascal VOC 형식의 XML 바운딩 박스 주석을 읽는 데이터셋."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class VOCDetectionDataset(Dataset):
    """VOC 이미지·XML 주석을 읽고, 요청 시 ``ImageSets/Main`` 분할을 적용합니다."""

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        root: str | Path,
        class_names: Sequence[str],
        split: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "JPEGImages"
        self.annotation_dir = self.root / "Annotations"
        self.class_names = list(class_names)
        self.label_ids = {
            name.casefold(): index + 1 for index, name in enumerate(self.class_names)
        }

        if not self.image_dir.is_dir() or not self.annotation_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {self.image_dir} and {self.annotation_dir} "
                "(Pascal VOC-style dataset layout)"
            )
        available_images = {
            path.stem: path for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in self.extensions
        }
        if split is None:
            self.image_ids = sorted(available_images)
        else:
            if Path(split).name != split:
                raise ValueError(f"Invalid VOC split name: {split!r}")
            split_file = self.root / "ImageSets" / "Main" / f"{split}.txt"
            if not split_file.is_file():
                raise FileNotFoundError(f"VOC split file not found: {split_file}")
            self.image_ids = [
                line.strip().split()[0]
                for line in split_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(self.image_ids) != len(set(self.image_ids)):
                raise ValueError(f"Duplicate image IDs in VOC split file: {split_file}")
            missing_ids = [image_id for image_id in self.image_ids if image_id not in available_images]
            if missing_ids:
                raise FileNotFoundError(
                    f"{len(missing_ids)} image(s) from {split_file} are missing; "
                    f"first missing image ID: {missing_ids[0]}"
                )
        self.images = [available_images[image_id] for image_id in self.image_ids]
        if not self.images:
            raise ValueError(f"No images found for split {split!r} in {self.image_dir}")

        missing_xml = [
            image.with_suffix(".xml").name
            for image in self.images
            if not (self.annotation_dir / image.with_suffix(".xml").name).is_file()
        ]
        if missing_xml:
            raise FileNotFoundError(
                f"Missing XML annotations for {len(missing_xml)} image(s); "
                f"first missing file: {missing_xml[0]}"
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        annotation_path = self.annotation_dir / image_path.with_suffix(".xml").name
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            width, height = image.size
            image_tensor = to_tensor(image)

        try:
            root = ET.parse(annotation_path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"Invalid Pascal VOC XML: {annotation_path}") from exc

        boxes: list[list[float]] = []
        labels: list[int] = []
        difficult_flags: list[int] = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            label = self.label_ids.get(name.casefold())
            if label is None:
                raise ValueError(
                    f"Unknown class {name!r} in {annotation_path}; "
                    f"declared classes: {self.class_names}"
                )

            box = obj.find("bndbox")
            if box is None:
                raise ValueError(f"Object without bndbox in {annotation_path}")
            try:
                # VOC 좌표는 1부터 시작하고 torchvision 이미지 경계 좌표는 0부터 시작합니다.
                x1 = max(0.0, float(box.findtext("xmin")) - 1.0)
                y1 = max(0.0, float(box.findtext("ymin")) - 1.0)
                x2 = min(float(width), float(box.findtext("xmax")))
                y2 = min(float(height), float(box.findtext("ymax")))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid bndbox in {annotation_path}") from exc
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                raise ValueError(f"Non-finite bndbox in {annotation_path}")
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Degenerate bndbox in {annotation_path}: {(x1, y1, x2, y2)}")
            boxes.append([x1, y1, x2, y2])
            labels.append(label)
            difficult_text = (obj.findtext("difficult") or "0").strip()
            try:
                difficult_value = int(difficult_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid difficult flag in {annotation_path}: {difficult_text!r}"
                ) from exc
            difficult_flags.append(int(difficult_value != 0))

        box_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.as_tensor(labels, dtype=torch.int64)
        area = (
            (box_tensor[:, 2] - box_tensor[:, 0])
            * (box_tensor[:, 3] - box_tensor[:, 1])
        )
        target = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
            "difficult": torch.as_tensor(difficult_flags, dtype=torch.int64),
        }
        return image_tensor, target
