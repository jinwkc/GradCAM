# Project Instructions

## Project scope

- This project uses a locally defined Inception v3 model for CIFAR-10 transfer learning and is being extended with object detection and Grad-CAM visualization.
- Python is 3.12. Use the existing `uv` project configuration and `.venv`; do not add another package manager or lockfile.
- Keep model code under `models/`, and expose reusable Grad-CAM functionality as imports from the `gradcam/` package.
- Keep CIFAR-10 training entry points under `training/`, detector entry points under `detection/`, and dataset utility scripts under the relevant package's `scripts/` directory.
- Run package entry points from the repository root with `python -m package.module` so project imports resolve consistently.

## Checkpoints and datasets

- The CIFAR-10 checkpoint stores a `model_state_dict` plus class and training metadata. When initializing an object-detection backbone, load only compatible Inception feature weights; do not treat the CIFAR-10 classifier head as a detection head.
- CIFAR-10 provides image-level class labels, not object boxes. Never train or describe an object detector as trained from CIFAR-10 unless separate box annotations are provided.
- Object-detection training must consume an explicitly annotated dataset and preserve the dataset's declared class-to-label mapping.
- Keep downloaded datasets and trained checkpoints under the ignored `data/` and `checkpoints/` directories. Do not delete or overwrite an existing checkpoint unless the user requests it.

## Grad-CAM and visualization

- Grad-CAM modules must be import-safe: no model downloads, image reads, training, or UI work at import time.
- Keep device selection, paths, thresholds, and class labels explicit; do not hard-code CUDA, VGG, or a particular user's paths.
- For detector explanations, target a selected detection's class score and display its predicted box with its heatmap. A Grad-CAM heatmap is an explanation overlay, not a box annotation or proof of localization accuracy.
- Use the project checkpoint and detector checkpoint formats explicitly; fail with a clear message when a required checkpoint or annotation is missing or incompatible.

## Change boundaries

- Preserve unrelated working-tree files and user data. Do not stage, commit, or remove generated data or checkpoints.
- Keep CLI usage documented in each executable's `--help` text and in concise project documentation when a dataset layout is required.
- Report which validation checks were run and which were not; do not claim detector accuracy without annotated validation data.

## 문서 작성

- 한국어 문서의 설명 문장과 항목은 명사형 종결을 기본 원칙으로 적용
- 종결 예시: `사용합니다`는 `사용`, `저장됩니다`는 `저장`으로 작성
- 코드 식별자, CLI 옵션, 인용 원문은 원형 유지
