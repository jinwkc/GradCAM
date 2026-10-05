# Graph Report - ETRI  (2026-10-05)



## Corpus Check
- 10 files · ~11,343 words
- Verdict: corpus is large enough that graph structure adds value.



## Summary
- 202 nodes · 281 edges · 16 communities (9 shown, 7 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.85)
- Token cost: unavailable (semantic extraction usage not returned)



## Community Hubs (Navigation)
- Inception Feature Backbone
- VOC Dataset and Evaluation
- Legacy Inception Model
- Detection and Legacy Grad-CAM
- Grad-CAM Calculation API
- Project Guidance and Documentation
- Grad-CAM Package
- Training Entry Points
- Faster R-CNN Detector
- Legacy Guided Backprop
- VOC Dataset Download
- Training Package
- CIFAR-10 Training Launcher
- Grad-CAM Concepts
- Project Dependencies
- Project Overview



## God Nodes (most connected - your core abstractions)
1. `GradCAM` - 10 edges
2. `BasicConv2d` - 10 edges
3. `BasicConv2d` - 10 edges
4. `evaluate_detector()` - 9 edges
5. `InceptionFPNBackbone` - 8 edges
6. `overlay_detection_cam()` - 8 edges
7. `inception_v3()` - 8 edges
8. `main()` - 8 edges
9. `VOCDetectionDataset` - 7 edges
10. `calculate_voc_map()` - 7 edges



## Surprising Connections (you probably didn't know these)
- `Python Vision Dependencies` --references--> `InceptionFPNBackbone`  [EXTRACTED]
  requirements.txt → models/detection.py
- `Python Vision Dependencies` --references--> `inception_v3()`  [EXTRACTED]
  requirements.txt → models/inception.py
- `Python Vision Dependencies` --references--> `overlay_detection_cam()`  [EXTRACTED]
  requirements.txt → gradcam/visualization.py
- `main()` --calls--> `choose_device()`  [INFERRED]
  detection/detect_gradcam.py → training/TLInception.py
- `main()` --calls--> `choose_device()`  [INFERRED]
  detection/train_detector.py → training/TLInception.py



## Import Cycles
- None detected.



## Communities (16 total, 7 thin omitted)

#

## Community 0 - "Inception Feature Backbone"
Cohesion: 0.11
Nodes (14): 프로젝트의 Inception v3 체크포인트를 사용하는 Faster R-CNN 객체 탐지 모델., BasicConv2d, Inception3, inception_v3(), InceptionA, InceptionAux, InceptionB, InceptionC (+6 more)

#

## Community 1 - "VOC Dataset and Evaluation"
Cohesion: 0.10
Nodes (22): DataLoader, Dataset, Pascal VOC 형식의 XML 바운딩 박스 주석을 읽는 데이터셋., VOC 이미지·XML 주석을 읽고, 요청 시 ``ImageSets/Main`` 분할을 적용합니다., VOCDetectionDataset, _average_precision(), _box_iou(), calculate_voc_map() (+14 more)

#

## Community 2 - "Legacy Inception Model"
Cohesion: 0.13
Nodes (10): BasicConv2d, Inception3, inception_v3(), InceptionA, InceptionAux, InceptionB, InceptionC, InceptionD (+2 more)

#

## Community 3 - "Detection and Legacy Grad-CAM"
Cohesion: 0.08
Nodes (10): main(), Function, deprocess_image(), FeatureExtractor, GradCam, GuidedBackpropReLU, ModelOutputs, see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 (+2 more)

#

## Community 4 - "Grad-CAM Calculation API"
Cohesion: 0.14
Nodes (10): GradCAM, Tensor, 순전파 훅을 제거합니다. 인자: 없음. 반환값: 없음., 스칼라 점수와 모델 레이어 하나를 사용해 Grad-CAM 맵을 생성합니다. 기존 ``gradCam.py``의 활성화값·기울기 가중 방식에…, ``with`` 구문에서 사용할 현재 객체를 반환합니다. 인자: 없음. 반환값: 현재 ``GradCAM`` 인스턴스., ``with`` 구문을 벗어날 때 훅을 제거합니다. 인자: _exc_type: 블록에서 예외가 발생한 경우 그 예외 형식. _exc: 블록에서…, 텐서 출력을 그대로 반환합니다. 인자: output: 대상 모듈에서 가져온 출력. 반환값: 출력 텐서. 예외: TypeError: 출력이…, 순전파 훅에서 선택한 NCHW 활성화값을 저장합니다. 인자: _module: 훅이 등록된 모듈(콜백에서는 사용하지 않음). _inputs:… (+2 more)

#

## Community 5 - "Project Guidance and Documentation"
Cohesion: 0.14
Nodes (16): Project Instructions, Explicit Object Box Annotations, Import Safe Grad-CAM, Inception v3 Transfer Learning, Grad-CAM Public API, Grad-CAM Guide, Detection Score Explanation Overlay, FPN stage2 Feature Selection (+8 more)

#

## Community 6 - "Grad-CAM Package"
Cohesion: 0.15
Nodes (11): 모델 종류에 종속되지 않는 Grad-CAM 기본 기능., 가져오기(import)하여 사용할 수 있는 Grad-CAM API. 인자: 없음. 이 패키지를 가져와도 모델을 불러오거나 파일을 읽지…, overlay_detection_cam(), Path, Tensor, 히트맵 및 바운딩 박스 시각화 도우미., 탐지 결과의 히트맵을 합성하고 모델이 예측한 사각형을 그립니다. JET 히트맵 합성은 기존 ``gradCAM2.py``의 시각화 방식을…, Image (+3 more)

#

## Community 7 - "Training Entry Points"
Cohesion: 0.27
Nodes (11): collate_detection_batch(), main(), parse_classes(), build_datasets(), choose_device(), main(), make_finetune_optimizer(), make_loaders() (+3 more)

#

## Community 8 - "Faster R-CNN Detector"
Cohesion: 0.22
Nodes (9): FasterRCNN, build_inception_faster_rcnn(), InceptionFPNBackbone, Path, Tensor, 객체 탐지기를 구성합니다. 레이블 0은 배경, 1부터 N까지는 객체 클래스입니다., Inception의 세 단계를 Faster R-CNN용 특징 피라미드로 제공합니다., 프로젝트에서 학습한 Inception과 동일한 입력 변환을 적용합니다. (+1 more)



## Knowledge Gaps
- **12 isolated node(s):** `etri`, `Grad-CAM Heatmap Example Image`, `PyTorch and TorchVision`, `download_voc2012.sh script`, `train_cifar10.sh script` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 91 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.



## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Python Vision Dependencies` connect `Grad-CAM Package` to `Faster R-CNN Detector`, `Inception Feature Backbone`?**
  _High betweenness centrality (0.131) - this node is a cross-community bridge._
- **Why does `inception_v3()` connect `Inception Feature Backbone` to `Faster R-CNN Detector`, `Grad-CAM Package`?**
  _High betweenness centrality (0.111) - this node is a cross-community bridge._
- **What connects `etri`, `Grad-CAM Heatmap Example Image`, `PyTorch and TorchVision` to the rest of the system?**
  _12 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Inception Feature Backbone` be split into smaller, more focused modules?**
  _Cohesion score 0.10795454545454546 - nodes in this community are weakly interconnected._
- **Should `VOC Dataset and Evaluation` be split into smaller, more focused modules?**
  _Cohesion score 0.0960591133004926 - nodes in this community are weakly interconnected._
- **Should `Legacy Inception Model` be split into smaller, more focused modules?**
  _Cohesion score 0.1339031339031339 - nodes in this community are weakly interconnected._
- **Should `Detection and Legacy Grad-CAM` be split into smaller, more focused modules?**
  _Cohesion score 0.08307692307692308 - nodes in this community are weakly interconnected._