# 🍅TOMATO_DISEASE_DETECTOR

## 🕒개발 기간
23.06.20 - 23.07.03

## 🖥️주제
### 식물 병충해 탐지 
YOLO, Retinanet, Faster R-CNN을 이용하여 식물 병충해를 탐지하는 DETECTOR를 제작하고 세 모델 간의 성능비교와 one stage detector와 two stage detector 간의 성능 비교를 진행했다.

## DATASET
### Ai-hub "시설 작물 질병 진단 이미지" DataSet 
Ai-hub에서 제공하는 dataset 중 토마토의 data를 추출하여 사용했다.
링크 : [Ai-hub "시설 작물 질병 진단 이미지"](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=153)

## YOLOv8 구현
```python
 !pip install ultralytics

 import ultralytics

 ultralytics.checks()

 from ultralytics import YOLO

 model = YOLO('yolov8s.pt')

```
