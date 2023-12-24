# 🍅TOMATO_DISEASE_DETECTOR

## 🕒Development period
23.06.20 - 23.07.03

## 🖥️Subject
### 식물 병충해 탐지 
YOLO, Retinanet, Faster R-CNN을 이용하여 식물 병충해를 탐지하는 DETECTOR를 제작하고 세 모델 간의 성능비교와 one stage detector와 two stage detector 간의 성능 비교를 진행했다.

## DATASET
### Ai-hub "시설 작물 질병 진단 이미지" DataSet 
Ai-hub에서 제공하는 dataset 중 토마토의 data를 추출하여 사용했다.
**링크** : [Ai-hub "시설 작물 질병 진단 이미지"](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=153)

## YOLOv8 구현
#### google colab에서 진행 
```python
 !pip install ultralytics

 import ultralytics

 ultralytics.checks()

 from ultralytics import YOLO

 model = YOLO('yolov8s.pt')

 model.train(data = '/content/drive/MyDrive/dataset_root/data.yaml', epochs = 10, batch= 16)

```
## YOLOv8 결과 확인 
```python
import cv2
from google.colab.patches import cv2_imshow

%cd '/content/drive/MyDrive/ultralytics'
original_image = cv2.imread('/content/drive/MyDrive/KakaoTalk_20230707_143733806.jpg')
model = YOLO('/content/drive/MyDrive/BEST.pt (class7 & class 3)/runs/detect/train3_성공 class 3개/weights/best.pt')
result = model.predict(source='/content/drive/MyDrive/KakaoTalk_20230707_143733806.jpg', save=True, imgsz=640)

# 이미지 출력
cv2_imshow(original_image)
# 이미지 로드
image = cv2.imread('/content/drive/MyDrive/ultralytics/runs/detect/predict24/KakaoTalk_20230707_143733806.jpg')

# 이미지 출력
cv2_imshow(image)
```
자세한 내용은 다음 ppt 자료를 참고해주세요.
[23 봄 기수 오픈 세미나 발표 자료_콤비네이션.pdf](https://github.com/aajinlee/Tomato_Disease_Detector/files/12356812/23._.pdf)
