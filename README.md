<<<<<<< HEAD
# 🍅 TOMATO_DISEASE_DETECTOR
## 🕒Development period
23.06.20 - 23.07.03

## 🖥️Subject
### 식물 병충해 탐지 
YOLO, Retinanet, Faster R-CNN을 이용하여 식물 병충해를 탐지하는 DETECTOR를 제작하고 세 모델 간의 성능비교와 one stage detector와 two stage detector 간의 성능 비교를 진행했다.

## DATASET
### Ai-hub "시설 작물 질병 진단 이미지" DataSet 
Ai-hub에서 제공하는 dataset 중 토마토의 data를 추출하여 사용했다.
**링크** : [Ai-hub "시설 작물 질병 진단 이미지"](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=153)

## Result
### 정상
먼저, 정상 식물에 대해 세 가지 모델로 탐지한 결과입니다. Faster R-CNN만이 정상 식물로 탐지했고 YOLOv8과 RetinaNet은 황화잎말이바이러스로 탐지했습니다. 상대적으로 정상 이미지 개수가 적었고 말려있듯한 잎 때문에 질병2로 탐지해냈음을 알 수 있었습니다. 
| Faster R-CNN        | YOLOv8                     | RetinaNet                                |
|---------------------------------|------------------------------------|-----------------------------------------------|
| <p align="center"><img src="./docs/normal_rcnn.jpg" width="300px" height="300px"></p> | <p align="center"><img src="./docs/normal_yolov8.jpg" width="300px" height="300px"></p> | <p align="center"><img src='./docs/normal_retinanet.png' width="300px" height="300px"></p>|

### 질병1(토마토 잎 곰팡이병)
다음은 첫 번째 질병인 토마토 잎 곰팡이병입니다. 잎 뒷면에 병반이 생기고 갈색 곰팡이가 생기면서 황색으로 변하는 특징이 있습니다. 세 탐지 모델 모두 잘 탐지해냈습니다. 
| Faster R-CNN        | YOLOv8                     | RetinaNet                                |
|---------------------------------|------------------------------------|-----------------------------------------------|
| <p align="center"><img src="./docs/disease1_rcnn.jpg" width="300px" height="300px"></p> | <p align="center"><img src="./docs/disease1_yolov8.jpg" width="300px" height="300px"></p> | <p align="center"><img src='./docs/disease1_retinanet.png' width="300px" height="300px"></p>|

### 질병2(황화잎말이바이러스)
그리고 두 번째 질병인 황화잎말이바이러스입니다. 황화잎말이바이러스는 어린잎이 쪼그라들고 노랗게 변한다는 특징이 있습니다. 세 탐지 모델 모두 잘 탐지해냈습니다. 
| Faster R-CNN        | YOLOv8                     | RetinaNet                                |
|---------------------------------|------------------------------------|-----------------------------------------------|
| <p align="center"><img src="./docs/disease2_rcnn.jpg" width="300px" height="300px"></p> | <p align="center"><img src="./docs/disease2_yolov8.jpg" width="300px" height="300px"></p> | <p align="center"><img src='./docs/disease2_retinanet.png' width="300px" height="300px"></p>|

## Inference
### YOLOv8
#### google colab에서 진행 
```python
 !pip install ultralytics

 import ultralytics

 ultralytics.checks()

 from ultralytics import YOLO

 model = YOLO('yolov8s.pt')

 model.train(data = '/content/drive/MyDrive/dataset_root/data.yaml', epochs = 10, batch= 16)

```
#### 결과 확인  
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

### Faster R-CNN 구현
#### train code : FasterRCNN/src/engine
```python

#20 epoch, 4 batch 학습

#class 분류
#0 : background
#1 : 정상
#2 : 18번 질병 - 토마토잎곰팡이병
#3 : 19번 질병 - 토마토황화잎말이바이러스

```
### Faster R-CNN 결과 확인
#### inference code : FasterRCNN/src/inference
```python

detection_threshold = 0.4

for i in range(len(test_images)):
    image_name = test_images[i].split('\\')[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(float)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1)).astype(float)
    image = torch.tensor(image, dtype=torch.float).cuda()
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 0, 255), 2)


            cv2.putText(orig_image, str(pred_classes[j]),
                        (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),
                        5, lineType=cv2.LINE_AA)


        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(1)
        cv2.imwrite(f"../test_predictions/{image_name}.jpg", orig_image, )
    print(f"Image {i + 1} done...")
    print('-' * 50)
print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()

```
## RetinaNet
#### Inference
```python
# load retinanet model
model = models.load_model('/content/drive/MyDrive/tomato/resnet50_csv_13_infer.h5', backbone_name='resnet50')

# load label to names mapping for visualization purposes
labels_to_names = {0:'normal',1:'disease1',2:'disease2'}
# load image
image = read_image_bgr('/content/test.JPG')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

# correct for image scale
boxes /= scale

for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.45:
        break

    color = label_color(label)

    b = box.astype(int)
    draw_box(draw, b, color=color)

    caption = "{}".format(labels_to_names[label])
    cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 3)
    cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 2)

plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()
```

자세한 내용은 다음 ppt 자료를 참고해주세요.
[23 봄 기수 오픈 세미나 발표 자료_콤비네이션.pdf](https://github.com/aajinlee/Tomato_Disease_Detector/files/12356812/23._.pdf)
