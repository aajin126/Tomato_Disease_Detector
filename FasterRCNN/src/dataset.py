import torch
import cv2
import numpy as np
import os
import glob as glob
import json
from config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
# from utils import collate_fn
from utils import collate_fn, get_train_transform, get_valid_transform

class CropdiseaseDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes

        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.all_images = [image_path.split('\\')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)
        # read the image
        src = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # # capture the corresponding XML file for getting the annotations
        annot_filename = image_name[:-4] + '.jpg.json'
        # annot_file_path = os.path.join(self.dir_path, annot_filename)
        annot_file_path = f'{self.dir_path}{annot_filename}'

        boxes = []
        labels = []

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]





        for filename in os.listdir(self.dir_path):
            if filename.endswith('.json') and filename.startswith(image_name[:-4]):
                file_path = os.path.join(self.dir_path, filename)
                with open(file_path) as f:
                    data = json.load(f)
                    if data["annotations"]["disease"]==0:
                        labels.append(1)
                    else:
                        labels.append(data["annotations"]["disease"])
                    # [x,y,w,h]
                    # labels.append(str(data['annotations'].get('disease')))
                    xmin = int(data["annotations"]['points'][0]["xtl"])
                    xmax = int(data["annotations"]["points"][0]["xbr"])
                    ymin = int(data["annotations"]["points"][0]["ytl"])
                    ymax = int(data["annotations"]["points"][0]["ybr"])

                    xmin_final = (xmin / image_width) * self.width
                    xmax_final = (xmax / image_width) * self.width
                    ymin_final = (ymin / image_height) * self.height
                    ymax_final = (ymax / image_height) * self.height

                    boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])


                # bounding box to tensor
                labels = [int(label) for label in labels]
                # print(labels)
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                # area of the bounding boxes
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                # no crowd instances
                iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
                # labels to tensor

                labels = torch.as_tensor(labels)
                labels[labels==1]=1
                labels[labels == 18] = 2
                labels[labels == 19] = 3


                # labels[labels == 1] = "normal"


                # prepare the final `target` dictionary
                target = {}
                target["boxes"] = boxes
                target["labels"] = labels
                target["area"] = area
                target["iscrowd"] = iscrowd
                image_id = torch.tensor([idx])
                target["image_id"] = image_id
                # apply the image transforms
                if self.transforms:
                    sample = self.transforms(image=image_resized,
                                             bboxes=target['boxes'],
                                             labels=labels)
                    image_resized = sample['image']
                    target['boxes'] = torch.Tensor(sample['bboxes'])

                return image_resized, target

    def __len__(self):
        return len(self.all_images)

# train_dataset = CropdiseaseDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES)
# valid_dataset = CropdiseaseDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES)
train_dataset = CropdiseaseDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())

valid_dataset = CropdiseaseDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)
# print(f"Number of training samples: {len(train_dataset)}")
# print(f"Number of validation samples: {len(valid_dataset)}\n")

if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = CropdiseaseDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    print(f"Number of training images: {len(dataset)}")


    # function to visualize a single sample
    def visualize_sample(image, target):
        print(target)
        box = target['boxes'][0]
        label = CLASSES[target['labels']]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0), 1
        )
        cv2.putText(
            image, str(label), (int(box[0]), int(box[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        cv2.imshow('Image', image)
        cv2.waitKey(0)


    NUM_SAMPLES_TO_VISUALIZE = 3
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)

