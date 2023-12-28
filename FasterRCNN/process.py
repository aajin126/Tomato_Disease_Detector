# 필요없는 클래스 지우는 용도로 사용한 코드 ( 클래스 개수 및 필요없는 클래스 정보 알아내기)
import os
import json
#
directory = "D:/cropData/crop disease data/train/라벨링데이터/11.토마토/[라벨]11.토마토_9.증강/"
# D:\cropData\crop disease data\train\라벨링데이터\11.토마토\[라벨]11.토마토_9.증강
class_counts = {}
count = 0
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        with open(file_path) as f:
            data = json.load(f)
            class_name = str(data['annotations'].get('disease'))
            if class_name in ["11","1","3","15"]:
                print(data["description"]["image"])
                count+=1

            class_counts[class_name] = class_counts.get(class_name, 0) + 1

for class_name, count in class_counts.items():
    print(f"{class_name}: {count}개")
print(count)

# folder_path = "D:/cropData/train/"
#
# image_ext = '.jpg'  # 이미지 파일의 확장자
# json_ext = '.json'  # JSON 파일의 확장자
#
# image_files = []  # 이미지 파일 리스트
# json_files = []
#
# for filename in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, filename)
#     if filename.endswith(image_ext):
#         image_files.append(filename)
#     elif filename.endswith(json_ext):
#         json_files.append(filename)
#
# for img in image_files:
#     for jsonf in json_files:

            # class_counts[class_name] = class_counts.get(class_name, 0) + 1

# for class_name, count in class_counts.items():
#     print(f"{class_name}: {count}개")
# print(count)
# print(set(nname))



# 짝 안맞는거 찾기

import os

# def unify_image_extension(file_path, target_extension=".jpg"):
#     """
#     파일 경로의 이미지 파일 확장자를 지정된 대상 확장자로 통일합니다.
#     """
#     root, ext = os.path.splitext(file_path)
#     if ext.lower() in [".jpg", ".jpeg"]:
#         new_file_path = root + target_extension
#         os.rename(file_path, new_file_path)
#         return new_file_path
#     return file_path
#
# def unify_dataset_extension(dataset_dir, target_extension=".jpg"):
#     """
#     데이터셋 디렉토리 내의 모든 이미지 파일의 확장자를 지정된 대상 확장자로 통일합니다.
#     """
#     for root, dirs, files in os.walk(dataset_dir):
#         for file_name in files:
#             file_path = os.path.join(root, file_name)
#             unify_image_extension(file_path, target_extension)

# def rename_jpeg_to_jpg(file_path):
#     """
#     파일명에서 "jpeg"를 "jpg"로 변경합니다.
#     """
#     directory, filename = os.path.split(file_path)
#     new_filename = filename.replace(".jpeg", ".jpg")
#     new_file_path = os.path.join(directory, new_filename)
#     os.rename(file_path, new_file_path)
#     return new_file_path
#
# def rename_jpeg_to_jpg_in_directory(directory):
#     """
#     지정된 디렉토리 내의 모든 파일명에서 "jpeg"를 "jpg"로 변경합니다.
#     """
#     for root, dirs, files in os.walk(directory):
#         for filename in files:
#             if ".jpeg" in filename:
#                 file_path = os.path.join(root, filename)
#                 rename_jpeg_to_jpg(file_path)
#
#
# # 개별 이미지 파일의 확장자 통일
# file_path = "D:/071.시설 작물 질병 진단/crop disease data/train/"
# unify_image_extension(file_path, ".jpg")
#
# # 전체 데이터셋 디렉토리 내의 이미지 파일 확장자 통일
# dataset_dir = "D:/071.시설 작물 질병 진단/crop disease data/train/"
# unify_dataset_extension(dataset_dir, ".jpg")
#
#
# new_file_path = rename_jpeg_to_jpg(file_path)
# directory ="D:/071.시설 작물 질병 진단/crop disease data/train/"
# # rename_jpeg_to_jpg_in_directory(directory)
# import glob
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
# dir_path = "D:/cropData/crop disease data/train/"
# image_paths = glob.glob(f"{dir_path}*.json")
# all_images = [image_path.split('\\')[-1] for image_path in image_paths]
# print(all_images)
# # # print(image_paths)
# #
# # # print(image_name)
# image_name = all_images[1]
# image_path = os.path.join(f"{dir_path}{image_name}")
# print(image_path)
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
#
# plt.imshow(image)
# plt.show()

# img = cv2.imread(dir_path,image_name)
# plt.imshow(img)
# plt.show()
# labels=[]

# import json
# annot_file_path = "D:/cropData/train/"
#
# label = []
# for filename in os.listdir(annot_file_path):
#     if filename.endswith(('.json')):
#         file_path = os.path.join(annot_file_path,filename)
#         with open(file_path) as f:
#             data = json.load(f)
#             label.append(data["annotations"]["disease"])
#             if data["annotations"]["disease"]==15:
#                 print(data["description"]["image"])
#
# print(set(label))

import os
import json

# folder_path = 'D:/cropData/train/'
#
# for file_name in os.listdir(folder_path):
#     if file_name.endswith('.json'):
#         file_path = os.path.join(folder_path, file_name)
#         with open(file_path, 'r') as f:
#             data = json.load(f)
#             data["annotations"]["disease"]=1
#
# # 예시 사용법
# # JSON 파일들이 들어있는 폴더 경로

import os
import json

import pandas as pd

# image_name = []
# xmin = []
# ymin = []
# xmax = []
# ymax = []
# width = []
# height = []


# for file_name in os.listdir('D:/cropData/train/'):
#     if file_name.endswith('.json'):
#         file_path = os.path.join('D:/cropData/train/', file_name)
#         with open(file_path, 'r') as f:
#             data = json.load(f)
#             image_name.append(data['description']['image'])
#             xmin.append(int(data["annotations"]['points'][0]["xtl"]))
#             ymin.append(data["annotations"]["points"][0]["xbr"])
#             xmax.append(data["annotations"]["points"][0]["ytl"])
#             ymax.append(data["annotations"]["points"][0]["ybr"])
#             width.append(data['description']['width'])
#             height.append(data['description']['height'])
#
# df = pd.DataFrame(zip(image_name, xmin, ymin, xmax,ymax,width,height))
#
# df.columns=["image_name", "xmin", "ymin", "xmax","ymax",'width','height']
# df.to_csv("train_bounidngboxes.csv")

# import json
# import os
#
# def replace_value_in_json(json_file_path, target_value, replacement_value):
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)
#
#     modified = False
#     data[""]
#
#     if modified:
#         with open(json_file_path, 'w') as f:
#             json.dump(data, f)
#
#         print(f"Modified and saved: {json_file_path}")
#
# # 예시 사용법
# folder_path = 'D:/cropData/train/'  # JSON 파일들이 들어있는 폴더 경로
# target_value = 0  # 변경하고자 하는 값
# replacement_value = 1  # 대체할 값
#
# for file_name in os.listdir(folder_path):
#     if file_name.endswith('.json'):
#         file_path = os.path.join(folder_path, file_name)
#         replace_value_in_json(file_path, target_value, replacement_value)


