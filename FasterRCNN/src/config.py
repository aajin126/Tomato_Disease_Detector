import torch
import os
# print(torch.cuda.is_available())

BATCH_SIZE = 4# increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 20 # number of epochs to train for

torch.cuda.set_device(0)  # 0번 GPU를 사용하도록 설정 (필요에 따라 변경 가능)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
TRAIN_DIR = "D:/cropData/train_h/"
# validation images and XML files directory
VALID_DIR = "D:/cropData/validation/"
# classes: 0 index is reserved for background
# CLASSES = [
#     0,"normal",'disease_1','disease_2'
# ]
CLASSES = [
    0,1,2,3
]

NUM_CLASSES = 4
# whether to visualize images after crearing thce data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs
