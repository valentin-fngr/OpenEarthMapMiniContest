import os 

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'

# Path to the OpenEarthMap directory
OEM_DATA_DIR = "./dataset/OpenEarthMap_Mini/"

# Training and validation file list
TRAIN_LIST = os.path.join(OEM_DATA_DIR, "train.txt")
VAL_LIST = os.path.join(OEM_DATA_DIR, "val.txt")

IMG_SIZE = 1024
N_CLASSES = 9
LR = 0.0005
BATCH_SIZE = 32
NUM_EPOCHS = 350
IS_TRANSFORMER = True
BACKBONE = "swsl_resnet101_32x4d"
DEVICE = "cuda:1"
OUTPUT_DIR = "./weights/"
OPTIMIZER = "Adam"
CRITERION = "Jacard"
WINDOW_SIZE = 8
DECODE_CHANNELS = 64
DROPOUT = 0.28
PRETRAINED_WEIGHTS= ""
WEIGHT_DECAY = 0.001
