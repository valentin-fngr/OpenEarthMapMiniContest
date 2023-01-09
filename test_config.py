import os 



os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'


# Path to the OpenEarthMap directory
OEM_DATA_DIR = "./dataset/OpenEarthMap_Mini/"

# Training and validation file list
TEST_LIST = os.path.join(OEM_DATA_DIR, "test.txt")

IMG_SIZE = 1024
N_CLASSES = 9
LR = 0.0002
BATCH_SIZE = 8
NUM_EPOCHS = 350
IS_TRANSFORMER = True
BACKBONE = "swsl_resnet101_32x4d"
DEVICE = "cuda:1"
OUTPUT_DIR = "./weights"
PRED_DIR =  "./preds"
OPTIMIZER = "Adam"
CRITERION = "Jacard"
WINDOW_SIZE = 8
DECODE_CHANNELS = 64
DROPOUT = 0.2
PRETRAINED_WEIGHTS = "/home/fontanger/remote_sensing/project/weights/_backbone=swsl_resnet101_32x4d_Unet_optimizer=AdamW_loss=Jacard_dp=0.3_bs=8_medium_augmentation_window_size=16_image_size=1024_decode_channels=128LR=0.0005_NUM_EPOCHS=200_scheduler=MultiStepLR_weight_decay=0.000120230103-070458"
