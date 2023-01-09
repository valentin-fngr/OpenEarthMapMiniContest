import os 
import test_config as config
import numpy as np
import torch
import torchvision
from pathlib import Path
import math
from PIL import Image
import sys
sys.path.append('./OEM') # <= change path where you save code
import oem
import argparse


def get_data(): 

    img_paths = [f for f in Path(config.OEM_DATA_DIR).rglob("*.tif") if "/images/" in str(f)]
    test_fns = [str(f) for f in img_paths if f.name in np.loadtxt(config.TEST_LIST, dtype=str)]
    test_data = oem.dataset.OpenEarthMapDataset(test_fns, n_classes=config.N_CLASSES, augm=None, testing=True)

    return test_data



def get_network(args, config_name): 
    if args.transformer: 
        print("Loading transformer ")
        network = oem.networks.UNetFormer(in_channels=3, n_classes=config.N_CLASSES, window_size=args.window_size, decode_channels=args.decode_channels, dropout=args.dropout)
    else: 
        print("Loading normal UNET")
        network = oem.networks.UNet(n_classes=config.N_CLASSES)
    
    network = network.to(args.device)
    print("Moved network to : ", args.device)

    try:
        network = oem.utils.load_checkpoint(network, model_name=args.pretrained_weights +"/"+"model.pth", model_dir=config.OUTPUT_DIR)
    except Exception as e: 
        raise e

    network.eval()
    network = network.to(config.DEVICE)
    return network


def test(network, config_name): 


    if not os.path.exists(config.PRED_DIR + "/" + config_name): 
        os.mkdir(config.PRED_DIR + "/" + config_name)
    else: 
        print("Path already exists")


    img_paths = [f for f in Path(config.OEM_DATA_DIR).rglob("*.tif") if "/images/" in str(f)]
    test_fns = [str(f) for f in img_paths if f.name in np.loadtxt(config.TEST_LIST, dtype=str)]

    print("Total samples   :", len(img_paths))
    print("Testing samples :", len(test_fns))

    save_fns = []

    for test_fn in test_fns:
        img = Image.fromarray(oem.dataset.load_multiband(test_fn))

        w, h = img.size[:2]
        power_h = math.ceil(np.log2(h) / np.log2(2))
        power_w = math.ceil(np.log2(w) / np.log2(2))
        if 2**power_h != h or 2**power_w != w:
            img = img.resize((2**power_w, 2**power_h), resample=Image.BICUBIC)
        img = np.array(img)

        # test time augmentation
        imgs = []
        imgs.append(img.copy())
        imgs.append(img[:, ::-1, :].copy())
        imgs.append(img[::-1, :, :].copy())
        imgs.append(img[::-1, ::-1, :].copy())

        input = torch.cat([torchvision.transforms.functional.to_tensor(x).unsqueeze(0) for x in imgs], dim=0).float().to(config.DEVICE)

        pred = []
        with torch.no_grad():
            msk = network(input) 
            msk = torch.softmax(msk[:, :, ...], dim=1)
            msk = msk.cpu().numpy()
            pred = (msk[0, :, :, :] + msk[1, :, :, ::-1] + msk[2, :, ::-1, :] + msk[3, :, ::-1, ::-1])/4

            pred = Image.fromarray(pred.argmax(axis=0).astype("uint8"))
            y_pr = pred.resize((w, h), resample=Image.NEAREST)

            filename = os.path.basename(test_fn).replace('tif','png')
            save_fn = os.path.join(config.PRED_DIR + "/" + config_name, filename)
            y_pr.save(save_fn)
            save_fns.append(save_fn)


    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", required=False)
    parser.add_argument("--img_size", type=int, default=config.IMG_SIZE, required=False)
    parser.add_argument("--lr", type=float, default=config.LR, required=False) 
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, required=False) 
    parser.add_argument("--bs", type=int, default=config.BATCH_SIZE, required=False) 
    parser.add_argument("--transformer", type=lambda x: bool(int(x.lower())), default=config.IS_TRANSFORMER, required=False) 
    parser.add_argument("--backbone", type=str, default=config.BACKBONE, required=False) 
    parser.add_argument("--num_epochs", type=int, default=config.NUM_EPOCHS, required=False) 
    parser.add_argument("--optim", type=str, default=config.OPTIMIZER, required=False)
    parser.add_argument("--criterion", type=str, default=config.CRITERION, required=False)
    parser.add_argument("--window_size", type=int, default=config.WINDOW_SIZE, required=False)
    parser.add_argument("--decode_channels", type=int, default=config.DECODE_CHANNELS, required=False)
    parser.add_argument("--dropout", type=float, default=config.DROPOUT, required=False)
    parser.add_argument("--pretrained_weights", type=str, default=config.PRETRAINED_WEIGHTS, required=True)
    args = parser.parse_args()

    config_name = f"""is_transformer={args.transformer}_backbone={args.backbone}_Unet_optimizer={args.optim}_loss_fctn={args.criterion}_batch_size={args.bs}_medium_augmentation_window_size={args.window_size}_image_size={args.img_size}_decode_channels={args.decode_channels}LR={args.lr}_NUM_EPOCHS={args.epochs}_scheduler=MultiStepLR"""

    # check if config exists 
    if not os.path.exists(os.path.join(config.OUTPUT_DIR, config_name)): 
        raise ValueError("Cannot find model checkpoint : ", config_name)    

    print("Found model checkpoint ! ", "\n")

    print("--- Loading model ---")
    network = get_network(args, config_name)
    print("--- Loading model --- : DONE", "\n")
    print(network)

    test(network, config_name=args.pretrained_weights)



if __name__ == "__main__": 

    main()