import config as config 
import os
import time
import warnings
import numpy as np
import torch
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import argparse
import datetime
from torch.utils.tensorboard import SummaryWriter
sys.path.append('./OEM') # <= change path where you save code
import oem

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()


def get_data(args): 

    img_paths = [f for f in Path(config.OEM_DATA_DIR).rglob("*.tif") if "/images/" in str(f)]
    train_fns = [str(f) for f in img_paths if f.name in np.loadtxt(config.TRAIN_LIST, dtype=str)]
    val_fns = [str(f) for f in img_paths if f.name in np.loadtxt(config.VAL_LIST, dtype=str)]


    train_augm = torchvision.transforms.Compose(
    [   
        oem.transforms.Rotate(),
        oem.transforms.Crop(args.img_size),
        oem.transforms.RandomHorizontalFlip(), 
        oem.transforms.GaussianBlur(k_size=(3, 5), sigma=(0.1, 5))
    ],
    )

    val_augm = torchvision.transforms.Compose(
        [
            oem.transforms.Resize(args.img_size),
        ],
    )

    train_data = oem.dataset.OpenEarthMapDataset(
        train_fns, # train data
        n_classes=config.N_CLASSES,
        augm=train_augm,
    )

    val_data = oem.dataset.OpenEarthMapDataset(
        val_fns, # val data
        n_classes=config.N_CLASSES,
        augm=val_augm,
    )

    print("Train data : ", len(train_data))
    print("Val data : ", len(val_data))


    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.bs,
        num_workers=10,
        shuffle=True,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.bs,
        num_workers=10,
        shuffle=False,
    )

    return train_data_loader, val_data_loader 


def get_network(args): 

    if args.transformer: 
        print("Loading transformer ")
        network = oem.networks.UNetFormer(in_channels=3, n_classes=config.N_CLASSES, window_size=args.window_size, decode_channels=args.decode_channels, dropout=args.dropout)
    else: 
        print("Loading normal UNET")
        network = oem.networks.UNet(n_classes=config.N_CLASSES)
    # load pretrained weights if any 
    if args.pretrained_weights:
        try:
            network = oem.utils.load_checkpoint(network, model_name=args.pretrained_weights+"/"+"model.pth", model_dir=config.OUTPUT_DIR)
            print("Successfully loaded pretrained weights !")
        except Exception as e: 
            raise e

    network = network.to(args.device)
    print("Moved network to : ", args.device)
    
    return network


def get_optimizer(network, args): 
    
    if args.optim == "Adam": 
        optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else: 
        raise ValueError("Only Adam or SGD as optimizers")
    return optimizer 

def get_scheduler(args, optimizer): 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.lr)
    return scheduler 


def get_criterion(args): 
    if args.criterion == "Jacard": 
        criterion = oem.losses.JaccardLoss()
    elif args.criterion == "Focal":
        criterion = oem.losses.FocalLoss()
    elif args.criterion == "Dice":
        criterion = oem.losses.DiceLoss()
    elif args.criterion == "MCC": 
        criterion = oem.losses.MCCLoss()
    elif args.criterion == "OHEMB":
        criterion = oem.losses.OHEMBCELoss()
    else:
        raise ValueError("Loss not implemented ...")
    return criterion


def train(args, network, train_loader, val_loader, criterion, optimizer, config_name, scheduler=None, writer=None): 

    if not os.path.exists(config.OUTPUT_DIR + config_name):
        os.makedirs(config.OUTPUT_DIR + config_name)
    else: 
        print("Weight folder already exists ...")


    start = time.time()

    max_score = 0
    for epoch in range(args.num_epochs):

        print(f"\nEpoch: {epoch + 1}")

        train_logs = oem.runners.train_epoch(
            model=network,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_loader,
            device=args.device,
        )

        valid_logs = oem.runners.valid_epoch(
            model=network,
            criterion=criterion,
            dataloader=val_loader,
            device=args.device,
        )

        # writer 
        writer.add_scalar("Train/Loss", train_logs["Loss"], epoch)
        writer.add_scalar("Train/Score", train_logs["Score"], epoch)
        writer.add_scalar("Val/Loss", valid_logs["Loss"], epoch)
        writer.add_scalar("Val/Score", valid_logs["Score"], epoch)
#        writer.add_image("Val/Confusion_matrix", conf_mat[None, :, :], epoch)
        # save model if needed
        epoch_score = valid_logs["Score"]
        if max_score < epoch_score:
            max_score = epoch_score
            oem.utils.save_model(
                model=network,
                epoch=epoch,
                best_score=max_score,
                model_name="model.pth",
                output_dir=config.OUTPUT_DIR + config_name,
            )

        
        if scheduler is not None: 
            scheduler.step() 
            last_lr = scheduler.get_lr()[0]
            print("Last learning rate : ", last_lr)
        else: 
            last_lr = args.lr

        writer.add_scalar("LR_scheduler", last_lr, epoch)
            

    print("Elapsed time: {:.3f} min".format((time.time() - start) / 60.0))

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
    parser.add_argument("--pretrained_weights", type=str, default=config.PRETRAINED_WEIGHTS, required=False)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY, required=False)

    args = parser.parse_args()


    print("Detected GPUS : ", torch.cuda.device_count())
    print("Current GPU : ", torch.cuda.current_device())
    print("Asked GPU : ", config.DEVICE)

    print("--- Loading network ---")    
    network = get_network(args)
    print("--- Loading network : DONE ---", "\n")

    print("--- Loading data ---")
    train_loader, val_loader = get_data(args)
    print("--- Loading data : DONE ---", "\n")

    print("--- Loading optimizer  ---")
    optimizer = get_optimizer(network, args)
    print("--- Loading optimizer : DONE ---", "\n")

    print("--- Loading scheduler ---")
    scheduler = get_scheduler(args, optimizer)
    print("--- Loading scheduler : DONE ---", "\n")

    print("--- Loading criterion ---")
    criterion = get_criterion(args)
    print("--- Loading criterion : DONE ---", "\n")

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    config_name = f"""{'Finetune' if args.pretrained_weights else ''}_backbone={args.backbone}_Unet_optimizer={type(optimizer).__name__}_loss={args.criterion}_dp={args.dropout}_bs={args.bs}_medium_augmentation_window_size={args.window_size}_image_size={args.img_size}_decode_channels={args.decode_channels}LR={args.lr}_NUM_EPOCHS={args.epochs}_scheduler={type(scheduler).__name__ if scheduler is not None else None}_weight_decay={args.weight_decay}""" + current_time

    print("Config name : ", config_name)
    print()

    # writer 
    writer = SummaryWriter(f"runs/{config_name}")
    train(args, network, train_loader, val_loader, criterion, optimizer, config_name=config_name, scheduler=scheduler, writer=writer)




if __name__ == "__main__": 
    main()