import os
import re
import torch
import datetime
import argparse

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler

from timm.utils import NativeScaler

from models.build_models import MSCAEfficientFormerV2

from datasets import build_dataset
from datasets.kvasir import get_transform

from utils.losses import get_loss, DiceBCELoss
from utils.schedulers import get_scheduler, create_lr_scheduler
from utils.optimizers import get_optimizer, Lion
from utils import distributed_utils as dist
import utils

from terminaltables import AsciiTable
from utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from engine import train_one_epoch, evaluate


def get_argparser():
    parser = argparse.ArgumentParser('Pytorch MSCA-EfficientFormer Model training and evaluation script', add_help=False)

    # Datset Options
    parser.add_argument("--Kvasir_path", type=str, default='/mnt/d/MedicalSeg/Kvasir-SEG/',
                        help="path to Kvasir Dataset")
    parser.add_argument("--ClinicDB_path", type=str, default='/mnt/d/MedicalSeg/CVC-ClinicDB/',
                        help="path to CVC-ClinicDBDataset")
    parser.add_argument("--img_size", type=int, default=224, help="input size")
    parser.add_argument("--ignore_label", type=int, default=255, help="the dataset ignore_label")
    parser.add_argument("--ignore_index", type=int, default=255, help="the dataset ignore_index")
    parser.add_argument("--dataset", type=str, default='kvasir & CVC-ClinicDB',
                        choices=['cityscapes', 'pascal', 'coco', 'synapse', 'kvasir & CVC-ClinicDB'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2, help="num classes (default: 2 for kvasir & CVC-ClinicDB)")
    parser.add_argument("--pin_mem", type=bool, default=True, help="Dataloader ping_memory")
    parser.add_argument("--batch_size", type=int, default=4, help='batch size (default: 4)')
    parser.add_argument("--val_batch_size", type=int, default=1, help='batch size for validation (default: 1)')

    # DA-TransUNet Options
    parser.add_argument("--model", type=str, default='MSCAEfficientFormerV2',
                        choices=['MSCAEfficientFormerV2', ], help='model type')

    # Train Options
    # parser.add_argument("--amp", type=bool, default=True, help='auto mixture precision')
    parser.add_argument("--epochs", type=int, default=2, help='total training epochs')
    parser.add_argument("--device", type=str, default='cuda', help='device (cuda:0 or cpu)')
    parser.add_argument("--num_workers", type=int, default=0, help='num_workers, set it equal 0 when run programs in windows platform')
    parser.add_argument("--train_print_freq", type=int, default=100)
    parser.add_argument("--val_print_freq", type=int, default=50)

    # Loss Options
    parser.add_argument("--loss_fn_name", type=str, default='DiceBCELoss')

    # Optimizer & LR-scheduler Options
    parser.add_argument("--optimizer", type=str, default='adamw')
    parser.add_argument('--clip-grad', type=float, default=0.02, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='agc',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help='weight decay (default: 1e-5)')

    parser.add_argument("--lr_scheduler", type=str, default='WarmupPolyLR')
    parser.add_argument("--lr_power", type=float, default=0.9)
    parser.add_argument("--lr_warmup", type=int, default=10)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.1)

    # save checkpoints
    parser.add_argument("--save_weights_dir", default='./save_weights', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--save_dir", type=str, default='./', help='SummaryWriter save dir')
    # parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")

    # training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):
    print(args)
    utils.init_distributed_mode(args)

    if not os.path.exists(args.save_weights_dir):
        os.makedirs(args.save_weights_dir, exist_ok=True)

    # start = time.time()
    best_mIoU = 0.0
    best_F1 = 0.0
    best_acc = 0.0
    device = args.device

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_set, valid_set = build_dataset(args)


    if args.distributed:
        sampler_train = DistributedSampler(train_set, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        sampler_val = DistributedSampler(valid_set)
    else:
        sampler_train = RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(valid_set)

    trainloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                             drop_last=True, pin_memory=args.pin_mem, sampler=sampler_train)

    valloader = DataLoader(valid_set, batch_size=args.val_batch_size, num_workers=args.num_workers,
                           drop_last=True, pin_memory=args.pin_mem, sampler=sampler_val)

    model = MSCAEfficientFormerV2(img_size=args.img_size, num_classes=args.num_classes)
    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('********ESTABLISH ARCHITECTURE********')
    print(f'Model: {args.model}\nNumber of parameters: {n_parameters}')
    print('**************************************')


    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    # args.lr = linear_scaled_lr
    #
    # print('*****************')
    # print('Initial LR is ', linear_scaled_lr)
    # print('*****************')

    optimizer = torch.optim.AdamW(params=model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler = get_scheduler(args.lr_scheduler, optimizer, args.epochs * iters_per_epoch, args.lr_power,
    #                           iters_per_epoch * args.lr_warmup, args.lr_warmup_ratio)

    scheduler = create_lr_scheduler(optimizer, len(trainloader), args.epochs, warmup=True)

    loss_scaler = NativeScaler()
    # scaler = GradScaler(enabled=args.amp) if torch.cuda.is_bf16_supported() else None

    # writer = SummaryWriter(str(args.save_dir / 'logs'))

    if args.resume:
        checkpoint_save_path = f'./save_weights/{args.model}_best_model_{best_mIoU}.pth'
        checkpoint = torch.load(checkpoint_save_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state'])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_mIoU = checkpoint['best_mIoU']
        best_F1 = checkpoint['F1_Score']
        best_ACC = checkpoint['Acc']
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        print(f'The Best MeanIou is {best_mIoU:.4f}')
        print(f'The Best best_F1 is {best_F1}')
        print(f'The Best best_ACC is {best_ACC}')

    for epoch in range(args.epochs):

        if args.distributed:
            trainloader.sampler.set_epoch(epoch)

        mean_loss, lr = train_one_epoch(model, optimizer, trainloader,
                                        epoch, device, args.train_print_freq, args.clip_grad, args.clip_mode,
                                        loss_scaler, args)

        confmat, metric = evaluate(args, model, valloader, device, args.val_print_freq)
        # mae_info, f1_info = mae_metric.compute(), f1_metric.compute()
        all_f1, mean_f1 = metric.compute_f1()
        all_acc, mean_acc = metric.compute_pixel_acc()
        print(f"[epoch: {epoch}] val_meanF1: {mean_f1}\nval_meanACC: {mean_acc}")

        scheduler.step()

        val_info = f'{str(confmat)}\nval_meanF1: {mean_f1}\nval_meanACC: {mean_acc}'

        print(val_info)

        if dist.is_main_process():
            with open(results_file, "a") as f:
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n"
                f.write(train_info + val_info + "\n\n")

        with open(results_file, 'r') as file:
            text = file.read()
        match = re.search(r'mean IoU:\s+(\d+\.\d+)', text)
        if match:
            mean_iou = float(match.group(1))

        if mean_f1 > best_F1:
            best_F1 = mean_f1
            checkpoint_save = {
                "model_state": model_without_ddp.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_mIoU": mean_iou,
                "F1_Score": mean_f1,
                "Acc": mean_acc,
                # "MAE": round(mae_info, 4),
                "scaler": loss_scaler.state_dict()
            }
            torch.save(checkpoint_save, f'{args.save_weights_dir}/{args.model}_best_model_{best_mIoU}.pth')

    # writer.close()
    # end = time.gmtime(time.time() - start)

        TITLE = 'Validation Results'
        TABLE_DATA = (
            ('Mean Pixel Acc', 'Mean Iou', 'Mean F1 Score'),
            ('{:.2f}'.format(mean_acc),
             '{:.2f}'.format(mean_iou),
             '{:.2f}'.format(mean_f1),
             ),
        )
        table_instance = AsciiTable(TABLE_DATA, TITLE)
        # table_instance.justify_columns[2] = 'right'
        print()
        print(table_instance.table)
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Pytorch MSCA-EfficientFormer Model training and evaluation script', parents=[get_argparser()])
    args = parser.parse_args()
    fix_seeds(2024)
    setup_cudnn()
    # gpu = setup_ddp()
    main(args)
    # cleanup_ddp()