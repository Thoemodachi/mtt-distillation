import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import warnings

from utils import get_face_loader, get_network, TensorDataset

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load real face data
    train_loader = get_face_loader(args.train_dir, image_size=args.im_size, batch_size=args.batch_real, shuffle=True)
    test_loader = get_face_loader(args.test_dir, image_size=args.im_size, batch_size=args.batch_real, shuffle=False)
    channel = 3
    im_size = (args.im_size, args.im_size)
    num_classes = args.num_classes

    # Prepare real data snapshot arrays
    images_all, labels_all = [], []
    for imgs, labs in tqdm(train_loader, desc="Collecting real images"):
        images_all.append(imgs)
        labels_all.append(labs)
    images_all = torch.cat(images_all, dim=0).cpu()
    labels_all = torch.cat(labels_all, dim=0).cpu()

    dst_train = TensorDataset(copy.deepcopy(images_all), copy.deepcopy(labels_all))
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(args.device)  # For FaceNet, VGGFace with classifier

    # Create save folder
    save_dir = os.path.join(args.buffer_path, args.model)
    os.makedirs(save_dir, exist_ok=True)

    trajectories = []

    # Train expert models
    for it in range(args.num_experts):
        teacher = get_network(args.model, channel, num_classes, im_size).to(args.device)
        teacher.train()
        optimizer = torch.optim.SGD(teacher.parameters(), lr=args.lr_teacher, momentum=args.mom, weight_decay=args.l2)

        timestamps = [[p.detach().cpu() for p in teacher.parameters()]]
        lr_schedule = [args.train_epochs // 2]

        for e in range(args.train_epochs):
            teacher.train()
            for imgs, labs in trainloader:
                imgs, labs = imgs.to(args.device), labs.to(args.device)
                optimizer.zero_grad()
                if args.model == 'ArcFace':
                    out = teacher(imgs, labs)  # ArcFace requires both inputs and labels
                else:
                    out = teacher(imgs)
                loss = criterion(out, labs)
                loss.backward()
                optimizer.step()


            teacher.eval()
            timestamps.append([p.detach().cpu() for p in teacher.parameters()])

            if e in lr_schedule:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1

            print(f"[{it}/{args.num_experts}] Epoch {e}: recorded trajectory.")

        trajectories.append(timestamps)

        # Save buffer
        if len(trajectories) >= args.save_interval:
            idx = len(os.listdir(save_dir))
            fname = f"replay_buffer_{idx}.pt"
            torch.save(trajectories, os.path.join(save_dir, fname))
            print("Saved expert buffer:", fname)
            trajectories = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--model', choices=['FaceNet', 'VGGFace', 'ArcFace'], required=True)
    parser.add_argument('--buffer_path', required=True)
    parser.add_argument('--num_experts', type=int, default=10)
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--lr_teacher', type=float, default=0.01)
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--batch_real', type=int, default=256)
    parser.add_argument('--batch_train', type=int, default=128)
    parser.add_argument('--im_size', type=int, default=160)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--num_classes', type=int, required=True)

    args = parser.parse_args()
    main(args)