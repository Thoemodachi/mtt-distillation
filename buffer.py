import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset, get_network, get_daparam,\
    TensorDataset, epoch, ParamDiffAug
import copy
import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("buffer_training.log"),
        logging.StreamHandler()
    ]
)

def main(args):
    logging.info("Starting main procedure...")

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    logging.info(f"Using device: {args.device}")
    logging.info("Loading dataset...")

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.subset, args=args
    )

    logging.info(f"Hyper-parameters: {args.__dict__}")

    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset in ["LFW", "CelebA"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.model)
    os.makedirs(save_dir, exist_ok=True)

    logging.info("Organising real dataset...")
    images_all, labels_all = [], []
    indices_class = [[] for _ in range(num_classes)]
    logging.info("Building dataset (loading samples)...")

    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        logging.info(f"Class {c}: {len(indices_class[c])} real images")

    for ch in range(channel):
        mean_val = torch.mean(images_all[:, ch])
        std_val = torch.std(images_all[:, ch])
        logging.info(f"Channel {ch}: mean={mean_val:.4f}, std={std_val:.4f}")

    criterion = nn.CrossEntropyLoss().to(args.device)
    trajectories = []

    dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)
    args.dc_aug_param['strategy'] = 'crop_scale_rotate'
    logging.info(f"DC augmentation parameters: {args.dc_aug_param}")

    for it in range(0, args.num_experts):
        logging.info(f"Training expert {it+1}/{args.num_experts}...")

        teacher_net = get_network(args.model, channel, num_classes, im_size).to(args.device)
        teacher_net.train()
        lr = args.lr_teacher
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
        teacher_optim.zero_grad()

        timestamps = [[p.detach().cpu() for p in teacher_net.parameters()]]
        lr_schedule = [args.train_epochs // 2 + 1]

        for e in range(args.train_epochs):
            train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher_net,
                                          optimizer=teacher_optim, criterion=criterion, args=args, aug=True)
            test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher_net,
                                        optimizer=None, criterion=criterion, args=args, aug=False)

            logging.info(f"Expert {it} | Epoch {e} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

            if e in lr_schedule and args.decay:
                lr *= 0.1
                logging.info(f"Learning rate decayed to {lr}")
                teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
                teacher_optim.zero_grad()

        trajectories.append(timestamps)

        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, f"replay_buffer_{n}.pt")):
                n += 1
            save_path = os.path.join(save_dir, f"replay_buffer_{n}.pt")
            logging.info(f"Saving buffer to {save_path}")
            torch.save(trajectories, save_path)
            trajectories = []

    logging.info("Main procedure complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CelebA', help='dataset')
    parser.add_argument('--subset', type=str, default=None, help='subset')
    parser.add_argument('--model', type=str, default='VGGFace', help='model')
    parser.add_argument('--res', type=int, default=128, help='resolution')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='datasets', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=10)


    args = parser.parse_args()
    main(args)


