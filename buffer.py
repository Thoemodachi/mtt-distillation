import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import warnings

from utils import get_dataset, get_network, get_daparam,\
    TensorDataset, epoch, ParamDiffAug

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()    
    
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    criterion = nn.CrossEntropyLoss().to(args.device)  # For FaceNet, VGGFace with classifier
    print('Hyper-parameters: \n', args.__dict__)

    save_dir = os.path.join(args.buffer_path, args.dataset)

    # If using LFW or CelebA or MS-Celeb-1M, add dataset-specific subfolders
    if args.dataset in ["LFW", "CelebA", "MS-Celeb-1M"]:
        save_dir = os.path.join(save_dir, args.dataset)

    # Optionally include image size, to distinguish settings (e.g., 160 for FaceNet)
    if hasattr(args, "im_size"):
        res = args.im_size if isinstance(args.im_size, int) else args.im_size[0]
        save_dir = os.path.join(save_dir, f"{res}px")

    # Add the model name
    save_dir = os.path.join(save_dir, args.model)

    # Ensure folder exists
    os.makedirs(save_dir, exist_ok=True)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    criterion = nn.CrossEntropyLoss().to(args.device)

    trajectories = []

    dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    ''' set augmentation for whole-dataset training '''
    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)
    args.dc_aug_param['strategy'] = 'crop_scale_rotate'  # for whole-dataset training
    print('DC augmentation parameters: \n', args.dc_aug_param)

    # Train expert models
    for it in range(args.num_experts):
        teacher = get_network(args.model, channel, num_classes, im_size).to(args.device)
        teacher.train()
        optimizer = torch.optim.SGD(teacher.parameters(), lr=args.lr_teacher, momentum=args.mom, weight_decay=args.l2)
        timestamps = []
        timestamps.append([p.detach().cpu() for p in teacher.parameters()])
        lr_schedule = [args.train_epochs // 2 + 1]

        for e in range(args.train_epochs):

            train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher, optimizer=teacher_optim,
                                        criterion=criterion, args=args, aug=True)

            test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher, optimizer=None,
                                        criterion=criterion, args=args, aug=False)

            print("Itr: {}\tEpoch: {}\tTrain Acc: {}\tTest Acc: {}".format(it, e, train_acc, test_acc))

            timestamps.append([p.detach().cpu() for p in teacher.parameters()])

            if e in lr_schedule and args.decay:
                lr *= 0.1
                teacher_optim = torch.optim.SGD(teacher.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
                teacher_optim.zero_grad()

        trajectories.append(timestamps)

        # Save buffer
        if len(trajectories) >= args.save_interval:
            idx = len(os.listdir(save_dir))
            fname = f"replay_buffer_{idx}.pt"
            torch.save(trajectories, os.path.join(save_dir, fname))
            print("Saved expert buffer:", fname)
            trajectories = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CelebA', help='dataset')
    parser.add_argument('--subset', type=str, default=None, help='subset')
    parser.add_argument('--model', type=str, default='VGGFace', help='model')
    parser.add_argument('--res', type=int, default=128, help='resolution for model images')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='datasets', help='dataset root folder for CelebA')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--batch_real', type=int, default=256)
    parser.add_argument('--batch_train', type=int, default=128)
    parser.add_argument('--im_size', type=int, default=160)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--num_classes', type=int, required=True)

    args = parser.parse_args()
    main(args)