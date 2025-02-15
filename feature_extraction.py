import models.oursmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms as trans

import sys, argparse, os, glob, copy
import pandas as pd
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle
from models.resnet import resnet50



class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = trans.Compose([ trans.Resize(256),
                           trans.CenterCrop(224),
                           trans.ColorJitter(saturation=0.2),
                           trans.ToTensor(),
                           trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)
        img = self.transform(img)
        sample = {'input': img}

        return sample 


def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path)
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def compute_feats(args, bags_list, i_classifier, save_path=None, magnification='single'):
    i_classifier.eval()
    #print(i_classifier)
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    for i in range(0, num_bags):
        feats_list = []
        if magnification=='single' or magnification=='low':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.png')) + glob.glob(os.path.join(bags_list[i], '*.jpeg'))
        elif magnification=='high':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*'+os.sep+'*.png')) + glob.glob(os.path.join(bags_list[i], '*'+os.sep+'*.jpeg'))
            print()
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda() 
                feats, classes = i_classifier(patches) #1024-dimension
                feats = feats.cpu().numpy()

                feats_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            df = pd.DataFrame(feats_list)
            os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
            df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')


def main():
    parser = argparse.ArgumentParser(description='Compute features from ResNet or SimCLR')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size of dataloader')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--backbone', default='resnet50', type=str, help='backbone [resnet18]')
    parser.add_argument('--norm_layer', default='instance', type=str, help='Normalization layer [instance]')
    parser.add_argument('--magnification', default='single', type=str, help='Magnification to compute features.')
    parser.add_argument('--weights', default=None, type=str, help='Folder of the pretrained weights, simclr/runs/*')
    parser.add_argument('--weights_high', default=None, type=str, help='Folder of the pretrained weights of high magnification, FOLDER < `simclr/runs/[FOLDER]`')
    parser.add_argument('--weights_low', default=None, type=str, help='Folder of the pretrained weights of low magnification, FOLDER <`simclr/runs/[FOLDER]`')
    parser.add_argument('--dataset', default='TCGA-lung-single', type=str, help='Dataset folder name [TCGA-lung-single]')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)

    if args.norm_layer == 'instance':
        norm = nn.InstanceNorm2d
        pretrain = False
    elif args.norm_layer == 'batch':
        norm = nn.BatchNorm2d
        if args.weights == 'ImageNet':
            pretrain = True
        else:
            pretrain = False

    # if args.backbone == 'resnet18':
    #     resnet = models.resnet18(pretrained=pretrain, norm_layer=norm)
    #     num_feats = 512

    resnet = resnet50(num_classes=2)
    # load model weights
    model_weight_path= "./resnet50-pre.pth"
    pre_weights= torch.load(model_weight_path)
    pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
    model_dict = resnet.state_dict()
    model_dict.update(pre_dict)
    resnet.load_state_dict(pre_dict,strict=False)

    num_feats = 1024
    # for param in resnet.parameters():
    #     param.requires_grad = False
    resnet.fc = nn.Identity()


    if args.magnification == 'single':
        i_classifier = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()
        print("model:",i_classifier)

        if args.weights == 'ImageNet':
            if args.norm_layer == 'batch':
                print('Use ImageNet features.')
            else:
                print('Please use batch normalization for ImageNet feature')
        else:
            if args.weights is not None:
                weight_path = os.path.join('simclr', 'runs', args.weights, 'checkpoints', 'model-v1.pth')
            else:
                weight_path = glob.glob('simclr/runs/*/checkpoints/*.pth')[-1]
            print("weight_path:", weight_path)
            state_dict_weights = torch.load(weight_path)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.dataset, 'embedder.pth'))
            print('Use pretrained features.')

    bags_path = os.path.join('WSI', args.dataset, 'single', '*', '*')
    print("bags_path:",bags_path)
    feats_path = os.path.join('datasets', args.dataset)

    os.makedirs(feats_path, exist_ok=True)
    bags_list = glob.glob(bags_path)


    compute_feats(args, bags_list, i_classifier, feats_path, args.magnification)
    n_classes = glob.glob(os.path.join('datasets', args.dataset, '*' + os.path.sep))
    n_classes = sorted(n_classes)
    all_df = []
    for i, item in enumerate(n_classes):
        bag_csvs = glob.glob(os.path.join(item, '*.csv'))
        bag_df = pd.DataFrame(bag_csvs)
        bag_df['label'] = i
        bag_df.to_csv(os.path.join('datasets', args.dataset, item.split(os.path.sep)[2] + '.csv'), index=False)
        all_df.append(bag_df)
    bags_path = pd.concat(all_df, axis=0, ignore_index=True)
    bags_path = shuffle(bags_path)
    bags_path.to_csv(os.path.join('datasets', args.dataset, args.dataset + '.csv'), index=False)


if __name__ == '__main__':
    main()