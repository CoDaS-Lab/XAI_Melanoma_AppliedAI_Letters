import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.utils as utils
import torchvision.transforms as torch_transforms
from networks1 import AttnVGG, VGG
from loss import FocalLoss
from data import preprocess_data_2016, preprocess_data_2017, ISIC
from utilities import *
from transforms import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]
parser = argparse.ArgumentParser()
parser.add_argument("--preprocess", action='store_true', help="run preprocess_data")
parser.add_argument("--dataset", type=str, default="ISIC2017", help='ISIC2017 / ISIC2016')
parser.add_argument("--outf", type=str, default="logs_extract", help='path of log files')
parser.add_argument("--base_up_factor", type=int, default=8, help="number of epochs")
parser.add_argument("--normalize_attn", action='store_true', help='if True, attention map is normalized by softmax; otherwise use sigmoid')
parser.add_argument("--no_attention", action='store_true', help='turn off attention')
parser.add_argument("--log_images", action='store_true', help='visualze images in Tensoeboard')
# Reference: https://stackoverflow.com/questions/38834378/path-to-a-directory-as-argparse-argument
def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)
parser.add_argument("--checkpoint_path", type=file_path, help='path to checkpoints')
parser.add_argument("--data_type", type=str, default="test", help='train/test/val')
opt = parser.parse_args()

opt.dataset == "ISIC2016"
opt.data_type == "test"
#opt.checkpoint_path == 'C:\Users\harsh\Desktop\HBK008\CoDaS Lab\Project XAI\data_2016\checkpoint.pth'
opt.preprocess

def main():
    # load data
    print('\nloading the dataset ...')
    assert opt.dataset == "ISIC2016" or opt.dataset == "ISIC2017"
    if opt.dataset == "ISIC2016":
        normalize = Normalize((0.7012, 0.5517, 0.4875), (0.0942, 0.1331, 0.1521))
    elif opt.dataset == "ISIC2017":
        normalize = Normalize((0.6820, 0.5312, 0.4736), (0.0840, 0.1140, 0.1282))
    transform_extract = torch_transforms.Compose([
         RatioCenterCrop(0.8),
         Resize((256,256)),
         CenterCrop((224,224)),
         ToTensor(),
         normalize
    ])
    extractset = ISIC(csv_file=f'{opt.data_type}.csv', transform=transform_extract)
    extractloader = torch.utils.data.DataLoader(extractset, batch_size=64, shuffle=False, num_workers=8)
    print('done\n')
    # load network
    print('\nloading the model ...')
    if not opt.no_attention:
        print('turn on attention ...')
        if opt.normalize_attn:
            print('use softmax for attention map ...')
        else:
            print('use sigmoid for attention map ...')
    else:
        print('turn off attention ...')
    net = AttnVGG(num_classes=2, attention=not opt.no_attention, normalize_attn=opt.normalize_attn)
    # net = VGG(num_classes=2, gap=False)
    #checkpoint = torch.load(opt.checkpoint_path)
    checkpoint = torch.load('checkpoint.pth')
    net.load_state_dict(checkpoint['state_dict'])
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    model.eval()
    print('done\n')
    # extracting
    print('\nstart extracting ...\n')
    writer = SummaryWriter(opt.outf)
    total = 0
    correct = 0
    predictionsF = []
    pred_extractF = [] 
    a1F= [] 
    a2F= []
    FLF = []
    labels_extractF = []
    images_extractF=[]
    DNNProbsF = []
   
    with torch.no_grad():
        with open('extract_results.csv', 'wt', newline='') as csv_file:
         with open(f'{opt.data_type}_results.csv', 'wt', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for i, data in enumerate(extractloader, 0):
                images_extract, labels_extract = data['image'], data['label']
                images_extract, labels_extract = images_extract.to(device), labels_extract.to(device)
                #pred_extract,FL, __, __, = model(images_extract)
                pred_extract, FL, a1, a2 = model(images_extract)
                #FL = list(FL)
                predict = torch.argmax(pred_extract, 1)
                total += labels_extract.size(0)
                correct += torch.eq(predict, labels_extract).sum().double().item()
                # record extract predicted responses
                responses = F.softmax(pred_extract, dim=1).squeeze().cpu().numpy()
                responses = [responses[i] for i in range(responses.shape[0])]
                csv_writer.writerows(responses)
                
                               
                # log images
                if opt.log_images:
                    I_extract = utils.make_grid(images_extract, nrow=8, normalize=True, scale_each=True)
                    writer.add_image('extract/image', I_extract, i)
                    # accention maps
                    if not opt.no_attention:
                        __,__, a1, a2 = model(images_extract)
                        if a1 is not None:
                            attn1 = visualize_attn(I_extract, a1, up_factor=opt.base_up_factor, nrow=8)
                            writer.add_image('extract/attention_map_1', attn1, i)
                        if a2 is not None:
                            attn2 = visualize_attn(I_extract, a2, up_factor=2*opt.base_up_factor, nrow=8)
                            writer.add_image('extract/attention_map_2', attn2, i)
                #FF = torch.stack(FL, dim=0)
                images_extractF+=[images_extract]
                pred_extractF += [pred_extract]
                predictionsF += [predict]
                FLF += [FL]
                a1F += [a1]
                a2F += [a2]
                labels_extractF += labels_extract
                

    AP, AUC, precision_mean, precision_mel, recall_mean, recall_mel = compute_metrics(f'{opt.data_type}_results.csv', f'{opt.data_type}.csv')
    print("\nextract result: accuracy %.2f%%" % (100*correct/total))
    print("\nmean precision %.2f%% mean recall %.2f%% \nprecision for mel %.2f%% recall for mel %.2f%%" %
            (100*precision_mean, 100*recall_mean, 100*precision_mel, 100*recall_mel))
    print("\nAP %.4f AUC %.4f\n" % (AP, AUC))

    # save gpu tensors
    torch.save({
        'a1': a1F,
        'a2': a2F,
        'extract': pred_extractF,
        'FinalLayer': FLF,
        'Labels': labels_extractF,
        'Images': images_extractF,
        'predictions': predictionsF,
        
    }, 'extract_test_final_new.pt')


if __name__ == "__main__":
    if opt.preprocess:
        if opt.dataset == "ISIC2016":
            preprocess_data_2016(root_dir='../data_2016')
        elif opt.dataset == "ISIC2017":
            preprocess_data_2017(root_dir='../data_2017', seg_dir='Train_Lesion')
    main()
