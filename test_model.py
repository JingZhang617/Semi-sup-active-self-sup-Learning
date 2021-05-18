import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from scipy import misc

from model.ResNet_models import Generator
from data import test_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
opt = parser.parse_args()

dataset_path = '/home1/bowen/dataset/IMG/'

generator = Generator(channel=64)
generator.load_state_dict(torch.load('/home1/bowen/models/gan_semi/Modeltexture_29_gen.pth'))

generator.cuda()
generator.eval()

test_datasets = ['ECSSD','DUT','DUTS','PASCAL','THUR','HKU-IS']

for dataset in test_datasets:
    
    save_path = '/home1/bowen/results/ResNet50/' + dataset + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + dataset + '/'
    #print(image_root)

    test_loader = test_dataset(image_root, opt.testsize)
    #print(test_loader.size)
    for i in range(test_loader.size):
        print i
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        generator_pred ,_ = generator.forward(image)
        res =torch.sigmoid(generator_pred)
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        misc.imsave(save_path+name, res)
