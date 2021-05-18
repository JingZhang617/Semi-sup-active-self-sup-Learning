import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os, argparse
from datetime import datetime

from model.ResNet_models import Generator, FCDiscriminator
from data import get_loader
from utils import adjust_lr, AvgMeter,label_edge_prediction,pred_edge_prediction
from scipy import misc
import cv2
import torchvision.transforms as transforms
from utils import l2_regularisation

os.environ["CUDA_VISIBLE_DEVICES"] = '5'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=5e-5, help='learning rate')
parser.add_argument('--lr_dis', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=24, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')
parser.add_argument('-beta1_gen', type=float, default=0.5,help='beta of Adam for generator')
parser.add_argument('-beta1_dis', type=float, default=0.9,help='beta of Adam for discriminator')
parser.add_argument('-weight_decay', type=float, default=0.0005,help='Regularisation parameter for L2-loss')

opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
print('Discriminator Learning Rate: {}'.format(opt.lr_dis))
# build models
generator = Generator(channel=64)
discriminator = FCDiscriminator()
generator.cuda()
discriminator.cuda()

generator_optimizer = torch.optim.Adam(generator.parameters(), opt.lr_gen, betas=[opt.beta1_gen, 0.999])
generator_optimizer.zero_grad()
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), opt.lr_dis, betas=[opt.beta1_dis, 0.999])
discriminator_optimizer.zero_grad() 


labeled_image_root = '/home1/bowen/al_semi/al/img/'
labeled_gt_root = '/home1/bowen/al_semi/al/gt/'
labeled_train_loader = get_loader(labeled_image_root, labeled_gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
labeled_total_step = len(labeled_train_loader)

unlabeled_image_root = '/home1/bowen/al_semi/DUTS-TR/DUTS-TR-Image/'
unlabeled_gt_root = '/home1/bowen/al_semi/DUTS-TR/DUTS-TR-Mask/'
unlabeled_train_loader = get_loader(unlabeled_image_root, unlabeled_gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
unlabeled_total_step = len(unlabeled_train_loader)

labeled_data_iter = enumerate(labeled_train_loader)

CE = torch.nn.BCELoss()
size_rates = [0.75, 1, 1.25]  # multi-scale training


def edgepred_loss(pred, gt, size_average=True):

    gt = torch.sigmoid(gt)
    edge = pred_edge_prediction(torch.sigmoid(pred))
    edge_loss = CE(edge, gt)

    return edge_loss, edge
    
def process_gt(gts):
    for kk in range(gts.shape[0]):
        cur_gt = gts[kk,:,:,:]

        if cur_gt.max() == 0:
            cur_gt = cur_gt + 0.0005

        gts[kk,:,:,:] = cur_gt

    return gts

def laplacian_edge(img):
    laplacian_filter = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filter = torch.reshape(laplacian_filter, [1, 1, 3, 3])
    filter = filter.cuda()
    lap_edge = F.conv2d(img, filter, stride=1, padding=1)
    return lap_edge

def gradient_x(img):
    sobel = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter = torch.reshape(sobel,[1,1,3,3])
    filter = filter.cuda()
    gx = F.conv2d(img, filter, stride=1, padding=1)
    return gx


def gradient_y(img):
    sobel = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter = torch.reshape(sobel, [1, 1,3,3])
    filter = filter.cuda()
    gy = F.conv2d(img, filter, stride=1, padding=1)
    return gy

def charbonnier_penalty(s):
    cp_s = torch.pow(torch.pow(s, 2) + 0.001**2, 0.5)
    return cp_s

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def smoothness_loss(pred, gt, size_average=True):
    alpha = 10
    s1 = 10
    s2 = 1

    ## first oder derivative: sobel
    sal_x = torch.abs(gradient_x(pred))
    sal_y = torch.abs(gradient_y(pred))
    sal_xy=torch.tanh(torch.pow(torch.pow(sal_x, 2) + torch.pow(sal_y, 2) + 0.00001, 0.5))
    w1 = torch.exp(torch.abs(gt) * (-alpha))
    cps_xy = charbonnier_penalty(sal_xy * w1 )

    ## second order derivative: laplacian
    lap_sal = torch.abs(laplacian_edge(pred))
    weight_lap = torch.exp(torch.abs(gt) * (-alpha))
    weighted_lap = charbonnier_penalty(lap_sal*weight_lap)

    smooth_loss = s1*torch.mean(cps_xy) + s2*torch.mean(weighted_lap)

    return smooth_loss

def edge_loss(edge_map, gt, size_average=True):

    edge = torch.sigmoid(edge_map)
    lap_gt = torch.relu(torch.tanh(laplacian_edge(gt)))

    edge_loss = CE(edge,lap_gt)

    return edge_loss, lap_gt


def visualize_prediction(pred):

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_pred.png'.format(kk)
        misc.imsave(save_path + name, pred_edge_kk)

def visualize_unlabeled_prediction(pred):

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_unpred.png'.format(kk)
        misc.imsave(save_path + name, pred_edge_kk)

def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        misc.imsave(save_path + name, pred_edge_kk)

def visualize_unlabeled_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_ungt.png'.format(kk)
        misc.imsave(save_path + name, pred_edge_kk)

def visualize_confidence_map(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_unconf.png'.format(kk)
        misc.imsave(save_path + name, pred_edge_kk)

def visualize_confidence_map1(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_conf.png'.format(kk)
        misc.imsave(save_path + name, pred_edge_kk)

def visualize_original_img(rec_img):
    img_transform = transforms.Compose([
        transforms.Normalize(mean = [-0.4850/.229, -0.456/0.224, -0.406/0.225], std =[1/0.229, 1/0.224, 1/0.225])])
    for kk in range(rec_img.shape[0]):
        current_img = rec_img[kk,:,:,:]
        current_img = img_transform(current_img)
        current_img = current_img.detach().cpu().numpy().squeeze()
        current_img = current_img * 255
        current_img = current_img.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_img.png'.format(kk)
        current_img = current_img.transpose((1,2,0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(save_path+name, new_img)

def visualize_unlabeled_img(rec_img):
    img_transform = transforms.Compose([
        transforms.Normalize(mean = [-0.4850/.229, -0.456/0.224, -0.406/0.225], std =[1/0.229, 1/0.224, 1/0.225])])
    for kk in range(rec_img.shape[0]):
        current_img = rec_img[kk,:,:,:]
        current_img = img_transform(current_img)
        current_img = current_img.detach().cpu().numpy().squeeze()
        current_img = current_img * 255
        current_img = current_img.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_unimg.png'.format(kk)
        current_img = current_img.transpose((1,2,0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(save_path+name, new_img)

def visualize_edge(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_edge.png'.format(kk)
        misc.imsave(save_path + name, pred_edge_kk)

def visualize_edge_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_edgegt.png'.format(kk)
        misc.imsave(save_path + name, pred_edge_kk)

def make_Dis_label(label,gts):
    D_label = np.ones(gts.shape)*label
    D_label = Variable(torch.FloatTensor(D_label)).cuda()

    return D_label

def adjust_maskt(init, final, epoch, epoch_all):
    
    maskt = init - (init - final) * (epoch / epoch_all)

    return maskt

# labels for adversarial training
pred_label = 0
gt_label = 1
dis_steps = 5

lambda_semi = 0.1
lambda_adv_pred = 0.1
lambda_smooth = 0.1
lambda_semi_adv=0.01

## main parameters to tune
mask_t = 0.2

unlabeled_steps = 10 ## train one batch of labeled data, and test on ten batch of unlabeled data

semi_start_adv = 0
semi_start = 1
############################
print("Let's go!")
for epoch in range(opt.epoch):
    # adjustG_lr(generator_optimizer, epoch)
    # adjustD_lr(discriminator_optimizer, epoch)
    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
    adjust_lr(discriminator_optimizer, opt.lr_dis, epoch, opt.decay_rate, opt.decay_epoch)
    #maskt = adjust_maskt(0.6, 0.2, epoch, opt.epoch)
    # print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    # print('Discriminator Learning Rate: {}'.format(discriminator_optimizer.param_groups[0]['lr']))
    loss_record = AvgMeter()
    for i, labeled_pack in enumerate(labeled_train_loader, start=1):
        generator_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()
        generator.train()
        discriminator.train()

        loss_seg_value = 0
        loss_adv_pred_value = 0
        loss_D_value = 0
        loss_semi_value = 0
        loss_semi_adv_value = 0
        loss_edge_value = 0

        for param in discriminator.parameters():
            param.requires_grad = False

        ## process unlabeled data
        unlabeled_data_iter = enumerate(unlabeled_train_loader)

        if epoch > semi_start_adv:
            for kk in range(unlabeled_steps):
                _, unlabeled_pack = unlabeled_data_iter.next()
                unlabeled_images, unlabeled_gt = unlabeled_pack
                unlabeled_images = Variable(unlabeled_images)
                unlabeled_images = unlabeled_images.cuda()
                unlabeled_gt = Variable(unlabeled_gt)
                unlabeled_gt = unlabeled_gt.cuda()

                # print(unlabeled_images.size())
                ## G1: obtain prediction for unlabeled data
                unlabeled_generator_pred, unlabeled_edge_pred = generator(unlabeled_images)
                unlabeled_edge_pred= unlabeled_edge_pred.detach()
                loss_smoothness, _ = edgepred_loss(torch.sigmoid(unlabeled_generator_pred), unlabeled_edge_pred)

                unlabeled_pred = unlabeled_generator_pred
                unlabeled_pred = unlabeled_pred.detach()
                
                ## D1: obtain prediction from discrininator
                dis_out = discriminator(torch.cat([unlabeled_images, torch.sigmoid(unlabeled_generator_pred)],1))
                dis_out_sigmoid = torch.sigmoid(dis_out)
                loss_semi_adv = lambda_semi_adv * CE(torch.sigmoid(dis_out),make_Dis_label(gt_label, unlabeled_gt))
                loss_semi_adv_value += loss_semi_adv / lambda_semi_adv


                for param in discriminator.parameters():
                    param.requires_grad = False

                if epoch < semi_start:
                    #loss_semi_adv.backward(retain_graph=True)
                    loss_semi_adv.backward()
                    loss_semi_value = 0
                    print('{} Epoch [{:03d}/{:03d}],  Unlabeled_Step [{:01d}/{:01d}], Semi_adv Loss: {:.4f}'.
                              format(datetime.now(), epoch, opt.epoch, kk, unlabeled_steps, loss_semi_adv_value.data))
                else:
                    semi_mask = (pred_edge_prediction(dis_out_sigmoid) > mask_t)
                    semi_mask = 1-semi_mask.float()
                    semi_mask1 = semi_mask.cpu().numpy()
                    semi_gt = torch.sigmoid(unlabeled_generator_pred)
                    img_size = semi_mask1.shape[2] * semi_mask1.shape[3] * semi_mask1.shape[0]
                    #print(semi_mask.size)
                    #print(img_size)
                    semi_ratio = float(img_size)/(torch.sum(semi_mask)+1e-8)
                    semi_ratio_value = float(img_size)/(float(semi_mask1.sum())+1e-8)
                    #print('2')
                    print('semi ratio: {:.4f}'.format(semi_ratio.data))
                    if semi_ratio_value == 0.0:
                        loss_semi_value += 0
                    else:
                        target_cur = (semi_gt*semi_mask)
                        target_cur = target_cur.detach()
                        #loss_semi = (lambda_semi/semi_ratio) * CE(torch.sigmoid(unlabeled_generator_pred)*semi_mask, target_cur)
                        loss_semi = (lambda_semi*semi_ratio) * CE(unlabeled_generator_pred*semi_mask, target_cur)
                        loss_semi_value += loss_semi / lambda_semi
                        loss =loss_semi + loss_semi_adv + lambda_smooth * loss_smoothness
                        loss.backward()
                        #loss_semi.backward(retain_graph=True)
                        print('{} Epoch [{:03d}/{:03d}], Unlabeled_Step [{:01d}/{:01d}], Semi Loss: {:.4f}, Smoothness Loss: {:.4f}, Semi_adv Loss: {:.4f}'.
                                  format(datetime.now(), epoch, opt.epoch, kk, unlabeled_steps, loss_semi_value.data, loss_smoothness.data, loss_semi_adv_value.data))

                visualize_unlabeled_prediction(torch.sigmoid(unlabeled_generator_pred))
                visualize_unlabeled_gt(unlabeled_gt)
                visualize_unlabeled_img(unlabeled_images)
                visualize_confidence_map(torch.sigmoid(dis_out))

        else:
            loss_semi = None
            loss_semi_adv = None

        ## process labeled data
        #for rate in size_rates:
        labeled_images, gts, edges = labeled_pack
        labeled_images = Variable(labeled_images)
        gts = Variable(gts)
        #edges = Variable(edges)
        labeled_images = labeled_images.cuda()
        gts = gts.cuda()
        #gts = process_gt(gts)
        #edges = edges.cuda()

            # multi-scale training samples
            #trainsize = int(round(opt.trainsize * rate / 32) * 32)
            #if rate != 1:
                #labeled_images = F.upsample(labeled_images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                #gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

        generator_pred, edge_pred = generator(labeled_images)
        loss_gen = structure_loss(generator_pred, gts)

            #if rate == 1:
            #    loss_record.update(loss_gen.data, opt.batchsize)
        
        Dis_out = discriminator(torch.cat([labeled_images, torch.sigmoid(generator_pred)],1))
        loss_adv_pred = CE(torch.sigmoid(Dis_out), make_Dis_label(gt_label, gts))
        loss_edge, edge_gt = edge_loss(edge_pred, gts)
        loss = loss_gen +  0.5 * loss_edge + lambda_adv_pred * loss_adv_pred
        loss.backward()
        loss_seg_value += loss_gen
        loss_edge_value += loss_edge
        loss_adv_pred_value += loss_adv_pred / lambda_adv_pred

        ## D0: train discriminator with labeled images
        # bring back requires_grad
        for param in discriminator.parameters():
            param.requires_grad = True

        ## train discriminator
        generator_pred = generator_pred.detach()
        Dis_out1 = discriminator(torch.cat([labeled_images, torch.sigmoid(generator_pred)],1))
        loss_D1 = CE(torch.sigmoid(Dis_out1), make_Dis_label(pred_label, generator_pred))
        loss_D1 = loss_D1 / 2
  
        Dis_out2 = discriminator(torch.cat([labeled_images, gts],1))
            #D_out2 = discriminator(gts)
        loss_D2 = CE(torch.sigmoid(Dis_out2), make_Dis_label(gt_label, gts))
        loss_D2 = loss_D2 / 2

        loss_D = loss_D1 + loss_D2
        loss_D.backward()
        loss_D_value += loss_D

        generator_optimizer.step()
        discriminator_optimizer.step()

        visualize_prediction(torch.sigmoid(generator_pred))
        visualize_gt(gts)
        visualize_original_img(labeled_images)
        visualize_confidence_map1(Dis_out1)
        visualize_edge(torch.sigmoid(edge_pred))
        visualize_edge_gt(edge_gt)

        if i % 10 == 0 or i == labeled_total_step:
            #print('0')
            print('{} Epoch [{:03d}/{:03d}], Labeled_Step [{:04d}/{:04d}], Gen_seg Loss: {:.4f},Gen_edge Loss: {:.4f}, Gen_adv Loss: {:.4f}, Dis Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, labeled_total_step, loss_seg_value.data, loss_edge_value.data,loss_adv_pred_value.data, loss_D_value.data))

    save_path = '/home1/bowen/models/gan_semi/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if (epoch+1) % 30 == 0:
        torch.save(generator.state_dict(), save_path + 'Modeltexture' + '_%d' % epoch + '_gen.pth')
        torch.save(discriminator.state_dict(), save_path + 'Modeltexture' + '_%d' % epoch + '_dis.pth')
