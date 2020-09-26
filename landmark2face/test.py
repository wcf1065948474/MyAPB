import os
from options.test_options import TestOptions
import cv2
import torchvision.transforms as transforms
from PIL import Image
from models.l2face_model import *
import numpy as np
from util.util import *
from APB.APBDataset import *
from APB.APBNet import *
import torch


def tuple_shape(shape):
    r_data = []
    for p in shape:
        r_data.append([p.x, p.y])
    return r_data


def drawCircle(img, shape, radius=1, color=(255, 255, 255), thickness=1):
    for p in shape:
        img = cv2.circle(img, (int(p[0]), int(p[1])), radius, color, thickness)
    return img


def vector2points(landmark):
    shape = []
    for i in range(len(landmark) // 2):
        shape.append([landmark[2 * i], landmark[2 * i + 1]])
    return shape


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.isTrain = False
    opt.name = 'man1'
    opt.model = 'l2face'
    opt.netG = 'resnet_9blocks_l2face'
    opt.dataset_mode = 'l2face'
    model = L2FaceModel(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()
    transforms_label = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # audio2landmark
    # audio_net = APBNet()
    # checkpoint = torch.load('APB/man1_best.pth')
    # audio_net.load_state_dict(checkpoint['net_G'])
    # audio_net.cuda()
    # audio_net.eval()
    # dataset
    feature_path = '../../AnnVI/feature'
    idt_name = 'man1'
    testset = APBDataset(feature_path, idt_name=idt_name, mode='test', img_size=256)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    out_path = 'result'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(os.path.join(out_path, '{}.avi'.format(idt_name)), fourcc, 25.0, (256 * 2, 256))

    for idx, data in enumerate(dataloader):
        # landmark=data['label'].cuda()

        print('\r{}/{}'.format(idx+1, len(dataloader)), end='')

        # lab_template = np.zeros((256, 256, 3)).astype(np.uint8)
        # lab = drawCircle(lab_template.copy(), vector2points(landmark), radius=1, color=(255, 255, 255), thickness=4)
        # lab = Image.fromarray(lab).convert('RGB')
        # lab = transforms_label(lab).unsqueeze(0)

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        save_many_images_from_dict(visuals,'result/{}.jpg'.format(idx))

