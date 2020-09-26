import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import numpy as np


# class L2FaceDataset(BaseDataset):
#     def __init__(self, opt):
#         BaseDataset.__init__(self, opt)
#         img_size = opt.img_size
#         root = '../../AnnVI/feature/{}'.format(opt.name.split('_')[0])
#         image_dir = '{}/{}_image_crop'.format(root, img_size)
#         label_dir = '{}/{}_landmark_crop_thin'.format(root, img_size)
#         # label_dir = '{}/512_landmark_crop'.format(root)
#         self.labels = []

#         imgs = os.listdir(image_dir)
#         # if 'man' in opt.name:
#         #     imgs.sort(key=lambda x:int(x.split('.')[0]))
#         # else:
#         #     imgs.sort(key=lambda x: (int(x.split('.')[0].split('-')[0]), int(x.split('.')[0].split('-')[1])))
#         for img in imgs:
#             img_path = os.path.join(image_dir, img)
#             lab_path = os.path.join(label_dir, img)
#             if os.path.exists(lab_path):
#                 self.labels.append([img_path, lab_path])
#         # transforms.Resize([img_size, img_size], Image.BICUBIC),
#         self.transforms_image = transforms.Compose([transforms.ToTensor(),
#                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         # transforms.Resize([img_size, img_size], Image.BICUBIC),
#         self.transforms_label = transforms.Compose([transforms.ToTensor(),
#                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         self.shuffle()

#     def shuffle(self):
#         random.shuffle(self.labels)


#     def __getitem__(self, index):
#         img_path, lab_path = self.labels[index]
#         img = Image.open(img_path).convert('RGB')
#         lab = Image.open(lab_path).convert('RGB')
#         img = self.transforms_image(img)
#         lab = self.transforms_label(lab)

#         imgA_path, labA_path = random.sample(self.labels, 1)[0]
#         imgA = Image.open(imgA_path).convert('RGB')
#         imgA = self.transforms_image(imgA)


#         return {'A': imgA, 'A_label': lab, 'B': img, 'B_label': lab}

#     def __len__(self):
#         """Return the total number of images in the dataset."""
#         return len(self.labels)
class L2FaceDataset(BaseDataset):
    def __init__(self,opt):
        BaseDataset.__init__(self,opt)
        self.crop_len = opt.crop_len
        self.image_dir = "256_image_crop"
        self.label_dir = "256_landmark_crop_thin"
        filename = "256_train.t7"
        self.root_dir = "/content/AnnVI/feature/{}".format(opt.name)
        data = torch.load(os.path.join(self.root_dir,filename))
        self.img = data['img_paths']
        self.audio = data['audio_features']

        self.transformsfunc = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self,index):
        random_idx = random.randint(0,len(self.audio)-1)
        random_name = self.img[random_idx][0].split('/')[1]
        imageA = Image.open(os.path.join(self.root_dir,self.image_dir,random_name))
        imageA = self.transformsfunc(imageA)

        audio_list = []
        labelB_list = []
        imageB_list = []
        for i in range(self.crop_len):
            name = self.img[index+i][0].split('/')[1]
            audio = self.audio[index+i][None]
            imageB = Image.open(os.path.join(self.root_dir,self.image_dir,name))
            labelB = Image.open(os.path.join(self.root_dir,self.label_dir,name))
            imageB = self.transformsfunc(imageB)
            labelB = self.transformsfunc(labelB)
            audio_list.append(audio)
            labelB_list.append(labelB)
            imageB_list.append(imageB)
        imageB = torch.stack(imageB_list)
        labelB = torch.stack(labelB_list)
        audio  = np.stack(audio_list)
        return {'A': imageA, 'B': imageB, 'label': labelB, 'audio':audio}

    def __len__(self):
        return len(self.audio)-self.crop_len


if __name__ == '__main__':
    from options.train_options import TrainOptions
    opt = TrainOptions().parse()
    dataset = L2FaceDataset(opt)
    dataset_size = len(dataset)
    print(dataset_size)
    for i, data in enumerate(dataset):
        print(data)
