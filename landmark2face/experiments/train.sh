#!/usr/bin bash
python train.py --name man1 --gpu_ids 0 --batch_size 16 --img_size 256 --model l2face --dataset_mode l2face --netG resnet_9blocks_l2face
