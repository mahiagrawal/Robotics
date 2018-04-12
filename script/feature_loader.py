# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 19:00:15 2017
feature extractor for inital databse storage
@author: Mahim
"""
import torch.nn as nn
import torch
import torchvision.models as models
import skimage.io
import skimage.transform
from torch.autograd import Variable as V
import glob
import os
import pickle
from sklearn.decomposition import IncrementalPCA

def down_scale(image):
    return skimage.transform.resize(image, (228, 228))

alexnet = models.alexnet(pretrained=True).cuda()

filelist=[]
room_labels = []
area_labels = []

layers = ['fc7', 'fc6', 'pool5', 'conv4']
layers_index = [19, 17, 12, 10]

features = {}

j = 0

rooms = os.listdir("../data/images")
for room in rooms:
    areas = os.listdir("../data/images/"+room)
    for area in areas:
        for file in glob.glob('../data/images/'+room+ '/' + area + '/*.jpg'):
            filelist.append(file)
            room_labels.append(room)
            area_labels.append(area)
            img = skimage.io.imread(file)
            x = V(torch.from_numpy(down_scale(img)).float().unsqueeze(0), volatile=True).cuda()
            if j==0:
                for l_idx in range(len(layers)):
                    i = layers_index[l_idx]
                    if(i<=13):
                        ftrs = nn.Sequential(*list(alexnet.features.children())[0:i])(x.transpose(1,3))
                        ftrs = ftrs.view(-1, ftrs.size(1)*ftrs.size(2)*ftrs.size(3))
                        features[layers[l_idx]] = ftrs
                    elif(i>13):
                        f = nn.Sequential(*list(alexnet.features.children())[0:13])(x.transpose(1,3))
                        f = f.view(x.size(0), 256 * 6 * 6)
                        ftrs = nn.Sequential(*list(alexnet.classifier.children())[0:i-13])(f)
                        features[layers[l_idx]] = ftrs
                j += 1
            else:
                for l_idx in range(len(layers)):
                    i = layers_index[l_idx]
                    if(i<=13):
                        ftrs = nn.Sequential(*list(alexnet.features.children())[0:i])(x.transpose(1,3))
                        ftrs = ftrs.view(-1, ftrs.size(1)*ftrs.size(2)*ftrs.size(3))
                        features[layers[l_idx]] = torch.cat((features[layers[l_idx]],ftrs.data),0)
                    elif(i>13):
                        f = nn.Sequential(*list(alexnet.features.children())[0:13])(x.transpose(1,3))
                        f = f.view(x.size(0), 256 * 6 * 6)
                        ftrs = nn.Sequential(*list(alexnet.classifier.children())[0:i-13])(f)
                        features[layers[l_idx]] = torch.cat((features[layers[l_idx]],ftrs.data),0)

for layer in layers:
    file_name = layer+'.pkl'
    file_path = os.path.join('../data', 'feats', file_name)
    pickle.dump(features[layer], open(file_path, 'wb'))
    
file_name = 'location.pkl'
file_path = os.path.join('../data', 'feats', file_name)
pickle.dump([room_labels, area_labels], open(file_path, 'wb'))
