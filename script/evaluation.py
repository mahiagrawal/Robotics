# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 23:05:27 2017

@author: Mahim
"""

down vote
The documentation provides an example (about three quarters of the way down the page):

import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torchvision.models as models
import skimage.io
import skimage.transform
from torch.autograd import Variable as V
import glob
import pickle
import os
import numpy as np
from sklearn.neighbors import BallTree
from PIL import Image
from scipy.stats import entropy
from numpy.linalg import norm
from scipy.stats import pearsonr, spearmanr
import time

def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def down_scale(image):
    return skimage.transform.resize(image, (228, 228))

alexnet = models.alexnet(pretrained=True).cuda()


comp_algorithm = 'kpca'

layer = 'fc7'
path = '../data/compress_feats/kpca/' + layer
file_path = os.path.join(path, comp_algorithm+'_'+layer+'.pkl')
fc7_feat = pickle.load(open(file_path, "rb"))
file_path = os.path.join(path, comp_algorithm+'_'+layer+'_feat.pkl')
fc7_cmpfeat = pickle.load(open(file_path, "rb"))

layer = 'fc6'
path = '../data/compress_feats/kpca/' + layer 
file_path = os.path.join(path, comp_algorithm+'_'+layer+'.pkl')
fc6_feat = pickle.load(open(file_path, "rb"))
file_path = os.path.join(path, comp_algorithm+'_'+layer+'_feat.pkl')
fc6_cmpfeat = pickle.load(open(file_path, "rb"))

layer = 'pool5'
path = '../data/compress_feats/kpca/' + layer
file_path = os.path.join(path, comp_algorithm+'_'+layer+'.pkl')
pool5_feat = pickle.load(open(file_path, "rb"))
file_path = os.path.join(path, comp_algorithm+'_'+layer+'_feat.pkl')
pool5_cmpfeat = pickle.load(open(file_path, "rb"))

filelist=[]
test_file = []
room_labels = []
area_labels = []
features = {}

layers = ['fc7', 'fc6', 'pool5']
layers_index = [19, 17, 12]

j=0

rooms = os.listdir("../data/images")
for room in rooms:
    areas = os.listdir("../data/images/"+room)
    for area in areas:
        for file in glob.glob('../data/images/'+room+ '/' + area + '/*.jpg'):
            filelist.append(file)
            room_labels.append(room)
            area_labels.append(area)
            
for file in glob.glob('../data/images/test' + '/*.jpg'):
    test_file.append(file)
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

start = time.time()
test_fc7 = fc7_feat.transform(features['fc7'].cpu().data.numpy())
test_fc6 = fc6_feat.transform(features['fc6'].cpu().data.numpy())
test_pool5 = pool5_feat.transform(features['pool5'].cpu().data.numpy())
print(time.time()-start)

tree = BallTree(fc7_cmpfeat, leaf_size=5)
start = time.time()
dist, idx = tree.query(test_fc7, 10)
print(time.time()-start)
w7 = [1, 0, 0, 0.6, 0.20, 0.1]
w6 = [0, 1, 0, 0.3, 0.65, 0.3]
wp = [0, 0, 1, 0.1, 0.15, 0.6]

dist = {}
idx = {}

pear_out = {}
spear_out = {}

for j in range(idx.shape[0]):
    for i in range(6):
        target = np.concatenate((w7[i]*fc7_cmpfeat[idx[j]], w6[i]*fc6_cmpfeat[idx[j]], wp[i]*pool5_cmpfeat[idx[j]]), axis=1)
        test = np.concatenate((w7[i]*test_fc7[j],w6[i]*test_fc6[j], wp[i]*test_pool5[j]))
        
        for k in range(10):
            try:
                #pearsonr
                pear_out[j].append((i, idx[j][k],pearsonr(target[k], test)[0]))
        
                #spearmanr
                spear_out[j].append((i, idx[j][k],spearmanr(target[k], test)[0]))
            except KeyError:
                #pearsonr
                pear_out[j] = [(i, idx[j][k],pearsonr(target[k], test)[0])]
                start = time.time()
                #spearmanr
                spear_out[j] = [(i, idx[j][k],spearmanr(target[k], test)[0])]
                print(time.time()-start)
sort_pear = {}
sort_spear = {}
for j in range(idx.shape[0]):
    sort_pear[j] = sorted(pear_out[j], key = lambda i: i[2], reverse=True)
    sort_spear[j] = sorted(spear_out[j], key = lambda i: i[2], reverse=True)
    
for j in range(idx.shape[0]):
    k = [760,740,838,750]
    fig = plt.figure(figsize=(15,15))
    fig.add_subplot(1,5,1)
    plt.imshow(Image.open(test_file[21]).resize((128,128)))
    plt.axis('off')
    fig.add_subplot(1,5,2)
    plt.imshow(Image.open(filelist[k[0]]).resize((128,128)))
    plt.axis('off')
    fig.add_subplot(1,5,3)
    plt.imshow(Image.open(filelist[k[1]]).resize((128,128)))
    plt.axis('off')
    fig.add_subplot(1,5,4)
    plt.imshow(Image.open(filelist[k[2]]).resize((128,128)))
    plt.axis('off')
    fig.add_subplot(1,5,5)
    plt.imshow(Image.open(filelist[k[3]]).resize((128,128)))
    plt.axis('off')
    plt.savefig('image215.png', bbox_inches = 'tight')