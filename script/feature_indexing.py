# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 19:55:28 2017
compress all the features depending on the copression algorithm and dimensions specified
@author: Mahim
"""
import os
import pickle
from sklearn.neighbors import BallTree

def load_feature_db(layer, compression):
    file_path = os.path.join('../data/compress_feats', compression, compression+layer+'.pkl')
    return pickle.load(open(file_path, "rb"))

layers = ['fc7', 'fc6', 'pool5', 'conv4']
compressions = ['ipca', 'kpca', 'grp', 'srp']
layer='fc7'
compression = 'ipca'
for compression in compressions:
    for layer in layers:
        compress_feat = load_feature_db(layer, compression)
        tree = BallTree(compress_feat, leaf_size = 5)
        file_path = os.path.join('../data/tree', compression, layer, compression+'_'+layer+'.pkl')
        pickle.dump(tree, open(file_path, 'wb'))