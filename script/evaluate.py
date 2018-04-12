# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:35:54 2017

@author: Mahim
"""
import pickle
import os
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import BallTree

def load_feature_db(layer):
    file_path = os.path.join('../data/compress_feats/ipca', layer,comp_algorithm+'_'+layer+'.pkl')
    return pickle.load(open(file_path, "rb"))

def load_db_labels():
    file_path = os.path.join('../data', 'feats', 'location.pkl')
    return pickle.load(open(file_path, "rb"))
    
def load_tree(comp_algorithm, layer):
    file_path = os.path.join('../data', 'tree', comp_algorithm, layer, comp_algorithm+'_'+layer +'.pkl')
    return pickle.load(open(file_path, "rb"))

feature_layers = ['fc7', 'fc6', 'pool5', 'conv4', 'conv3', 'pool2', 'pool1']
indexing_algorithm = 'BallTree'
comp_algorithm = 'ipca'
layer = 'fc7'
algorithm = 'BallTree'

#features = load_feature_db(layer)
room_labels, area_labels = load_db_labels()

tree = load_tree(comp_algorithm, layer)
#tree = BallTree(features, leaf_size = 5)

test_features = compute_features(layer, x)

transform_features = transform(test_features, s)

prediction = compute_proximity(tree, transform_features)
