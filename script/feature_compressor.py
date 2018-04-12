# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 19:55:28 2017
compress all the features depending on the copression algorithm and dimensions specified
@author: Mahim
"""
import os
import pickle
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

layers = ['fc7', 'fc6', 'pool5', 'conv4']

#compressing training features
for layer in layers:
    file_path = os.path.join('../data', 'feats', layer+'.pkl')
    f = pickle.load(open(file_path, "rb"))
    f = f.cpu()
    ## ipca
    ipca = IncrementalPCA(n_components=128)
    ipca.fit(f.data.numpy())
    x_new = ipca.transform(f.data.numpy())
    file_path = os.path.join('../data/compress_feats/ipca', layer, 'ipca_'+layer+'_feat.pkl')
    pickle.dump(x_new, open(file_path, 'wb'))
    file_path = os.path.join('../data/compress_feats/ipca', layer, 'ipca_'+layer+'.pkl')
    pickle.dump(ipca, open(file_path, 'wb'))
    ## kpca
    kpca = KernelPCA(n_components=128)
    kpca.fit(f.data.numpy())
    x_new = kpca.transform(f.data.numpy())
    file_path = os.path.join('../data/compress_feats/kpca', layer, 'kpca_'+layer+'_feat.pkl')
    pickle.dump(x_new, open(file_path, 'wb'))
    file_path = os.path.join('../data/compress_feats/kpca', layer, 'kpca_'+layer+'.pkl')
    pickle.dump(kpca, open(file_path, 'wb'))
    #grp
    grp = GaussianRandomProjection(n_components=128)
    grp.fit(f.data.numpy())
    x_new = grp.transform(f.data.numpy())
    file_path = os.path.join('../data/compress_feats/grp', layer, 'grp_'+layer+'_feat.pkl')
    pickle.dump(x_new, open(file_path, 'wb'))
    file_path = os.path.join('../data/compress_feats/grp', layer, 'grp_'+layer+'.pkl')
    pickle.dump(grp, open(file_path, 'wb'))
    #srp
    srp = SparseRandomProjection(n_components=128)
    srp.fit(f.data.numpy())
    x_new = srp.transform(f.data.numpy())
    file_path = os.path.join('../data/compress_feats/srp', layer, 'srp_'+layer+'_feat.pkl')
    pickle.dump(x_new, open(file_path, 'wb'))
    file_path = os.path.join('../data/compress_feats/srp', layer, 'srp_'+layer+'.pkl')
    pickle.dump(srp, open(file_path, 'wb'))
    