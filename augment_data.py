import argparse
import datetime

import h5py
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import mixture


parser = argparse.ArgumentParser()
parser.add_argument('infile')
parser.add_argument('outfile')
parser.add_argument('--method', default='gmm')
parser.add_argument('--normalize-weight', default='inverse_mean_std')
parser.add_argument('--input-num-points', type=int)
parser.add_argument('--num-clusters', type=int)
args = parser.parse_args()


import datetime

def augment_one_sample_with_super_points(data, num_clusters, method='gmm', normalize_weight='inverse_std'):
    aug_data = np.zeros((data.shape[0], 7))
    aug_data[:,:3] = data
    
    if method == "km":
        km = KMeans(n_clusters=num_clusters).fit(data)
        spnts = []
        weights = []
        for l in range(num_clusters):
            cluster = data[km.labels_==l]
            spnts.append(cluster.mean(axis=0))
            weights.append(cluster.shape[0])
    
        labels = km.labels_
        sp = np.array(spnts)
        sw = np.array(weights)/data.shape[0]
    
    elif method == "gmm":
        gmm = mixture.GaussianMixture(n_components=num_clusters, covariance_type='spherical').fit(data)
        labels = gmm.predict(data)
        sp = gmm.means_
        sw = gmm.weights_
            
    # Normalize weights
    if normalize_weight == 'inverse':
        sw = 1/sw
    elif normalize_weight == 'inverse_std':
        sw = 1/sw
        sw /= sw.std()
    elif normalize_weight == 'inverse_mean_std':
        sw = 1/sw
        sw -= sw.mean()
        sw /= sw.std()
    elif normalize_weight == 'std':
        sw /= sw.std()
    elif normalize_weight == 'mean_std':
        sw -= sw.mean()
        sw /= sw.std()
    elif normalize_weight is not False:
        raise ValueError("Unexpected normalize_weight option")
        
    # Augment with cluster centers, weights
    for i in range(aug_data.shape[0]):
        aug_data[i, 3:6] = sp[labels[i]]
        aug_data[i, 6] = sw[labels[i]]
    
    return sp, sw, aug_data

def augment_with_super_points(batch_data, num_clusters, method='gmm', normalize_weight='inverse_std'):
    aug_batch_data = np.zeros((batch_data.shape[0], batch_data.shape[1], 7))
    for i in range(aug_batch_data.shape[0]):
        _, _, aug_batch_data[i,:,:] = augment_one_sample_with_super_points(
            batch_data[i,:,:], num_clusters, method, normalize_weight
        )
        if (i+1) % 32 == 0:
            print("{}: {}".format(datetime.datetime.now(), i+1))
    return aug_batch_data

fin = h5py.File(args.infile, 'r')

data = fin['data'][:]
aug_data = augment_with_super_points(
    data[:,:args.input_num_points,:], args.num_clusters, method=args.method, normalize_weight=args.normalize_weight
)
aug_data = np.array(aug_data, dtype=np.float32)

fout = h5py.File(args.outfile, 'w')
fout.create_dataset('data', data=aug_data)
fout.create_dataset('label', data=fin['label'][:])
fout.close()

fin.close()

