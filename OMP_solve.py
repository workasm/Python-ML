
#from __future__ import absolute_import, division, print_function, unicode_literals

#import pandas as pd
#import datetime as dt
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as slinalg # needed for linalg.dft
import scipy.misc
import sklearn.feature_extraction.image
from sklearn.datasets import make_sparse_coded_signal
from sklearn import decomposition
from sklearn import feature_extraction as fet
import imageio
import dictlearn as dl
import skimage.transform as trans

import OMP_algs

def loadImageSet(imageDir, pattern, maxTrainImages, rescale=1.0, normalize=False):
    flist = glob.glob(imageDir + '/' + pattern)

    images = []
    for i, fname in enumerate(flist):
        if i < maxTrainImages:
            img = np.array(imageio.imread(fname),dtype='float')/255
            if rescale < 1.0:
                img = trans.rescale(img, 0.5)
            if normalize:
                intercept = np.mean(img, axis=0) # this computes average over columns..
                print(f"mean sz: {intercept.shape}")
                img -= intercept
            images.append(img)
    dataSet = images[0]
    # axis = 0: rows; axis = 1: columns
    trainSet = np.concatenate(images[1:], axis=1)
    print(f"flist size: {len(flist)}, trainSet: {trainSet.shape}, dataSet: {dataSet.shape}")
    return (trainSet,dataSet)

def dictLearnSKLearn(trainSet, dataSet):
    n_features = trainSet.shape[0]  # number of features is a signal size : equals number of concat image rows
    n_samples = trainSet.shape[1]  # number of different training samples
    n_components = n_features * 2

    learner = decomposition.DictionaryLearning(n_components=n_components, alpha=1, max_iter=1000, tol=1e-05, fit_algorithm='lars',
                                      transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None,
                                      n_jobs=1, code_init=None, dict_init=None, verbose=True, split_sign=False,
                                      random_state=None, positive_code=False, positive_dict=False,
                                      transform_max_iter=500)
    learner.fit(trainSet.T[0:10,:])

    #sklearn.feature_extraction.image.extract_patches_2d()


def dictLearn1D():
    dictSavePath = "C:/work/RR_dictionary.npy"
    imageDir = "C:/work/Playground/external/RingRemoverImages/"
    trainSet,dataSet = loadImageSet(imageDir, '*_Polar.png', maxTrainImages=10, rescale=0.5, normalize=True)

    D = np.load(dictSavePath)
    print(D.shape)

    n_features = trainSet.shape[0]  # number of features is a signal size : equals number of concat image rows
    n_samples = trainSet.shape[1]  # number of different training samples
    n_atoms = n_features * 2
    print(f"dictLearn1D: n_features: {n_features}, n_atoms: {n_atoms}, n_samples: {n_samples}")

    #D = np.matrix(Dict[0:n_features, :])  # create an array of size M x N where M < N (dictionary)

    rng = np.random.RandomState(0)
    # signals of size n_features x n_samples
    learner = dl.algorithms.Trainer(trainSet, method='online', regularization='l0')

    size = int(math.sqrt(n_features))
    if False : #size * size == n_features:
        dictionary = dl.dct_dict(n_atoms, size)
        print(f"Using DCT dictionary: {dictionary.shape}")
    else:
        dictionary = dl.random_dictionary(n_features, n_atoms)

    # if iters is less than n_samples => samples are drawn randomly from the set..
    # maximal # of non-zero coefficients is n_features..
    #learner.train(dictionary=None, n_atoms=n_atoms, iters=2000, n_nonzero=20, tolerance=1e-3, n_threads=4, verbose=True)

    # NOTE: tol overrides n_nonzero
    D = dl.odl(trainSet, dictionary, iters=4000, n_nonzero=n_features, tol=0,
            verbose=True, batch_size=5, n_threads=1, seed=1113)

    print(D)
    plt.plot(D[:, 0:21]) # plot several dictionary columns
    np.save(dictSavePath, D)
    dl.visualize_dictionary(D,16,16)

    # u = Dw
    # |x|_0 = n_nonzero_coefs
    u, D, w = make_sparse_coded_signal(n_samples=1,
                                       n_components=100,
                                       n_features=50,
                                       n_nonzero_coefs=27,
                                       random_state=10)
    # D is of size n_features x n_components
    #soln = OMP_solve_basic(D,u)
    #soln = OMP_algs.OMP_solve_fast(D, u, eps=1e-7)
    #dif = w - soln
    #print(f"difference truth-solution norm: {np.linalg.norm(dif)}")


def dictLearn2D():
    imageDir = "C:/work/Playground/external/RingRemoverImages/";
    trainSet, dataSet = loadImageSet(imageDir, '*_Polar.png', 6)

    n_features = trainSet.shape[0]  # number of features is a signal size : equals number of concat image rows
    n_samples = trainSet.shape[1]  # number of different training samples
    n_components = n_features * 2

    # dictionary = dl.dct_dict(256, 16)
    patch_dim = 16  # we generate patches of size 16x16
    n_features = patch_dim**2
    dictionary = dl.random_dictionary(n_features, n_features * 2)

    rng = np.random.RandomState(0)
    # patches = fet.image.extract_patches_2d(
    #     image=trainSet, patch_size=(patch_dim,patch_dim),
    #     max_patches=1000, random_state=rng
    # )
    # NOTE: stride parameter does not work!!!
    image_patches = dl.Patches(trainSet, patch_dim, stride=16, max_patches=5000, random=331)
    image_patches.remove_mean()

    print(f"patches size: {image_patches.patches.shape}")

    dictionary = dl.odl(image_patches.patches, dictionary, iters=5000, n_nonzero=50, tol=0, verbose=True, batch_size=1,
                           n_threads=4, seed=None)
    #dictionary = dl.ksvd(image_patches, dictionary, 50, n_nonzero=8,
     #                    n_threads=4, verbose=True)

    dl.visualize_dictionary(dictionary, 16, 16)
    np.save("C:/work/RR_dictionary.npy",dictionary)
    np.load()

def ffun(a,b,c,d,e,f):
    print(f"a={a},b={b},c={c},d={d},e={e},f={f}")

if __name__ == '__main__':
    ffun(1,2,3,4,f=5,e=6)
    #dictLearn1D()
    #dictLearn2D()
