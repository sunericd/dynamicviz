"""
Bootstrap visualization module
- Tools for generating bootstrap dimensionality reduction (DR) visualizations
- Standalone methods for DR visualizations
"""

# author: Eric David Sun <edsun@stanford.edu>
# (C) 2022 
from __future__ import print_function

import sklearn.manifold as skm
from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
import pandas as pd
import os
import umap
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")


def dimensionality_reduction(X, method, **kwargs):
    '''
    Runs a given DR method (specified by method) for the data X (n by p matrix/array)
    
    Uses default arguments unless additional argument values specified in **kwargs
    '''
    # initialize 2D DR method
    if isinstance(method, str):
        if method == "tsne":
            X_embedded = skm.TSNE(n_components=2, verbose=0, **kwargs).fit_transform(X)
        elif method == "mds":
            X_embedded = skm.MDS(n_components=2, verbose=0, **kwargs).fit_transform(X)
        elif method == "lle":
            X_embedded = skm.LocallyLinearEmbedding(n_components=2, **kwargs).fit_transform(X)
        elif method == "mlle":
            X_embedded = skm.LocallyLinearEmbedding(n_components=2, method='modified', **kwargs).fit_transform(X)
        elif method == "isomap":
            X_embedded = skm.Isomap(n_components=2, **kwargs).fit_transform(X)
        elif method == "umap":
            X_embedded = umap.UMAP(n_components=2, verbose=0, **kwargs).fit_transform(X)
        elif method == "pca":
            X_embedded = PCA(n_components=2, **kwargs).fit_transform(X)
        else:
            raise Exception(method+" is not a recognized dimensionality reduction algorithm")
            
    # apply DR and obtain 2D embedding
    else:
        X_embedded = method.fit_transform(X)
    
    return(X_embedded)


def bootstrap (X, method, B, sigma_noise=None, no_bootstrap=False, random_seed=None, num_jobs=None, use_n_pcs=False, subsample=False, **kwargs):
    '''
    Creates n bootstrap data from X and creates a DR visualizastion for each of them.
    
    Arguments:
        See generate() for details
    
    Returns:
        X_embedded_list = list of the 2D DR visualization embeddings (numpy arrays)
        bootstrap_indices_list = list of numpy arrays indicating the bootstrap row indices
    '''

    X_embedded_list = []
    bootstrap_indices_list = []
    
    # generate sequence of random states if random_seed is specified
    if isinstance(random_seed, int):
        seeded_rand1 = np.random.RandomState(random_seed)
        random_seeded_sequence = seeded_rand1.randint(0,1e6,B)
    else:
        random_seeded_sequence = False
    
    # bootstrap DR
    if isinstance(num_jobs, int): # in parallel
        result = Parallel(n_jobs=num_jobs)(delayed(run_one_bootstrap)(X, method, sigma_noise, no_bootstrap,
                                           random_seeded_sequence, b, use_n_pcs, subsample, **kwargs) for b in tqdm(range(B)))
        X_embedded_list = [x[0] for x in result]
        bootstrap_indices_list = [x[1] for x in result]
    else: # using only one core
        for b in tqdm(range(B)):
            X_embedded, boot_idxs = run_one_bootstrap(X, method, sigma_noise, no_bootstrap,
                                           random_seeded_sequence, b, use_n_pcs, subsample, **kwargs)
            X_embedded_list.append(X_embedded)
            bootstrap_indices_list.append(boot_idxs)
    
    return(X_embedded_list, bootstrap_indices_list)


def run_one_bootstrap(X, method, sigma_noise=None, no_bootstrap=False, random_seeded_sequence=False, b=0, use_n_pcs=False, subsample=False, **kwargs):
    '''
    Method for generating one bootstrap X and one DR visualization of the bootstrap
    
    Arguments:
        random_seeded_sequence = array or list of random seeds to use in generating the bootstrap sample
        b = integer specifying the index of random see in random_seeded_sequence to use for generating the bootstrap
        See generate() for more details
    
    Returns:
        X_embedded = 2D DR visualization embedding (numpy array)
        boot_idxs = numpy array indicating the bootstrap row indices
    '''
    # Create bootstrap X
    if subsample is False:
        if no_bootstrap is True: # don't bootstrap (will use intrinsic stochasticity of DR algorithm (if any) only)
            boot_X = X.copy()
            boot_idxs = np.arange(X.shape[0]) # set indices to be the original indices
        elif random_seeded_sequence is not False: # use specified random_seeded_sequence to generate bootstrap X
            seeded_rand2 = np.random.RandomState(random_seeded_sequence[b])
            boot_idxs = seeded_rand2.randint(0,X.shape[0],X.shape[0])
            boot_X = X.copy()[boot_idxs,:] 
        else: # if no random_seeded_sequence, use default random process
            boot_idxs = np.random.randint(0,X.shape[0],X.shape[0])
            boot_X = X.copy()[boot_idxs,:]
    # Subsample instead of bootstrapping
    else:
        if random_seeded_sequence is not False: # use specified random_seeded_sequence to generate subsample of X
            seeded_rand2 = np.random.RandomState(random_seeded_sequence[b])
            boot_idxs = seeded_rand2.choice(X.shape[0], subsample, replace=False)
            boot_X = X.copy()[boot_idxs,:] 
        else: # if no random_seeded_sequence, use default random process
            boot_idxs = np.random.choice(X.shape[0], subsample, replace=False)
            boot_X = X.copy()[boot_idxs,:]
        
    # add Gaussian noise to alleviate duplicate issues if specified
    if sigma_noise is not None:
        boot_X += np.random.normal(0,sigma_noise,boot_X.shape)
        
    # run TruncatedSVD if specified to do a first pass DR with PCA and take use_n_pcs top principal components for DR visualization
    if use_n_pcs is not False:
        boot_X = TruncatedSVD(n_components=use_n_pcs).fit_transform(boot_X)
    
    # generate DR visualization embedding
    X_embedded = dimensionality_reduction(boot_X, method, **kwargs)


    return(X_embedded, boot_idxs)


def generate(X, method, Y=None, B=0, sigma_noise=None, no_bootstrap=False, random_seed=None, save=False,
             num_jobs=None, use_n_pcs=False, subsample=False, **kwargs):
    '''
    Main method for generating aligned bootstrap visualizations, which are the input elements for dynamic visualization.
    
    Arguments:
        X = (n x p) numpy array where rows are observations, columns are features
        method = string, dimensionality reduction method to use; options include:
            "tsne", "mds", "lle", "mlle", "isomap", "umap", "pca"
        Y = pandas dataframe with same number of rows as X and columns containing relevant metadata to propagate to output
        B = integer, number of bootstraps to generate; if B==0: generates only for the original X
        sigma_noise = None or float, if float, adds zero-centered Gaussian noise to each bootstrap sample with standard deviation sigma_noise
        no_bootstrap = True or False, whether to bootstrap sample for each iteration or not
        random_seed = None or int, if int, uses that value to generate random sequence (i.e. bootstrap sequence will be the same)
        save = False or str, if str, path to save resulting Pandas dataframe as CSV
        num_jobs = None, -1, or >=1 int; if not None, runs multiprocessing with n_jobs, if n_jobs=-1, then uses all available
        use_n_pcs = False or int, specifying to apply PCA and keep to use_n_pcs components to use for method
        subsample = False or int, specifying whether to subsample INSTEAD OF bootstrapping with integer corresponding to size of subsample to take
    
    Returns:
        output = Pandas dataframe with "x1", "x2", "bootstrap_number", "original_index" as columns, along with columns of Y
                    2D embedding is (x1, x2)
    '''
    # process results into dataframe
    output = pd.DataFrame() # init df to be merged onto
    
    # DR on original dataset
    original_embedding = dimensionality_reduction(X, method, **kwargs) # Ex: can specify random_state in **kwargs to remove stochasticity
    
    # reference points is the original dataset
    points0 = np.hstack((original_embedding, np.zeros(original_embedding.shape[0]).reshape(original_embedding.shape[0],1)))# append uniform 3rd dimension

    # add basic info
    output["x1"] = points0[:,0]-np.mean(points0[:,0]) # center reference visualization at (0,0)
    output["x2"] = points0[:,1]-np.mean(points0[:,1])
    output["original_index"] = np.arange(len(points0[:,0]))
    output["bootstrap_number"] = -1

    # add metadata
    if isinstance(Y, pd.DataFrame):
        for col in Y.columns:
            output[col] = Y[col].values
    
    # bootstrap
    if B > 0:
        bootstrap_embedding_list, bootstrap_indices_list = bootstrap(X, method, B, sigma_noise, no_bootstrap, 
                                                                    random_seed, num_jobs, use_n_pcs, subsample, **kwargs)
    # add bootstraps
    for i in range(len(bootstrap_embedding_list)):
        
        new_df = pd.DataFrame() # new df to merge onto original df
        
        points = bootstrap_embedding_list[i]
        points = np.hstack((points, np.zeros(points.shape[0]).reshape(points.shape[0],1)))# append uniform 3rd dimension
        boot_idxs = bootstrap_indices_list[i]
        
        # rigid alignment w/ 3d rotation
        ref_points = points0[boot_idxs,:]
        points[:,0] = points[:,0]-np.mean(points[:,0])
        points[:,1] = points[:,1]-np.mean(points[:,1])
        r = Rotation.align_vectors(ref_points, points)[0]
        rpoints = r.apply(points)
        
        # add basic info
        new_df["x1"] = rpoints[:,0]
        new_df["x2"] = rpoints[:,1]
        new_df["original_index"] = boot_idxs
        new_df["bootstrap_number"] = i
        
        # add metadata
        if isinstance(Y, pd.DataFrame):
            for col in Y.columns:
                new_df[col] = Y[col].values[boot_idxs]
        
        # merge to original dataframe
        output = pd.concat([output, new_df], axis=0)

    # save output
    if save is not False:
        output.to_csv(save, index=False)
    
    return(output)
        
        
        