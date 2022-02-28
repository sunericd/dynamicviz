import sklearn.manifold as skm
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import umap
from scipy.spatial.transform import Rotation
#from numba import njit, prange
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")


def dimensionality_reduction(X, method, **kwargs):

    # init DR method
    if method == "tsne":
        DR = skm.TSNE(n_components=2, verbose=0, **kwargs).fit(X)
    elif method == "mds":
        DR = skm.MDS(n_components=2, verbose=0, **kwargs).fit(X)
    elif method == "lle":
        DR = skm.LocallyLinearEmbedding(n_components=2, verbose=0, **kwargs).fit(X)
    elif method == "mlle":
        DR = skm.LocallyLinearEmbedding(n_components=2, verbose=0, **kwargs).fit(X)
    elif method == "hlle":
        DR = skm.LocallyLinearEmbedding(n_components=2, verbose=0, **kwargs).fit(X)
    elif method == "ltsa":
        DR = skm.LocallyLinearEmbedding(n_components=2, verbose=0, **kwargs).fit(X)
    elif method == "isomap":
        DR = skm.Isomap(n_components=2, verbose=0, **kwargs).fit(X)
    elif method == "umap":
        DR = umap.UMAP(n_components=2, verbose=0, **kwargs).fit(X)
    elif method == "pca":
        DR = PCA(n_components=2, verbose=0, **kwargs).fit(X)
    else:
        raise Exception(method+" is not a recognized dimensionality reduction algorithm")
        
    # get embeddings
    if method == "pca":
        X_embedded = DR.transform(X)
    else:
        X_embedded = DR.embedding_
    
    return(X_embedded)


def bootstrap (X, method, n, sigma_noise, no_bootstrap, random_seed, num_jobs, **kwargs):

    X_embedded_list = []
    bootstrap_indices_list = []
    
    # generate sequence of random states if random_seed is specified
    if isinstance(random_seed, int):
        seeded_rand1 = np.random.RandomState(random_seed)
        random_seeded_sequence = seeded_rand1.randint(0,1e6,n)
    else:
        random_seeded_sequence = False
    
    # bootstrap DR
    if isinstance(num_jobs, int):
        result = Parallel(n_jobs=num_jobs)(delayed(run_one_bootstrap)(X, method, sigma_noise, no_bootstrap,
                                           random_seeded_sequence, b, **kwargs) for b in tqdm(range(n)))
        X_embedded_list = [x[0] for x in result]
        bootstrap_indices_list = [x[1] for x in result]
    else:
        for b in tqdm(range(n)):
            X_embedded, boot_idxs = run_one_bootstrap(X, method, sigma_noise, no_bootstrap,
                                           random_seeded_sequence, b, **kwargs)
            X_embedded_list.append(X_embedded)
            bootstrap_indices_list.append(boot_idxs)
    
    return(X_embedded_list, bootstrap_indices_list)


def run_one_bootstrap(X, method, sigma_noise, no_bootstrap, random_seeded_sequence, b, **kwargs):

    if no_bootstrap is True:
        boot_X = np.copy(X)
        boot_idxs = np.arange(X.shape[0])
    if random_seeded_sequence is not False: # use random_seeded_sequence
        seeded_rand2 = np.random.RandomState(random_seeded_sequence[b])
        boot_idxs = seeded_rand2.randint(0,X.shape[0],X.shape[0])
        boot_X = np.copy(X)[boot_idxs,:]
    else:
        boot_idxs = np.random.randint(0,X.shape[0],X.shape[0])
        boot_X = np.copy(X)[boot_idxs,:]
    if sigma_noise is not None:
        boot_X += np.random.normal(0,add_noise,boot_X.shape)
    #try:
    X_embedded = dimensionality_reduction(boot_X, method, **kwargs)
    #except:
    #    boot_idxs = np.nan
    #    print('error encountered on bootstrap number '+str(b))
    #    continue

    return(X_embedded, boot_idxs)


#@njit(parallel=True)
def generate(X, method, Y=None, n=0, sigma_noise=None, no_bootstrap=False, random_seed=None, save=False,
             num_jobs=None, **kwargs):
    '''
    Arguments:
        X = 2D numpy array where rows are observations, columns are features
        method = string, dimensionality reduction method to use; options include:
            "tsne", "mds", "lle", "mlle", "hlle", "ltsa", "isomap", "umap", "pca"
        Y = pandas dataframe with same number of rows as X and columns containing relevant metadata to propagate to output
        n = integer, number of bootstraps to generate; if n==0: generates only for the original X
        sigma_noise = None or float, if float, adds zero-centered Gaussian noise to each bootstrap sample
        no_bootstrap = True or False, whether to bootstrap sample for each iteration or not
        random_seed = None or int, if int, uses that value to generate random sequence (i.e. bootstrap sequence will be the same)
        save = False or str, if str, path to save resulting Pandas dataframe as CSV
        num_jobs = None, -1, or >=1 int; if not None, runs multiprocessing with n_jobs, if n_jobs=-1, then uses all available
    
    Returns:
        output = Pandas dataframe with "x1", "x2", "bootstrap_number", "original_index" as columns, along with columns of Y
                    2D embedding is (x1, x2)
    '''
    # process results into dataframe
    output = pd.DataFrame() # init df to be merged onto
    
    # DR on original dataset
    original_embedding = dimensionality_reduction(X, method, **kwargs) # specify random_state in **kwargs to fix algo
    
    # reference points is the original dataset
    points0 = np.hstack((original_embedding, np.zeros(original_embedding.shape[0]).reshape(original_embedding.shape[0],1)))# append uniform 3rd dimension

    # add basic info
    output["x1"] = points0[:,0]-np.mean(points0[:,0])
    output["x2"] = points0[:,1]-np.mean(points0[:,1])
    output["original_index"] = np.arange(len(points0[:,0]))
    output["bootstrap_number"] = -1

    # add metadata
    if isinstance(Y, pd.DataFrame):
        for col in Y.columns:
            output[col] = Y[col].values
    
    
    # bootstrap
    if n > 0:
        bootstrap_embedding_list, bootstrap_indices_list = bootstrap(X, method, n, sigma_noise, no_bootstrap, 
                                                                    random_seed, num_jobs, **kwargs)
    # add bootstraps
    for i in range(len(bootstrap_embedding_list)):
        
        new_df = pd.DataFrame() # new df to merge onto original df
        
        points = bootstrap_embedding_list[i]
        points = np.hstack((points, np.zeros(points.shape[0]).reshape(points.shape[0],1)))# append uniform 3rd dimension
        boot_idxs = bootstrap_indices_list[i]
        
        # rotation alignment
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
        
        
        