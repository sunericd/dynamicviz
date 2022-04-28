"""
Additional utilities module
- Tools for converting to/from AnnData format
"""

# author: Eric David Sun <edsun@stanford.edu>
# (C) 2022 
from __future__ import print_function

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

def convert_anndata (adata, obs, obsm="X_pca", n_components=None):
    '''
    Converts an AnnData object into X and Y inputs for boot.generate()
    
    Arguments:
        adata = AnnData object with all metadata of interest in anndata.obs and has adata.X and adata.obsm["X_pca"]
        obs = list of str, str correspond to a key in adata.obs
        obsm = str, str correspond to a key in adata.obsm (i.e. "X_pca") -- if no matches found, will default to adata.X
        n_components = None or int, if int and obsm="X_pca" is found, then take first n_components of X_pca
        
    Returns:
        X = numpy array with the same number of rows as adata.X and columns depending on obsm
        Y = Pandas dataframe with columns specified by obs, same number of rows as X
    '''
    # get X from adata.obsm or adata.X
    obsm_keys = list(adata.obsm.keys())
    if obsm in obsm_keys:
        X = adata.obsm[obsm]
        if obsm == "X_pca":
            X = X[:,:n_components]
    else:
        X = adata.X
    
    # get Y from adata.obs
    Y = adata.obs[obs]
    
    return (X, Y)

def regenerate_anndata (adata, df, bootstrap_number, obsm):
    '''
    Converts AnnData to match df at specified bootstrap_number
    
    Will replace obsm (e.g. "X_umap") with df[["x1", "x2"]]
    Will reorganize all other metadata and X to match bootstrap indices
    
    Arguments:
        adata = original AnnData
        df = pandas dataframe, output of boot.generate()
        bootstrap_number = int, index to use to regenerate anndata (i.e. an integer appearing in df["bootstrap_number"])
        obsm = str, key to replace adata.obsm[key] with df[["x1", "x2"]]
    
    Returns:
        adata_copy = new AnnData 
    '''
    bootstrap_idxs = df[df["bootstrap_number"] == bootstrap_number]["original_index"].values
    adata_copy = adata.copy()[bootstrap_idxs]
    adata_copy.obsm[obsm] = df[df["bootstrap_number"] == bootstrap_number][["x1", "x2"]].values # update DR embedding

    return(adata_copy)