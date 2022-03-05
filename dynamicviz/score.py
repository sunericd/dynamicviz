from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import eigh
from scipy.stats import pearsonr, spearmanr
import numpy as np
import os
#from numba import njit, jit, prange
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")
import time



# Stability score
def stability (df, method="global", alpha=1.0, k=20, X_orig=None, neighborhoods=None, normalize_pairwise_distance=False,
                include_original=True, num_jobs=None):
    '''
    Takes df (output dataframe of boot.generate()) and computes stability scores:
    
    Arguments:
        df = pandas dataframe, output of boot.generate()
        method = str, specifies the type of stability score to compute
            "global" - compute stability score across the entire dataset
            "random" - approximate global stability score by randomly selecting k "neighbors" for each observation
            "local" - compute stability over k-nearest neighbors (specify k, defaults to 20)
        alpha = float > 0, is the exponential paramter for stability score formula: stability = 1/(1+variance)^alpha
            defaults to alpha=1.0
        k = int, when method="local"
        neighborhood = array or list of size n, with elements labeling neighborhoods
        normalize_pairwise_distance = if True, then divide each set of d[i,j] by its mean before computing variance
        include_original = if True, include the original (bootstrap_number=-1) embedding in calculating scores
        num_jobs = None, -1, or >=1 int; if not None, runs multiprocessing with n_jobs, if n_jobs=-1, then uses all available
        
    Returns:
        stability_scores = numpy array with stability score for each observation
    '''
    # check alpha > 0
    if alpha <= 0:
        raise Exception("alpha must be >= 0")
    
    mean_variance_distances = variance(df, method=method, k=k, X_orig=X_orig, neighborhoods=neighborhoods,
                                        normalize_pairwise_distance=normalize_pairwise_distance,
                                        include_original=include_original, num_jobs=num_jobs)

    # compute stability score
    stability_scores = stability_from_variance(mean_variance_distances, alpha)
    
    return(stability_scores)


def variance (df, method="global", k=20, X_orig=None, neighborhoods=None, normalize_pairwise_distance=False,
                include_original=True, num_jobs=None):
    '''
    Takes df (output dataframe of boot.generate()) and computes variance scores:
    
    Arguments:
        df = pandas dataframe, output of boot.generate()
        method = str, specifies the type of stability score to compute
            "global" - compute stability score across the entire dataset
            "random" - approximate global stability score by randomly selecting k "neighbors" for each observation
            "local" - compute stability over k-nearest neighbors (specify k, defaults to 20)
        k = int, when method="local"
        neighborhood = array or list of size n, with elements labeling neighborhoods
        normalize_pairwise_distance = if True, then divide each set of d[i,j] by its mean before computing variance
        include_original = if True, include the original (bootstrap_number=-1) embedding in calculating scores
        num_jobs = None, -1, or >=1 int; if not None, runs multiprocessing with n_jobs, if n_jobs=-1, then uses all available
        
    Returns:
        mean_variance_distances = numpy array with variance score (mean variance in pairwise distance to neighborhood) for each observation
    '''
    # retrieve embeddings and bootstrap indices
    if include_original is True:
        embeddings = [np.array(df[df["bootstrap_number"]==b][["x1","x2"]].values) for b in np.unique(df["bootstrap_number"])]
        bootstrap_indices = [np.array(df[df["bootstrap_number"]==b]["original_index"].values) for b in np.unique(df["bootstrap_number"])]
    else:
        embeddings = [np.array(df[df["bootstrap_number"]==b][["x1","x2"]].values) for b in np.unique(df["bootstrap_number"]) if b!=-1]
        bootstrap_indices = [np.array(df[df["bootstrap_number"]==b]["original_index"].values) for b in np.unique(df["bootstrap_number"]) if b!=-1]
    
    # set up neighborhoods for variance score
    print("Setting up neighborhoods...")
    start_time = time.time()
    neighborhood_dict = get_neighborhood_dict(method, k, keys=np.unique(df["original_index"]), neighborhoods=neighborhoods, X_orig=X_orig)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # populate distance dict
    print("Populating distances...")
    start_time = time.time()
    dist_dict = populate_distance_dict(neighborhood_dict, embeddings, bootstrap_indices)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # compute mean pairwise distance for normalization
    print("Computing mean pairwise distance for normalization...")
    start_time = time.time()
    mean_pairwise_distance = compute_mean_distance(dist_dict, normalize_pairwise_distance)
    print("--- %s seconds ---" % (time.time() - start_time))

    # compute variances
    print("Computing variance scores...")
    start_time = time.time()
    mean_variance_distances = compute_mean_variance_distance(dist_dict, normalize_pairwise_distance, mean_pairwise_distance)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return(mean_variance_distances)


def stability_from_variance(mean_variance_distances, alpha):
    '''
    For alpha and mean_variance_distances, computes stability scores
    '''
    # compute stability score
    print("Computing stability score with alpha="+str(alpha)+" ...")
    start_time = time.time()
    stability_scores = 1 / (1 + mean_variance_distances)**alpha
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return(stability_scores)




#@njit(parallel=True)
def get_neighborhood_dict(method, k, keys, neighborhoods, X_orig):
    '''
    Numba parallelized
    Returns a neighborhood dictionary where keys are observation indices
    '''
    neighborhood_dict = dict()
    for key in keys:
        neighborhood_dict[str(key)] = []
        
    if neighborhoods is None:
    
        if method == "global":
            for n in range(len(neighborhood_dict.keys())):
                key = list(neighborhood_dict.keys())[n]
                neighborhood_dict[key] = [i for i in range(len(neighborhood_dict.keys()))]
        
        elif method == "random":
            for n in range(len(neighborhood_dict.keys())):
                key = list(neighborhood_dict.keys())[n]
                neighborhood_dict[key] = [i for i in np.random.randint(0,len(neighborhood_dict.keys()),k)]
        
        elif method == "local":
            if X_orig is None:
                raise Exception("Need to specify X_orig to compute nearest neighbors")
            for n in range(len(neighborhood_dict.keys())):
                key = list(neighborhood_dict.keys())[n]
                nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_orig) # compute w.r.t. reference
                distances_local, indices_local = nbrs.kneighbors(X_orig)
                neighborhood_dict[key] = [i for i in indices_local[int(key),1:k+1]]
        
        else:
            raise Exception ("Need to specify either method ('global', 'local') or neighborhood")  
    
    else:
        for n in range(len(neighborhood_dict.keys())):
            key = list(neighborhood_dict.keys())[n]
            label = neighborhoods[int(key)]
            neighborhood_dict[key] = [i for i in range(len(neighborhoods)) if neighborhoods[i]==label]
            
            
    return(neighborhood_dict)

#@njit(parallel=True)
def populate_distance_dict (neighborhood_dict, embeddings, bootstrap_indices):
    '''
    Numba parallelized
    Returns dictionary with pairwise dictionaries for all observations[i][j]
    '''
    dist_dict = dict()
    for key1 in neighborhood_dict.keys():
        dist_dict[str(key1)] = {}
        for key2 in neighborhood_dict[key1]:
            dist_dict[str(key1)][str(key2)] = []
            
    for b in range(len(embeddings)):
        dist_mat = pairwise_distances(embeddings[b], n_jobs=-1)
        boot_idxs = bootstrap_indices[b]
        
        for i in range(dist_mat.shape[0]):
            key1 = str(boot_idxs[i])
            neighbor_js = neighborhood_dict[key1]
            #for j in range(dist_mat.shape[1]):
            #    key2 = str(boot_idxs[j])
            #    if int(key2) in neighbor_js:
            #        dist_dict[key1][key2].append(dist_mat[i,j])
            for nj in neighbor_js:
                key2 = str(nj)
                js = [x[0] for x in np.argwhere(boot_idxs == nj)]
                for j in js:
                    dist_dict[key1][key2].append(dist_mat[i,j])

    return(dist_dict)

#@njit(parallel=True)
def compute_mean_distance(dist_dict, normalize_pairwise_distance):
    '''
    Computes mean pairwise distance across all (i,j)
    '''
    all_distances = []
    for n in range(len(dist_dict.keys())):
        key1 = list(dist_dict.keys())[n]
        for key2 in dist_dict[key1].keys():
            distances = np.array(dist_dict[key1][key2])
            if normalize_pairwise_distance is True: # perform additional local normalization before taking variance
                distances = distances/np.nanmean(distances)
            all_distances.append(np.nanmean(distances))
    mean_pairwise_distance = np.nanmean(all_distances)
    
    return(mean_pairwise_distance)

#@njit(parallel=True)
def compute_mean_variance_distance(dist_dict, normalize_pairwise_distance, mean_pairwise_distance):
    '''
    For each (i,j) compute the variance across all distances.
    Then for each i, average across all var(i,j)
    '''
    mean_variance_distances = np.ones(len(dist_dict.keys()))*np.inf
    
    for n in range(len(dist_dict.keys())):
        key1 = list(dist_dict.keys())[n]
        variances = []
        for key2 in dist_dict[key1].keys():
            distances = np.array(dist_dict[key1][key2])
            if normalize_pairwise_distance is True: # perform additional local normalization before taking variance
                distances = distances/np.nanmean(distances)
                
            variances.append(np.nanvar(distances / mean_pairwise_distance)) # normalize globally and compute variance
        
        mean_variance_distances[int(key1)] = np.nanmean(variances)
        
    return(mean_variance_distances)


        
# Concordance scores

def average_recall_precision(X_orig, X_red, k_list, precomputed=[False, False], return_points=False):
    '''
    X_orig, X_red are the n x p feature matrices
        If precomputed[0] or precomputed[1] is True, then the corresponding X is a distance/dissimilarity matrix
    k_list = list or array of k values that k1 and k2 can take on
    '''

    # INIT NEAREST NEIGHBORS
    orig_nns = []
    red_nns = []
    for k1 in k_list:
        # get k nearest neighbors in X_orig
        if precomputed[0] is False:
            nbrs = NearestNeighbors(n_neighbors=k1).fit(X_orig)
        else:
            nbrs = NearestNeighbors(n_neighbors=k1, metric='precomputed').fit(X_orig)
        distances_orig, indices_orig = nbrs.kneighbors(X_orig)
        orig_nns.append(indices_orig)
    for k2 in k_list:
        # get k nearest neighbors in X_red
        if precomputed[0] is False:
            nbrs = NearestNeighbors(n_neighbors=k2).fit(X_red)
        else:
            nbrs = NearestNeighbors(n_neighbors=k2, metric='precomputed').fit(X_red)
        distances_red, indices_red = nbrs.kneighbors(X_red)
        red_nns.append(indices_red)
    
    # AVERAGE PRECISION
    precisions = []
    for i in range(X_orig.shape[0]):
        k1_precs = []
        for i1, k1 in enumerate(k_list):
            k2_precs = []
            for i2, k2 in enumerate(k_list):
                # get indices of nns
                indices_orig = orig_nns[i1]
                indices_red = red_nns[i2]
                # compute average precision
                numerator = len(list(set(indices_orig[i,1:]) & set(indices_red[i,1:]))) # ignore self neighbor (first index)
                prec = numerator/k2
                k2_precs.append(prec)
            # Get best k2 precision for each k1
            k1_precs.append(np.max(k2_precs))
        # Get worst k1 precision
        precisions.append(np.min(k1_precs))
    avg_precision = np.mean(precisions)
    
    # AVERAGE RECALL
    recalls = []
    for i in range(X_orig.shape[0]):
        k2_recs = []
        for i2, k2 in enumerate(k_list):
            k1_recs = []
            for i1, k1 in enumerate(k_list):
                # get indices of nns
                indices_orig = orig_nns[i1]
                indices_red = red_nns[i2]
                # compute average precision
                numerator = len(list(set(indices_orig[i,1:]) & set(indices_red[i,1:]))) # ignore self neighbor (first index)
                rec = numerator/k1
                k1_recs.append(rec)
            # Get best k2 precision for each k1
            k2_recs.append(np.max(k1_recs))
        # Get worst k1 precision
        recalls.append(np.min(k2_recs))
    avg_recall = np.mean(recalls)
    
    if return_points is False:
        return(avg_recall, avg_precision)
    else:
        return(recalls, precisions)


def get_jaccard(X_orig, X_red, k, precomputed=[False, False]):
    '''
    Computes Jaccard coefficient at k for each point in X_orig and X_red
    '''
    # INIT NEAREST NEIGHBORS
    if precomputed[0] is False:
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_orig)
    else:
        nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed').fit(X_orig)
    distances_orig, indices_orig = nbrs.kneighbors(X_orig)
    
    if precomputed[0] is False:
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_red)
    else:
        nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed').fit(X_red)
    distances_red, indices_red = nbrs.kneighbors(X_red)
    
    # COMPUTE JACCARD
    jaccards = []
    for i in range(X_orig.shape[0]):
        list1 = list(indices_orig[i,1:])
        list2 = list(indices_red[i,1:])
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        jaccards.append(float(intersection) / union)
    
    return (jaccards)


def get_distortion(X_orig, X_red, k, precomputed=[False, False]):
    '''
    Computes Distortion at k for each point in X_orig and X_red
    
    Distortion = ABS ( LOG [ (D_furthest/D_nearest)_orig / (D_furthest/D_nearest)_red ] )
    Distortions normalized by the maximum in entire dataset to be from [0,1] and reframed so 1 is best
    '''
    # INIT NEAREST NEIGHBORS
    if precomputed[0] is False:
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_orig)
    else:
        nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed').fit(X_orig)
    distances_orig, indices_orig = nbrs.kneighbors(X_orig)
    
    if precomputed[0] is False:
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_red)
    else:
        nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed').fit(X_red)
    distances_red, indices_red = nbrs.kneighbors(X_red)
    
    # COMPUTE DISTORTION
    distortions = []
    for i in range(X_orig.shape[0]):
        orig_ratio = np.max(distances_orig[i,1:]) / np.min(distances_orig[i,1:])
        red_ratio = np.max(distances_red[i,1:]) / np.min(distances_red[i,1:])
        distortions.append(np.abs(np.log(orig_ratio/red_ratio)))
        
    distortions = np.array(distortions)/np.max(distortions)
    distortions = 1-distortions
    
    return (distortions)


def get_mean_projection_error(X_orig, X_red):
    '''
    Computes mean projection error (MPE) modified from aggregated projection error by Martins et al., 2014
    
    MPE_i = MEAN_j { ABS[ D[i,j]_orig / max(D[i,j]_orig) - D[i,j]_red / max(D[i,j]_red) ] }
    
    Normalized to [0,1] and then reframed so 1 is best
    '''
    orig_distance = pairwise_distances(X_orig)
    red_distance = pairwise_distances(X_red)
    
    # normalize distances
    for i in range(orig_distance.shape[0]):
        orig_distance[i,:] = orig_distance[i,:]/np.max(orig_distance[i,:])
        red_distance[i,:] = red_distance[i,:]/np.max(red_distance[i,:])
    
    # compute projection errors and then MPE
    projection_errors = np.abs(orig_distance-red_distance)
    MPEs = np.mean(projection_errors, axis=1)
    MPEs = 1-MPEs/np.max(MPEs)
    return(MPEs)


def get_projection_precision_score(X_orig, X_red, k, precomputed=[False, False]):
    '''
    Computes projection precision score (PPS) based on Schreck 2007
    
    For k nearest neighbors:
    
    PPS_i(k) = SUM_j { || D[i,j]_orig(k) / ||D[i,]_orig|| - D[i,j]_red / ||D[i,]_red|| || }
    
    The final PPS_i are normalized by max(PPS_i) to get something between [0,1] and then reframed so 1 is best
    '''
    # INIT NEAREST NEIGHBORS
    if precomputed[0] is False:
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_orig)
    else:
        nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed').fit(X_orig)
    distances_orig, indices_orig = nbrs.kneighbors(X_orig)
    
    if precomputed[0] is False:
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_red)
    else:
        nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed').fit(X_red)
    distances_red, indices_red = nbrs.kneighbors(X_red)
    
    # normalize distances
    for i in range(distances_orig.shape[0]):
        distances_orig[i,:] = distances_orig[i,:]/np.linalg.norm(distances_orig[i,:])
        distances_red[i,:] = distances_red[i,:]/np.linalg.norm(distances_red[i,:])
    
    # compute projection errors and then PPS
    projection_scores = np.abs(distances_orig-distances_red)
    PPSs = np.sum(projection_scores, axis=1)
    PPSs = PPSs/np.max(PPSs)
    PPSs = 1-PPSs
    
    return(PPSs)


def get_compression(X_orig, X_red):
    '''
    Implemented according to Aupetit, 2007:
    
    compression_i = [ u_i - min_k {u_k} ] / [ max_k {u_k} - min_k {u_k} ] 
    
    u_i = SUM_j D_ij^+  , D_ij^+ = max {D_orig_ij - D_red_ij , 0 }  -- D here being Euclidean distance matrix
    
    compressions reframed so 1 is best
    '''
    orig_distance = pairwise_distances(X_orig)
    red_distance = pairwise_distances(X_red)
    
    # normalize distances
    for i in range(orig_distance.shape[0]):
        orig_distance[i,:] = orig_distance[i,:]/np.linalg.norm(orig_distance[i,:])
        red_distance[i,:] = red_distance[i,:]/np.linalg.norm(red_distance[i,:])
    
    D_pos = orig_distance-red_distance
    D_pos[D_pos<0] = 0
    
    U = np.sum(D_pos, axis=1)
    compressions = (U-np.min(U))/(np.max(U)-np.min(U))
    compressions = 1-compressions
    
    return(compressions)


def get_stretch(X_orig, X_red):
    '''
    Implemented according to Aupetit, 2007:
    
    compression_i = [ u_i - min_k {u_k} ] / [ max_k {u_k} - min_k {u_k} ] 
    
    u_i = SUM_j D_ij^+  , D_ij^+ = max {-(D_orig_ij - D_red_ij), 0 }  -- D here being Euclidean distance matrix
    
    stretchs reframed so 1 is best
    '''
    orig_distance = pairwise_distances(X_orig)
    red_distance = pairwise_distances(X_red)
    
    # normalize distances
    for i in range(orig_distance.shape[0]):
        orig_distance[i,:] = orig_distance[i,:]/np.linalg.norm(orig_distance[i,:])
        red_distance[i,:] = red_distance[i,:]/np.linalg.norm(red_distance[i,:])
    
    D_neg = orig_distance-red_distance
    D_neg[D_neg>0] = 0
    D_neg = -D_neg
    
    U = np.sum(D_neg, axis=1)
    stretchs = (U-np.min(U))/(np.max(U)-np.min(U))
    stretchs = 1-stretchs
    
    return(stretchs)


def concordance(df, X_orig, method, k=None, bootstrap_number=-1):
    '''
    Computes concordance scores between the projections in df and X_orig
    
    Arguments:
        df = pandas dataframe: output of boot.generate()
        X_orig = nxp numpy array that is the original data from which df was generated
        method = str: 'precision', 'recall', 'pearson', 'spearman', 'jaccard', 'distortion',
                 'mean_projection_error', 'projection_precision_score', 'compression', 'stretch'
        k = int, neighborhood size to consider (jaccard, distortion, projection_precision_score, precision, recall)
        bootstrap_number = int, index of bootstrap to compute metrics for; defaults to -1 which is the original/unbootstrapped projection
        
    Returns:
        metrics = numpy array with quality score for each row of df (according to the method specified) [0 is bad, 1 is good]
    '''
    # retrieve embeddings
    X_red = df[df["bootstrap_number"]==bootstrap_number][["x1","x2"]].values
    
    # shuffle X_orig to matching format
    boot_idxs = df[df["bootstrap_number"]==bootstrap_number]["original_index"].values
    X_orig = X_orig[boot_idxs,:]
    
    # set k to a globally relevant value if None
    if k is None:
        k = round(X_orig.shape[0]/2-1)
    if k < 5:
        raise Exception('k needs to be >= 5 or number of observations in X is too small')
    
    if method == 'precision':
        recalls, metrics = average_recall_precision(X_orig, X_red, np.arange(5,k,5), return_points=True)
    elif method == 'recall':
        metrics, precisions = average_recall_precision(X_orig, X_red, np.arange(5,k,5), return_points=True)
    elif method == 'pearson':
        orig_distance = pairwise_distances(X_orig)
        red_distance = pairwise_distances(X_red)
        metrics = [pearsonr(orig_distance[i,:],red_distance[i,:])[0] for i in range(red_distance.shape[0])]
    elif method == 'spearman':
        orig_distance = pairwise_distances(X_orig)
        red_distance = pairwise_distances(X_red)
        metrics = [spearmanr(orig_distance[i,:],red_distance[i,:])[0] for i in range(red_distance.shape[0])]
    elif method == 'jaccard':
        metrics = get_jaccard(X_orig, X_red, k)
    elif method == 'distortion':
        metrics = get_distortion(X_orig, X_red, k)
    elif method == 'mean_projection_error':
        metrics = get_mean_projection_error(X_orig, X_red)
    elif method == 'projection_precision_score':
        metrics = get_projection_precision_score(X_orig, X_red, k)
    elif method == 'compression':
        metrics = get_compression(X_orig, X_red)
    elif method == 'stretch':
        metrics = get_stretch(X_orig, X_red)
        
    return(metrics)


def ensemble_concordance(df, X_orig, methods=None, k=None, bootstrap_number=-1, verbose=True):
    '''
    Compute emsemble concordation via spectral meta-weights
    
    Arguments:
        df = pandas dataframe: output of boot.generate()
        X_orig = nxp numpy array that is the original data from which df was generated
        methods = list of strings specifying the metrics to use
            Defaults to all the available metrics (except for precision and recall)
        k = int, neighborhood size to consider (jaccard, distortion, projection_precision_score, precision, recall)
        bootstrap_number = int, index of bootstrap to compute metrics for; defaults to -1 which is the original/unbootstrapped projection
        verbose = True or False, whether to return warnings about negative spectral weights
    
    Returns:
        ensemble_metric = numpy array of ensemble concordance scores
        pointwise_metrics_list = list of concordance score arrays corresponding to pointwise_metrics_labels
    '''
    if methods is None:
        methods = ['pearson', 'spearman', 'jaccard', 'distortion', 'mean_projection_error',
                                    'projection_precision_score', 'compression', 'stretch']
    
    # compute individual metrics
    pointwise_metrics_list = []

    for metric in tqdm(methods):
        m = concordance(df, X_orig, method=metric, k=k, bootstrap_number=bootstrap_number)
        pointwise_metrics_list.append(np.array(m))
    
    # compute correlation matrix and eigendecomposition
    mat = np.corrcoef(pointwise_metrics_list)
    w, v = np.linalg.eig(mat)
    pc1_score = v[:,0]
    
    # check pc1 for one sign only
    summed_signs = np.abs(np.sum(np.sign(pc1_score)))
    if verbose is True:
        if summed_signs == len(pc1_score):
            print ("PC1 is same signed")
        else:
            print ("Warning: PC1 is mixed signed")
            
    # compute meta-uncertainty
    weighted_metrics_list = [pointwise_metrics_list[i]*np.abs(pc1_score)[i] for i in range(len(pointwise_metrics_list))]
    ensemble_metric = np.sum(weighted_metrics_list,axis=0)/np.sum(np.abs(pc1_score))
    
    return(ensemble_metric, pointwise_metrics_list)