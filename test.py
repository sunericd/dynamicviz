# rudimentary test to see if bootstrapping and interactive visualization is functioning
# similar to tutorial.ipynb


# import packages
from dynamicviz import boot, viz, score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr
from sklearn.datasets import make_s_curve


if not os.path.exists('tests/outputs'):
    os.makedirs('tests/outputs')

# load data
S_X, S_y = make_s_curve(1000, random_state=0)
S_y = pd.DataFrame(S_y, columns=["label"])


# without parallelization
try:
    out = boot.generate(S_X, Y=S_y, method="tsne", B=2, save="tests/outputs/test.csv", random_seed=452, random_state=452)
except:
    raise Exception("Error encountered in boot.generate")


# with parallelization
try:
    out = boot.generate(S_X, Y=S_y, method="tsne", B=4, save="tests/outputs/test.csv", random_seed=452, random_state=452, num_jobs=2)
except:
    print("Warning: Error encountered parallelization of boot.generate() -- ignore if only using one core")


# make interactive visualization
fig = viz.interactive(out, 'label', show=False, save='tests/outputs/test.html', alpha=0.5, legend_title="Cell type", dpi=150)


# UNCOMMENT BELOW AND FINISH WITH TRY HANDLING
    # Also consider knowing some exact numerical value matching for testing

# # continuous labels
# fig = viz.animated(out, 'label', save='tests/outputs/test_continuous.gif', alpha=0.2, title="S curve", 
                  # xlabel="t-SNE 1", ylabel="t-SNE 2", dpi=150, marker="x", s=20, solid_cbar=True)

# # save particular frames
# fig = viz.animated(out, 'label', save='tests/outputs/test_discrete.gif',
                  # get_frames=[0,2,4])



# fig = viz.stacked(out, 'label', show=False, save='tests/outputs/test.png',
                 # xlabel="t-SNE 1", ylabel="t-SNE 2", dpi=150, marker="x", s=20, show_legend=True, solid_legend=True,
                     # cmap='hot')
                    
# # global
# variance_scores = score.variance(out, method="global")

# # global
# variance_scores_normed = score.variance(out, method="global", normalize_pairwise_distance=True)

# # random approximation to global (much faster)
# variance_scores_random = score.variance(out, method="random", k=50)

# # local
# variance_scores_local = score.variance(out, method="local", X_orig=S_X, k=50)

# plt.hist(variance_scores_local)
# plt.show()

# # compute stability score from variance score
# stability_scores = score.stability_from_variance(variance_scores, alpha=20)


# # pearson correlation
# concordance_scores = score.concordance(out, S_X, method="pearson", bootstrap_number=-1)


# # mean projection error (transformed)
# concordance_scores2 = score.concordance(out, S_X, method="mean_projection_error", k=50, bootstrap_number=-1)

# # ensemble concordance
# ensemble_scores, concordance_scores_list = score.ensemble_concordance(out, 
    # S_X, methods=['pearson', 'spearman', 'jaccard', 'distortion',
                        # 'compression', 'stretch'],
    # bootstrap_number=-1)



