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


# with parallelization
try:
    out = boot.generate(S_X, Y=S_y, method="tsne", B=4, save=False, random_seed=452, random_state=452, num_jobs=2)
except:
    print("Warning: Error encountered parallelization of boot.generate() -- ignore if only using one core")

# without parallelization
try:
    out = boot.generate(S_X, Y=S_y, method="tsne", B=4, save=False, random_seed=452, random_state=452)
except:
    raise Exception("Error encountered in boot.generate")


# check output
truth = pd.read_csv("tests/outputs/truth.csv")
assert out["x1"].values[333] == truth["x1"].values[333], "Output dataframe does not match truth.csv"
assert out["x2"].values[1111] == truth["x2"].values[1111], "Output dataframe does not match truth.csv"

# make interactive visualization
try:
    fig = viz.interactive(out, 'label', show=False, save=False, alpha=0.5, legend_title="Cell type", dpi=150)
except:
    print ("Warning: Error encountered for viz.interactive()")

# make animated visualization
try:
    fig = viz.animated(out, 'label', save=False)
except:
    print ("Warning: Error encountered for viz.animated()")

# make stacked visualization
try:
    fig = viz.stacked(out, 'label', show=False, save=False,
                    xlabel="t-SNE 1", ylabel="t-SNE 2", dpi=150, marker="x", s=20, show_legend=True, solid_legend=True, cmap='hot')
except:
    print ("Warning: Error encountered for viz.stacked()")
                    
# global variance score
try:
    variance_scores = score.variance(out, method="global")
except:
    print ("Warning: Error encountered for score.variance(method='global')")
    
# check variance scores
truth = np.genfromtxt("tests/outputs/truth_variances.txt")
assert variance_scores[333] == truth[333], "Variance scores do not match truth_variances.txt"
    
# random variance score
try:
    variance_scores = score.variance(out, method="random", k=50)
except:
    print ("Warning: Error encountered for score.variance(method='random')")
    
# concordance score test
for method in ["spearman", "distortion", "jaccard", "mean_projection_error", "stretch"]:
    try:
        concordance_scores = score.concordance(out, S_X, method=method, k=50, bootstrap_number=-1)
    except:
        print ("Warning: Error encountered for score.concordance(method='"+method+"')")


