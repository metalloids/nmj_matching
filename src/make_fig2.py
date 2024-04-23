#!/usr/bin/env python3

# so no GDK warning issues when running on fritos.
import matplotlib
matplotlib.use('Agg')

from matplotlib import pylab as plt
from matplotlib import rcParams
import seaborn as sns

#   v3: 2023-08-13
#       -- now parsing the efficiency-fairness table.
#

# Set plotting style.
plt.style.use('fivethirtyeight')
plt.rc('font', **{'sans-serif':'Arial', 'family':'sans-serif'})
rcParams.update({'font.size': 18})

TITLES = ["Peer review\nN = 73 papers\nM = 188 reviewers",
          "Co-expression similarity\nN = 1501 yeast genes\nM = 2251 worm genes",
          "MovieLens1M\nN = 3706 movies\nM = 6040 users",
          "Matching (NMJ)\nN = 15 neurons\nM = 217 fibers",
          "Matching (C. elegans)\nN = 47 neurons\nM = 47 neurons", 
          "Entity resolution\nN = 1076 Abt.com\nM = 1076 Buy.com",
          "Assignment\nN = 5000 agents\nM = 5000 items"]

FILES= ["uiuc_a0.001_b1.0.txt",
        "jessey2r_a0.001_b1.0.txt",
        "movielens_a0.001_b1.0.txt",
        "yaron_a0.001_b1.0.txt",
        "celegans_a0.001_b1.0.txt",
        "abtbuy_a0.001_b1.0.txt",
        "assignp5000_a0.001_b1.0.txt"]

#COLORS=["tab:gray","tab:red","tab:blue","tab:olive"]
COLORS=['#8c8c8c', '#c44e52','#4c72b0', '#ccb974']

opt_mean,alg_mean,gred_mean,rand_mean = [],[],[],[]
for filename in FILES:
    with open("../results/"+filename) as f:
        for line in f:
            if line.startswith("#"): continue
            cols = line.strip().split("\t")
            
            if cols[0].startswith("OPT"):
                opt_mean.append(float(cols[1].split("±")[0].strip()))
                #opt_std.append(float(cols[1].split("±")[1].strip()))

            elif cols[0].startswith("Neural"):
                alg_mean.append(float(cols[1].split("±")[0].strip()))
                #alg_std.append(float(cols[1].split("±")[1].strip()))
    
            elif cols[0].startswith("Greedy"):
                gred_mean.append(float(cols[1].split("±")[0].strip()))
                #gred_std.append(float(cols[1].split("±")[1].strip()))

            elif cols[0].startswith("Rand"):
                rand_mean.append(float(cols[1].split("±")[0].strip()))
                #rand_std.append(float(cols[1].split("±")[1].strip()))

            else:
                assert False

# Check all the sizes are the same.            
assert len(opt_mean) == len(alg_mean) == len(gred_mean) ==  len(rand_mean) == len(TITLES) == len(FILES)

# Plot.
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13.3,9.6))
fig.tight_layout(h_pad=3.75,w_pad=2.0) # h_pad adds more space between rows.
idx = 0

for i in range(2):
    for j in range(4):

        if i == 1 and j == 3: break # no plot for the last spot

        axes[i,j].bar(range(4),[100*opt_mean[idx]/opt_mean[idx],100*alg_mean[idx]/opt_mean[idx],100*gred_mean[idx]/opt_mean[idx],100*rand_mean[idx]/opt_mean[idx]],color=COLORS,tick_label=["OPT","Alg","Grdy","Rand"],alpha=0.80) 
        # yerr=[100*opt_std[idx]/opt_mean[idx],100*alg_std[idx]/opt_mean[idx],100*gred_std[idx]/opt_mean[idx],100*rand_std[idx]/opt_mean[idx]],
        
        # Only put ylabel on left-most plots.
        if j == 0: axes[i,j].set_ylabel("Efficiency (%)")

        axes[i,j].tick_params(axis='x', labelsize=15)
        axes[i,j].set_title(TITLES[idx],fontsize=15)

        idx += 1

plt.savefig("../figs/barplotv4.pdf",bbox_inches='tight')
plt.close()




