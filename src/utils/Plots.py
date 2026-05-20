import sys
sys.path.append('/home/trukhinmaksim/src')
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

size = (7,5)

# Sample data (replace with your actual data)

def buildDistributionGraphs(group1, group0):
    #np.random.seed(42) # for reproducibility
    plt.figure(figsize=size)
    sns.histplot(group1, color="skyblue", label="Related projects group", kde=True, stat="density", alpha=0.6, bins=60)
    sns.histplot(group0, color="orange", label="Unrelated projects group", kde=True, stat="density", alpha=0.6, bins=60)
    #plt.title('RelatePorjects')
    plt.xlabel('Cosine similarity value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
