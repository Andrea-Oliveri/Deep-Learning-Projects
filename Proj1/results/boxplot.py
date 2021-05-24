import os
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MultipleLocator


def autosplit(string):
    return string.replace('(', '\n( ').replace(',', ',\n').replace(')', ' )')



files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith(".log")]


names = []
lists = []
for file in files:
    with open(file, "r") as f:
        name, list_ = f.read().splitlines()[:2]
    
        list_ = eval(list_.replace("tensor", "torch.tensor"))
        list_ = [l[0] for l in list_]
        
        names.append(name)
        lists.append(torch.stack(list_))
        

plt.figure(figsize = (8, 5))
plt.boxplot(lists, labels = [autosplit(a) for a in names], vert = False,
            positions = [2,0,1,5,3,4])
plt.xlabel("Digits Comparison Test Accuracy")
plt.ylabel("Model")
plt.xlim(right = 1)
plt.grid()
plt.gca().xaxis.set_major_locator(MultipleLocator(0.05))


plt.suptitle("Distribution of Digits Comparison Test Accuracy of the different models", fontsize=14)
plt.tight_layout()
plt.savefig("boxplots.pdf")