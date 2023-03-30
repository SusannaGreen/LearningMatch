import seaborn as sns
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

x = np.random.rand(1000)
y = np.random.rand(1000)

#Creates a Actual Match and Predicted Match plot with residuals
fig, (ax1, ax2) = plt.subplots(2, figsize=(9, 7), sharex=True, height_ratios=[3, 1])
ax1.scatter(x,y, s=2, color='#0096FF')
ax1.axline((0, 0), slope=1, color='k')

sns.residplot(x=x, y=y, color = '#0096FF', scatter_kws={'s': 8}, line_kws={'linewidth':20})

fig.supxlabel('Actual Match')
fig.supylabel('Predicted Match')
plt.savefig('test_plot_10.pdf', dpi=300)