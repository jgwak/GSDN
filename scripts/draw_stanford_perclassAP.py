import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Times new roman'
rcParams['font.size'] = 17

CLASSES = ('table', 'chair', 'sofa', 'bookcase', 'board')


def draw_ap(key, title):
  fig, ax = plt.subplots()
  for i, class_name in enumerate(CLASSES):
    rec, prec = aps[key][i]
    rec = np.concatenate((rec, [rec[-1], 1]))
    prec = np.concatenate((prec, [0, 0]))
    ax.plot(rec, prec, label=class_name)
  ax.set_ylabel('precision')
  ax.set_xlabel('recall')
  ax.set_title(title)
  ax.legend(loc='lower right', prop={'size': 12}, labelspacing=0.2)
  plt.savefig(f'stanford_{key}.pdf')
  plt.show()
  plt.clf()


aps = np.load('ap_details.npz', allow_pickle=True)
draw_ap('ap25', 'P/R curve of S3DIS building 5 @ IoU0.25')
draw_ap('ap50', 'P/R curve of S3DIS building 5 @ IoU0.5')
