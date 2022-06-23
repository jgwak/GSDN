import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Times new roman'
rcParams['font.size'] = 17

COLOR_MAP_RGB = (
    (174., 199., 232.),
    (152., 223., 138.),
    (31., 119., 180.),
    (255., 187., 120.),
    (188., 189., 34.),
    (140., 86., 75.),
    (255., 152., 150.),
    (214., 39., 40.),
    (197., 176., 213.),
    (148., 103., 189.),
    (196., 156., 148.),
    (23., 190., 207.),
    (247., 182., 210.),
    (66., 188., 102.),
    (219., 219., 141.),
    (140., 57., 197.),
    (202., 185., 52.),
    (51., 176., 203.),
    (200., 54., 131.),
    (92., 193., 61.),
    (78., 71., 183.),
    (172., 114., 82.),
    (255., 127., 14.),
    (91., 163., 138.),
    (153., 98., 156.),
    (140., 153., 101.),
    (158., 218., 229.),
    (100., 125., 154.),
    (178., 127., 135.),
    (146., 111., 194.),
    (44., 160., 44.),
    (112., 128., 144.),
    (96., 207., 209.),
    (227., 119., 194.),
    (213., 92., 176.),
    (94., 106., 211.),
    (82., 84., 163.),
    (100., 85., 144.),
)
LINE_STYLES = ('-', '--', '--', '-', '-', '-', '-', '--', '-', '--', '--', '-', '--', '-', '--',
               '-.', '-.', '--')
CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
           'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
           'bathtub', 'otherfurniture')


def draw_ap(key, title, with_legend):
  fig, ax = plt.subplots()
  for i, class_name in enumerate(CLASSES):
    rec, prec = aps[key][i]
    rec = np.concatenate((rec, [rec[-1], 1]))
    prec = np.concatenate((prec, [0, 0]))
    line = ax.plot(rec, prec, label=class_name, linestyle=LINE_STYLES[i % len(LINE_STYLES)])
    line[0].set_color(np.array(COLOR_MAP_RGB[i]) / 255)
  ax.set_ylabel('precision')
  ax.set_xlabel('recall')
  ax.set_title(title)
  if with_legend:
    ax.legend(loc='lower right', prop={'size': 12}, labelspacing=0.2, bbox_to_anchor=(1.35, 0.00))
  plt.savefig(f'scannet_{key}.pdf', bbox_inches='tight')
  plt.clf()


aps = np.load('scannet_ap_details.npz', allow_pickle=True)
draw_ap('ap25', 'P/R curve of ScanNetv2 val @ IoU0.25', False)
draw_ap('ap50', 'P/R curve of ScanNetv2 val @ IoU0.5', True)
