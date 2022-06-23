import argparse
from lib.loss import ROT_LOSS_NAME2CLASS


def str2opt(arg):
  assert arg in ['SGD', 'Adagrad', 'Adam', 'RMSProp', 'Rprop', 'SGDLars']
  return arg


def str2scheduler(arg):
  assert arg in ['StepLR', 'PolyLR', 'ExpLR', 'SquaredLR']
  return arg


def str2freezer(arg):
  assert arg.lower() in ('none', 'detectron', 'detectron_light')
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


def str2intlist(l):
  return [int(i) for i in l.split(',')]


def str2floatlist(l):
  return [float(i) for i in l.split(',')]


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


arg_lists = []
parser = argparse.ArgumentParser()


pipeline_arg = add_argument_group('Pipeline')
pipeline_arg.add_argument('--pipeline', type=str, default='MaskRCNN', help='Pipeline name')

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--weights', type=str, default='None', help='Saved weights to load')
net_arg.add_argument(
    '--pretrained_weights', type=str, default='None', help='Pretrained weights to load')
net_arg.add_argument(
    '--dilations', type=str2intlist, default='1,1,1,1',
    help='Dilations used for ResNet or DenseNet')

# Detection arguments
net_arg.add_argument(
    '--backbone_model', type=str, default='ResNet34', help='Backbone model name')
net_arg.add_argument(
    '--upsample_feat_size', type=int, default=128, help='Feature size of dense upsample layer')
net_arg.add_argument(
    '--proposal_feat_size', type=int, default=128, help='Feature size of proposal layer')
net_arg.add_argument(
    '--ref_classification_feat_size', type=int, default=2048,
    help='Feature size of classification feature size')
net_arg.add_argument(
    '--mask_feat_size', type=int, default=256,
    help='Feature size of mask feature size')
# TODO(jgwak): make anchor scales dataset-specific.
net_arg.add_argument(
    '--rpn_anchor_scales', type=str2intlist, default='8,16,32,64',
    help='Length of square anchor side in voxels')
net_arg.add_argument(
    '--rpn_anchor_ratios',
    type=str2floatlist,
    default="""0.25,0.25,4.0,
    0.25,4.0,0.25,
    0.25,4.0,4.0,
    4.0,0.25,0.25,
    4.0,0.25,4.0,
    4.0,4.0,0.25,
    1.0,1.0,1.0""",
    help='Ratios of anchors at each cell')
net_arg.add_argument('--rpn_rot_nms', type=str2bool, default=True)
net_arg.add_argument('--detection_rot_nms', type=str2bool, default=False)
net_arg.add_argument('--rpn_aggregate_overlap', type=str2bool, default=False)
net_arg.add_argument('--detection_aggregate_overlap', type=str2bool, default=True)
net_arg.add_argument('--rpn_pre_nms_limit', type=int, default=6000, help='ROIs kept before NMS')
net_arg.add_argument(
    '--rpn_pre_nms_min_confidence', type=float, default=0., help='ROIS kept before NMS')
net_arg.add_argument('--rpn_num_proposals_training', type=int, default=2000,
                     help='ROIs kept after region proposal NMS during training')
net_arg.add_argument('--rpn_num_proposals_inference', type=int, default=1000,
                     help='ROIs kept after region proposal NMS during inference')
net_arg.add_argument('--roi_num_proposals_training', type=int, default=200,
                     help='ROIs kept for second stage region refinement and instance segmentation')
net_arg.add_argument('--roi_positive_ratio_training', type=float, default=0.33,
                     help='Ratio of positive ROIS to train second stage')

net_arg.add_argument(
    '--rpn_nms_threshold', type=float, default=0.35, help='NMS threshold to filter RPN proposals')
# TODO(jgwak): make bounding box std dataset-specific.
net_arg.add_argument(
    '--rpn_bbox_std', type=str2floatlist, default='0.15,0.15,0.15,0.3,0.3,0.3',
    help='Bounding box refinement standard deviation for RPN')
# TODO(jgwak): remove max_ptc_size.
net_arg.add_argument(
    '--max_ptc_size', type=str2intlist, default='512,512,256', help='Maximum bounding box size')
net_arg.add_argument(
    '--rpn_strides', type=str2intlist, default='4,8,16,32', help='Expected RPN strides')
net_arg.add_argument(
    '--rpn_train_anchors_per_batch', type=int, default=256,
    help='How many anchors per batch to use for RPN training')
net_arg.add_argument(
    '--load_sparse_gt_data', type=str2bool, default=False,
    help='Whether to load sparse anchor and regression target in the dataloader')
net_arg.add_argument(
    '--rpn_match_negative_iou_threshold', type=float, default=0.2,
    help='Anchor overlap below this threshold will be considered as negative match in the RPN.')
net_arg.add_argument(
    '--rpn_match_positive_iou_threshold', type=float, default=0.35,
    help='Anchor overlap above this threshold will be considered as positive match in the RPN.')
net_arg.add_argument(
    '--detection_match_positive_iou_threshold', type=float, default=0.25,
    help='Proposal overlap above this threshold will be considered as positive match in the '
         'refinement layer.')
net_arg.add_argument(
    '--fpn_max_scale', type=int, default=96,
    help='Maximum scale of fpn boxes. (equivalent of 224 in FPN paper eq (1))')
net_arg.add_argument(
    '--fpn_base_level', type=int, default=3,
    help='Base level of fpn boxes. (equivalent of $k_0$ in FPN paper eq (1))')
net_arg.add_argument('--sfpn_min_confidence', type=float, default=0.3,
                     help='Minimum probability value to upsample sparse feature on.')
net_arg.add_argument('--visualize_min_confidence', type=float, default=0.3,
                     help='Minimum probability value to visualize bounding boxes.')
net_arg.add_argument('--save_min_confidence', type=float, default=0.3,
                     help='Minimum probability value to visualize bounding boxes.')
net_arg.add_argument('--post_nms_min_confidence', type=float, default=0.0,
                     help='Minimum probability value to filter bounding boxes.')
net_arg.add_argument('--detection_min_confidence', type=float, default=0.3,
                     help='Minimum probability value to accept a detected instance.')
net_arg.add_argument('--detection_max_instances', type=int, default=100,
                     help='Max number of final detections')
net_arg.add_argument('--detection_nms_threshold', type=float, default=0.35,
                     help='NMS threshold for detection')
net_arg.add_argument('--mask_min_confidence', type=float, default=0.5,
                     help='Minimum probability value to accept a detected instance.')
net_arg.add_argument('--mask_nms_threshold', type=float, default=1.0,
                     help='Minimum probability value to accept a detected instance.')
net_arg.add_argument('--mask_class_nms', type=str2bool, default=False)
net_arg.add_argument(
    '--refinement_roialign_poolsize', type=int, default=7,
    help='Output size of ROIAlign layer used for bounding box classification and alignment.')
net_arg.add_argument(
    '--mask_roialign_poolsize', type=int, default=14,
    help='Output size of ROIAlign layer used for bounding box classification and alignment.')
net_arg.add_argument(
    '--roialign_align_corners', type=str2bool, default=False,
    help='Whether to consider the extrema of the sampling grid as center of the corner voxel.')
net_arg.add_argument('--sfpn_classification_loss', type=str, default='balanced',
                     choices=('ce', 'focal', 'balanced'))
net_arg.add_argument('--rpn_semantic_loss', type=str, default='ce',
                     choices=('ce', 'focal', 'balanced'))
net_arg.add_argument('--rpn_classification_loss', type=str, default='balanced',
                     choices=('ce', 'focal', 'balanced'))
net_arg.add_argument('--ref_classification_loss', type=str, default='ce',
                     choices=('ce', 'focal', 'balanced'))
net_arg.add_argument('--detection_nms_score', type=str, default='objsem',
                     choices=('obj', 'sem', 'objsem'))
net_arg.add_argument('--detection_ap_score', type=str, default='objsem',
                     choices=('obj', 'sem', 'objsem'))


# Meanfield arguments
net_arg.add_argument(
    '--meanfield_iterations', type=int, default=10, help='Number of meanfield iterations')
net_arg.add_argument('--crf_spatial_sigma', default=1, type=int, help='Trilateral spatial sigma')
net_arg.add_argument(
    '--crf_chromatic_sigma', default=12, type=int, help='Trilateral chromatic sigma')
net_arg.add_argument('--normalize_bbox', default=True, type=str2bool)

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--lr', type=float, default=1e-1)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--param_histogram_freq', type=int, default=100)
opt_arg.add_argument('--save_param_histogram', type=str2bool, default=False)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.02)

# Scheduler
opt_arg.add_argument('--scheduler', type=str2scheduler, default='StepLR')
opt_arg.add_argument('--freezer', type=str2freezer, default='None')
opt_arg.add_argument('--max_iter', type=int, default=6e4)
opt_arg.add_argument('--step_size', type=int, default=2e4)
opt_arg.add_argument('--step_gamma', type=float, default=0.1)
opt_arg.add_argument('--poly_power', type=float, default=0.9)
opt_arg.add_argument('--exp_gamma', type=float, default=0.9)
opt_arg.add_argument('--exp_step_size', type=float, default=1000)

# Directories
dir_arg = add_argument_group('Directories')
dir_arg.add_argument('--log_dir', type=str, default='outputs/default')
dir_arg.add_argument('--data_dir', type=str, default='data')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='SynthiaDataset')
data_arg.add_argument('--temporal_dilation', type=int, default=10)
data_arg.add_argument('--temporal_numseq', type=int, default=5)
data_arg.add_argument('--temporal_absolute_height', type=str2bool, default=False)
data_arg.add_argument('--point_lim', type=int, default=-1)
data_arg.add_argument('--pre_point_lim', type=int, default=-1)
data_arg.add_argument('--batch_size', type=int, default=4)
data_arg.add_argument('--val_batch_size', type=int, default=1)
data_arg.add_argument('--test_batch_size', type=int, default=1)
data_arg.add_argument('--cache_data', type=str2bool, default=False)
data_arg.add_argument(
    '--threads', type=int, default=4, help='num threads for train/test dataloader')
data_arg.add_argument('--val_threads', type=int, default=1, help='num threads for val dataloader')
data_arg.add_argument('--ignore_label', type=int, default=-1)
data_arg.add_argument('--elastic_distortion', type=str2bool, default=True)
data_arg.add_argument('--return_transformation', type=str2bool, default=False)
data_arg.add_argument('--ignore_duplicate_class', type=str2bool, default=False)
data_arg.add_argument('--partial_crop', type=float, default=0.)
data_arg.add_argument('--train_limit_numpoints', type=int, default=0)
data_arg.add_argument('--cache_anchors', type=str2bool, default=True)
data_arg.add_argument('--preload_anchor_data', type=str2bool, default=True)
data_arg.add_argument('--normalize_rotation', type=str2bool, default=False)
data_arg.add_argument('--normalize_rotation2', type=str2bool, default=False)

# Point Cloud Dataset
data_arg.add_argument(
    '--stanford3d_path',
    type=str,
    default='/cvgl2/u/jgwak/Datasets/stanford3d',
    help='Stanford dataset root dir')

data_arg.add_argument(
    '--scannet_path',
    type=str,
    default='/cvgl2/u/jgwak/Datasets/scannet_instance',
    help='Scannet online voxelization dataset root dir')

data_arg.add_argument(
    '--scannet_votenetrgb_path',
    type=str,
    default='/cvgl2/u/jgwak/Datasets/scannet2_instance_rgb',
    help='Scannet online voxelization dataset root dir')

# Point Cloud Dataset
data_arg.add_argument(
    '--scannet_votenet_path',
    type=str,
    default='/cvgl2/u/jgwak/Datasets/scannet2_instance',
    help='Scannet online voxelization dataset root dir')

data_arg.add_argument(
    '--scannet_alignment_path',
    type=str,
    default='/cvgl2/u/jgwak/Datasets/scannet_raw/scans/%s/%s.txt',
    help='Scannet alignment file path')

data_arg.add_argument(
    '--jrdb_path',
    type=str,
    default='/cvgl2/u/jgwak/Datasets/jrdb_d20_n15',
    help='SUNRGBD dataset root dir')

data_arg.add_argument(
    '--sunrgbd_path',
    type=str,
    default='/cvgl2/u/jgwak/Datasets/sunrgbd/detection',
    help='SUNRGBD dataset root dir')

data_arg.add_argument(
    '--synthia_path',
    type=str,
    default='/cvgl2/u/cornman/Datasets/Synthia/synthia-processed',
    help='Synthia pre-processed pointcloud dataset root dir')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--max_ngpu', type=int, default=4)
train_arg.add_argument('--skip_empty_boxes', type=str2bool, default=False)
train_arg.add_argument('--force_proposal_match', type=str2bool, default=False)
train_arg.add_argument('--rpn_rotation_overlap', type=str2bool, default=False)
train_arg.add_argument('--ref_rotation_overlap', type=str2bool, default=False)
train_arg.add_argument('--rpn_rotation_loss', type=str, default='circular',
                       choices=ROT_LOSS_NAME2CLASS.keys())
train_arg.add_argument('--ref_rotation_loss', type=str, default='circular',
                       choices=ROT_LOSS_NAME2CLASS.keys())
train_arg.add_argument('--num_rotation_bins', type=int, default=12)
train_arg.add_argument('--num_ref_rotation_bins', type=int, default=12)
train_arg.add_argument('--sfpn_class_weight', type=float, default=1.)
train_arg.add_argument('--rpn_class_weight', type=float, default=1.)
train_arg.add_argument('--rpn_semantic_weight', type=float, default=1.)
train_arg.add_argument('--rpn_bbox_weight', type=float, default=0.1)
train_arg.add_argument('--rpn_rotation_weight', type=float, default=1.)
train_arg.add_argument('--ref_class_weight', type=float, default=1.)
train_arg.add_argument('--ref_bbox_weight', type=float, default=1.)
train_arg.add_argument('--ref_rotation_weight', type=float, default=1.)
train_arg.add_argument('--mask_weight', type=float, default=1.)
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--stat_freq', type=int, default=10, help='print frequency')
train_arg.add_argument('--test_stat_freq', type=int, default=100, help='print frequency')
train_arg.add_argument('--save_freq', type=int, default=1000, help='save frequency')
train_arg.add_argument('--heldout_save_freq', type=int, default=-1, help='heldout save frequency')
train_arg.add_argument('--val_freq', type=int, default=2000, help='validation frequency')
train_arg.add_argument(
    '--empty_cache_freq', type=int, default=10, help='Clear pytorch cache frequency')
train_arg.add_argument('--train_phase', type=str, default='train', help='Dataset for training')
train_arg.add_argument('--val_phase', type=str, default='val', help='Dataset for validation')
train_arg.add_argument(
    '--overwrite_weights', type=str2bool, default=True, help='Overwrite checkpoint during training')
train_arg.add_argument(
    '--resume', default='', type=str, help='path to latest checkpoint (default: none)')
train_arg.add_argument(
    '--resume_optimizer',
    default=True,
    type=str2bool,
    help='Use checkpoint optimizer states when resume training')
train_arg.add_argument('--eval_upsample', type=str2bool, default=False)
train_arg.add_argument('--train_skip_rpnonly', type=str2bool, default=False)
train_arg.add_argument('--train_rpnonly', type=str2bool, default=False)
train_arg.add_argument('--train_ref_min_sample_per_batch', type=int, default=1,
                       help='Minimum number of samples to train refinement layer on.')

# Data augmentation
data_aug_arg = add_argument_group('DataAugmentation')
data_aug_arg.add_argument(
    '--use_feat_aug', type=str2bool, default=True, help='Simple feat augmentation')
data_aug_arg.add_argument(
    '--data_aug_color_trans_ratio', type=float, default=0.03, help='Color translation range')
data_aug_arg.add_argument(
    '--data_aug_color_jitter_std', type=float, default=0.03, help='STD of color jitter')
data_aug_arg.add_argument(
    '--data_aug_height_trans_std', type=float, default=0.01, help='STD of height translation')
data_aug_arg.add_argument(
    '--data_aug_height_jitter_std', type=float, default=0.01, help='STD of height jitter')
data_aug_arg.add_argument(
    '--data_aug_normal_jitter_std', type=float, default=0.01, help='STD of normal jitter')
data_aug_arg.add_argument('--normalize_color', type=str2bool, default=True)
data_aug_arg.add_argument('--data_aug_scale_min', type=float, default=0.9)
data_aug_arg.add_argument('--data_aug_scale_max', type=float, default=1.1)
data_aug_arg.add_argument('--temporal_rand_dilation', type=str2bool, default=False)
data_aug_arg.add_argument('--temporal_rand_numseq', type=str2bool, default=False)
data_aug_arg.add_argument('--bbox_loss_wallmask', type=str2bool, default=True)

# Test
test_arg = add_argument_group('Test')
test_arg.add_argument('--visualize', type=str2bool, default=False)
test_arg.add_argument('--visualize_freq', type=int, default=1)
test_arg.add_argument('--test_temporal_average', type=str2bool, default=False)
test_arg.add_argument('--visualize_path', type=str, default='outputs/visualize')
test_arg.add_argument('--test_rotation', type=int, default=-1)
test_arg.add_argument('--save_prediction', type=str2bool, default=False)
test_arg.add_argument('--save_pred_dir', type=str, default='outputs/pred')
test_arg.add_argument('--test_phase', type=str, default='test', help='Dataset for test')
test_arg.add_argument(
    '--test_original_pointcloud',
    type=bool,
    default=False,
    help='Test on the original pointcloud space as given by the dataset.')
test_arg.add_argument('--save_ap_log', type=bool, default=False, help='Save detailed AP log.')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--is_cuda', type=str2bool, default=True)
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--num_gpu', type=str2bool, default=1)
misc_arg.add_argument('--seed', type=int, default=123)
misc_arg.add_argument(
    '--debug', type=str2bool, default=True, help='print out detailed results for debugging')
data_aug_arg.add_argument(
    '--lenient_weight_loading',
    type=str2bool,
    default=False,
    help='Weights with the same size will be loaded')


def get_config():
  config = parser.parse_args()
  return config  # Training settings
