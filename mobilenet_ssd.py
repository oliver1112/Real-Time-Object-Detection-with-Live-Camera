from __future__ import print_function
import sys
sys.path.insert(0, "/home/chenqi-didi/Documents/work/caffe/python")
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess


# Add extra layers on top of a "base" network (e.g. VGGNet or Inception).
def add_extra_layers(net, use_batchnorm=True, lr_mult=1):
    use_relu = True

    # Add additional convolutional layers.
    # 19 x 19
    from_layer = net.keys()[-1]

    # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
    # 10 x 10
    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
                lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2,
                lr_mult=lr_mult)

    # 5 x 5
    from_layer = out_layer
    out_layer = "conv7_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
                lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv7_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
                lr_mult=lr_mult)

    # 3 x 3
    from_layer = out_layer
    out_layer = "conv8_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
                lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv8_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
                lr_mult=lr_mult)

    # 1 x 1
    # from_layer = out_layer
    # out_layer = "conv9_1"
    # ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
    #             lr_mult=lr_mult)

    # from_layer = out_layer
    # out_layer = "conv9_2"
    # ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
    #             lr_mult=lr_mult)
    name = net.keys()[-1]
    net.pool6 = L.Pooling(net[name], pool=P.Pooling.AVE, global_pooling=True)

    return net

# Modify the following parameters accordingly
# change this individually
data_root_dir = "/home/chenqi-didi/data/"
caffe_root = "/home/chenqi-didi/Documents/work/caffe"

current_dir = os.getcwd()

# Set true if you want to start training right after generating all files.
run_soon = True
# Set true if you want to load from most recently saved snapshot.
# Otherwise, we will load from the pretrain_model defined below.
resume_training = True
# If true, Remove old model files.
remove_old_models = True

# The database file for training data. Created by data/${dataset}/create_data.sh
train_data = data_root_dir + "KITTI/lmdb/KITTI_trainval_lmdb"
# The database file for testing data. Created by data/KITTI/create_data.sh
test_data = data_root_dir + "KITTI/lmdb/KITTI_test_lmdb"
# Specify the batch sampler.
# according your image width and height, change this
# this influence accuracy of bbox
resize_width = 414
resize_height = 125
resize = "{}x{}".format(resize_width, resize_height)
batch_sampler = [
        {
                'sampler': {
                        },
                'max_trials': 1,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.1,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.3,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.5,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.7,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.9,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'max_jaccard_overlap': 1.0,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        ]
train_transform_param = {
        'mirror': True,
        'mean_value': [104, 117, 123],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [
                        P.Resize.LINEAR,
                        P.Resize.AREA,
                        P.Resize.NEAREST,
                        P.Resize.CUBIC,
                        P.Resize.LANCZOS4,
                        ],
                },
        'distort_param': {
                'brightness_prob': 0.5,
                'brightness_delta': 32,
                'contrast_prob': 0.5,
                'contrast_lower': 0.5,
                'contrast_upper': 1.5,
                'hue_prob': 0.5,
                'hue_delta': 18,
                'saturation_prob': 0.5,
                'saturation_lower': 0.5,
                'saturation_upper': 1.5,
                'random_order_prob': 0.0,
                },
        'expand_param': {
                'prob': 0.5,
                'max_expand_ratio': 4.0,
                },
        'emit_constraint': {
            'emit_type': caffe_pb2.EmitConstraint.CENTER,
            }
        }
test_transform_param = {
        'mean_value': [104, 117, 123],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [P.Resize.LINEAR],
                },
        }

# If true, use batch norm for all newly added layers.
# Currently only the non batch norm version has been tested.
use_batchnorm = False
lr_mult = 1
# Use different initial learning rate.
if use_batchnorm:
    base_lr = 0.0004
else:
    # A learning rate for batch_size = 1, num_gpus = 1.
    base_lr = 0.000004

# Modify the job name if you want.
job_name = "SSD_{}".format(resize)
# The name of the model. Modify it if you want.
model_name = "VGG_KITTI_{}".format(job_name)

# Directory which stores the model .prototxt file.
save_dir = "models/VGGNet/KITTI/{}".format(job_name)
# Directory which stores the snapshot of models.
snapshot_dir = "models/VGGNet/KITTI/{}".format(job_name)
# Directory which stores the job script and log file.
job_dir = "jobs/VGGNet/KITTI/{}".format(job_name)
# Directory which stores the detection results.
output_result_dir = "{}/data/KITTI/results/{}/Main".format(os.environ['HOME'], job_name)

# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
# snapshot prefix.
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
# job script path.
job_file = "{}/{}.sh".format(job_dir, model_name)

# Stores the test image names and sizes. Created by data/KITTI/create_list.sh
name_size_file = "{}/data/test_name_size.txt".format(current_dir)
# The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
# pretrain_model = "models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel"
pretrain_model = "models/VGGNet/VGG16.v2.caffemodel"
# Stores LabelMapItem.
label_map_file = "{}/data/labelmap_kitti.prototxt".format(current_dir)

# MultiBoxLoss parameters.
# **This line should also changed into your dataset**
num_classes = 10
share_location = True
background_label_id = 9
train_on_diff_gt = True
normalization_mode = P.Loss.VALID
code_type = P.PriorBox.CENTER_SIZE
ignore_cross_boundary_bbox = False
mining_type = P.MultiBoxLoss.MAX_NEGATIVE
neg_pos_ratio = 3.
loc_weight = (neg_pos_ratio + 1.) / 4.
multibox_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'loc_weight': loc_weight,
    'num_classes': num_classes,
    'share_location': share_location,
    'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'overlap_threshold': 0.5,
    'use_prior_for_matching': True,
    'background_label_id': background_label_id,
    'use_difficult_gt': train_on_diff_gt,
    'mining_type': mining_type,
    'neg_pos_ratio': neg_pos_ratio,
    'neg_overlap': 0.5,
    'code_type': code_type,
    'ignore_cross_boundary_bbox': ignore_cross_boundary_bbox,
    }
loss_param = {
    'normalization': normalization_mode,
    }

# parameters for generating detection output.
det_out_param = {
    'num_classes': num_classes,
    'share_location': share_location,
    'background_label_id': background_label_id,
    'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
    'save_output_param': {
        'output_directory': output_result_dir,
        'output_name_prefix': "comp4_det_test_",
        'output_format': "VOC",
        'label_map_file': label_map_file,
        'name_size_file': name_size_file,
        'num_test_image': num_test_image,
        },
    'keep_top_k': 200,
    'confidence_threshold': 0.01,
    'code_type': code_type,
    }

# parameters for evaluating detection results.
det_eval_param = {
    'num_classes': num_classes,
    'background_label_id': background_label_id,
    'overlap_threshold': 0.5,
    'evaluate_difficult_gt': False,
    'name_size_file': name_size_file,
    }

# Create the MultiBoxLossLayer.
name = "mbox_loss"
mbox_layers.append(net.label)
net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
                           loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                           propagate_down=[True, True, False, False])

with open(train_net_file, 'w') as f:
    print('name: "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)
shutil.copy(train_net_file, job_dir)

# Create test net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
                                               train=False, output_label=True, label_map_file=label_map_file,
                                               transform_param=test_transform_param)

VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
           dropout=False)

add_extra_layers(net, use_batchnorm, lr_mult=lr_mult)

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
                                 use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
                                 aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
                                 num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
                                 prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)




for file in os.listdir(snapshot_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format(model_name))[1])
    if iter > max_iter:
      max_iter = iter

train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
if resume_training:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

if remove_old_models:
    # Remove any snapshots smaller than max_iter.
    for file in os.listdir(snapshot_dir):
        if file.endswith(".solverstate"):
            basename = os.path.splitext(file)[0]
            iter = int(basename.split("{}_iter_".format(model_name))[1])
            if max_iter > iter:
                os.remove("{}/{}".format(snapshot_dir, file))
        if file.endswith(".caffemodel"):
            basename = os.path.splitext(file)[0]
            iter = int(basename.split("{}_iter_".format(model_name))[1])
            if max_iter > iter:
                os.remove("{}/{}".format(snapshot_dir, file))


solver_prototxt_path = 'path/to/solver.prototxt'
pretrained_model_path = 'path/to/deploy_mobilessd.caffemodel'

# Initialize the solver
solver = caffe.get_solver(solver_prototxt_path)

# Load the pre-trained model weights
solver.net.copy_from(pretrained_model_path)

# Specify the number of iterations for fine-tuning
niter = 50000  # Adjust based on your needs
display_interval = 10  # Interval for displaying training loss

# Lists to store the values of loss and mAP for plotting
train_loss = []
test_mAP = []

# Fine-tuning loop
for it in range(niter):
    solver.step(1)  # Single step of training

    if it % display_interval == 0:
        loss = solver.net.blobs['loss'].data
        print(f'Iteration {it}, loss = {loss}')
        train_loss.append(loss)

        # Optionally, calculate mAP on validation set
        # Note: You'll need a separate method to calculate mAP
        # mAP = calculate_mAP(validation_data, solver.net)
        # test_mAP.append(mAP)

# Plotting
plt.figure()
plt.plot(np.arange(len(train_loss)) * display_interval, train_loss, label='Training Loss')
# plt.plot(np.arange(len(test_mAP)) * display_interval, test_mAP, label='Validation mAP')
plt.xlabel('Iterations')
plt.ylabel('Loss/mAP')
plt.title('Training Progress')
plt.legend()
plt.show()

# Save the fine-tuned model
fine_tuned_model_path = 'path/to/save/fine_tuned_mobilessd_kitti.caffemodel'
solver.net.save(fine_tuned_model_path)

print(f'Fine-tuning completed. Model saved to {fine_tuned_model_path}')
