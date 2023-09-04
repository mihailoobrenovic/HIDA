# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default="space", help="For now, space or rgbd")
parser.add_argument("--num_runs",type=int, default=1, help="Number of runs")
parser.add_argument("--color_mode", default="multispectral", help="rgb or multispectral for eurosat")
parser.add_argument("--usecase", default="u_hida", help="u_hida or ss_hida")
parser.add_argument("--idx", type=int, default=-1, help="Index of checkpoint, repetition...")
parser.add_argument("--checkpoint_on", type=int, default=1000, help="After how amny steps to save checkpoints, 0 for every epoch")
parser.add_argument("--num_class",type=int, default=8, help="Number of classes")
parser.add_argument("--processing_type", default="standardization", help="Standarization or normalization")
parser.add_argument("--aug", type=int, default=1, help="Use augmentation (1) or not (0)")
parser.add_argument("--source", default="resisc", help="Source domain")
parser.add_argument("--target", default="eurosat", help="Target domain")
parser.add_argument("--measure", default="cl_loss_s", help="Choose checkpoint based on this measure")
parser.add_argument("--threshold", default="6_25", help="Percentage of target data")
# parser.add_argument("--threshold", default="", help="Percentage of target data")
parser.add_argument("--batch_norm", type=int, default=0, help="Use batch normalization (1) or not (0)")
parser.add_argument("--architecture", default="hida", help="hida or vgg or...")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size (half go to source and half to target)")
parser.add_argument("--exp_name", default="exp_01", help="experiment name")
# parser.add_argument("--exp_name", default="", help="experiment name")
parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
parser.add_argument("--wd_loss_on", default="supervised", help="'supervised' or 'unsupervised'")
parser.add_argument("--last_chckp_only", type=int, default=1, help="Save model only at the end")
parser.add_argument("--with_critic", type=int, default=1, help="True default, False for ablation study")
parser.add_argument("--use_unlabelled",type=int, default=1, help="train domain critic separately")
# parser.add_argument("--use_unlabelled",type=int, default=1, help="train domain critic separately")

parser.add_argument("--train_dir", type=int, default=None, help="Which train dir to use, None if we use then all")

# parser.add_argument("--data_dir", default="0", help="data directory")
# parser.add_argument("--weight_t_loss", type=float, help="Weight of target cl loss")
# parser.add_argument("--csv_file", help="Path of input csv file")
# parser.add_argument("--data_dir_2", default="0", help="additional data directory")
# parser.add_argument("--labels", default="", help="Can be pseudo, semi, pseudo_semi, or pseudo_semi_all")
# parser.add_argument("--to_train", type=int, default=1, help="1-Train, 0-Test")
# parser.add_argument("--sep", type=int, default=0, help="False default, True for ablation study")
# parser.add_argument("--run_fa_mode", default="plot", help="Whether to calculate feature visualisation or to plot it")
# parser.add_argument("--restore", type=int, default=0, help="0-Start training from scratch. 1-Continue training")
# parser.add_argument("--dropout", type=int, default=0, help="Use dropout or not")
# parser.add_argument("--features", default='shared', help="For Pacmap/TSNE, what features to extract for visualisation, options - 'shared', 'separated'")
# parser.add_argument("--folder", default='test', help="A folder on which to evaluate. Usually 'test' or 'validation'.")


args = parser.parse_args()

dataset = args.dataset
num_runs = args.num_runs
color_mode = args.color_mode
usecase = args.usecase
idx = args.idx
checkpoint_on = args.checkpoint_on
num_class = args.num_class
processing_type = args.processing_type
aug = args.aug
source = args.source
target = args.target
measure = args.measure
threshold = args.threshold
batch_norm = args.batch_norm
architecture = args.architecture
batch_size = args.batch_size
exp_name = args.exp_name
epochs = args.epochs
wd_loss_on = args.wd_loss_on
last_chckp_only = args.last_chckp_only
with_critic = args.with_critic
use_unlabelled = args.use_unlabelled

train_dir = args.train_dir

# data_dir = args.data_dir
# weight_t_loss = args.weight_t_loss
# csv_file = args.csv_file
# data_dir_2 = args.data_dir_2
# labels = args.labels
# to_train = args.to_train
# sep = args.sep
# run_fa_mode = args.run_fa_mode
# restore = args.restore
# dropout = args.dropout
# features = args.features
# folder = args.folder

print("dataset:", dataset)
print("number of runs:", num_runs)
print("color mode:", color_mode)
print("usecase:", usecase)
# print("idx", idx)
print("checkpoint_on", checkpoint_on)
print("number of classes:", num_class)
print("processing_type", processing_type)
if aug:
    print("augmentation:", aug)
if source:
    print("source", source)
if target:
    print("target", target)
# if measure:
#     print("measure:", measure)
# if threshold != None:
#     print("Threshold", threshold)
# if batch_norm != None:
#     print("batch norm:", batch_norm)
print("architecture", architecture)
print("batch_size", batch_size)
if exp_name:
    print("experiment name:", exp_name)
print("epochs", epochs)
# print("wd_loss_on", wd_loss_on)
if last_chckp_only != None:
    print("last_chckp_only:", last_chckp_only)
# print("with_critic", with_critic)
# print("Use unlabelled", use_unlabelled)

# if train_dir:
#     print("train dir:", train_dir)

# if data_dir:
#     print("data directory:", data_dir)
# if weight_t_loss:
#     print("target cl loss weight:", weight_t_loss)
# if csv_file != None:
#     print("input csv file", csv_file)
# if data_dir_2:
#     print("additional data directory:", data_dir_2)
# if labels != None:
#     print("labels:", labels)
# if to_train:
#     print("to_train:", to_train)
# print("sep", sep)
# print("run_fa_mode", run_fa_mode)
# print("restore", restore)
# print("dropout", dropout)
# print("features", features)
# print("folder", folder)

