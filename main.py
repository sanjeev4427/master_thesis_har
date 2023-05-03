##################################################
# Main script used to commence experiments
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

import argparse
import os
import time
import sys
import numpy as np
import pandas as pd
# from GA_model.GA_activity_wise import ga_activity_wise
from SA_model.find_initial_temperature import find_initial_temperature
# import torch
# from sklearn.metrics import accuracy_score, f1_score

# from sklearn.model_selection import train_test_split
# from GA_model.GA_activity_wise import ga_activity_wise
# from GA_model.decode import decode
# from GA_model.genetic_algo import genetic_algorithm
# from Other_helpful_scripts.graph_activities_gt_val_mod_val import graph_gt_val_mod_val
# from SA_model.evaluation_simulated_annealing import evaluate_sim_ann_on_best
from SA_model.generate_input_data_for_SA import ml_generate_train_data
from SA_model.sim_ann_activity_wise import sim_ann_activity_wise
from SA_model.sim_ann_activity_wise_exp_gt import sim_ann_activity_wise_gt
from misc.close_excel import close_all_excel_files
# from ml_evaluate import ml_evaluate
from ml_validation import ml_validation


from model.train import predict
from model.DeepConvLSTM import DeepConvLSTM

from data_processing.preprocess_data import load_dataset
from data_processing.sliding_window import apply_sliding_window
from model.validation import cross_participant_cv
from model.train import predict


from misc.logging import Logger
from misc.torchutils import seed_torch
from RL.train_reinforce import train_reinforce

"""
PROJECT PARAMETERS:
"""
SAVING_TYPE = 'min'
SAVING_THRESHOLD = 0.2
SAVING_TOLERANCE = 2
SAVING_SKIP_WINDOWS = 5 
SAVING_WINDOW_THRESHOLD = 20

"""
DATASET OPTIONS:
- DATASET:
    - opportunity: Opportunity dataset
    - wetlab: Wetlab dataset
    - rwhar: RealWorld HAR dataset
    - sbhar: SBHAR dataset
    - hhar: HHAR dataset
- PRED_TYPE:
    - Opportunity: 'gestures' or 'locomotion'
    - Wetlab: 'actions' or 'tasks'
- SW_LENGTH: length of sliding window
- SW_UNIT: unit in which length of sliding window is measured
- SW_OVERLAP: overlap ratio between sliding windows (in percent, i.e. 60 = 60%)
- INCLUDE_NULL: boolean whether to include null class in datasets (does not work with opportunity_ordonez dataset)
"""

DATASET = 'wetlab'
PRED_TYPE = 'actions'
SW_LENGTH = 1
SW_UNIT = 'seconds'
SW_OVERLAP = 60
INCLUDE_NULL = True

"""
NETWORK OPTIONS:
- NETWORK: network architecture to be used (e.g. 'deepconvlstm')
- LSTM: boolean whether to employ a lstm after convolution layers
- NB_UNITS_LSTM: number of hidden units in each LSTM layer
- NB_LAYERS_LSTM: number of layers in LSTM
- CONV_BLOCK_TYPE: type of convolution blocks employed ('normal', 'skip' or 'fixup')
- NB_CONV_BLOCKS: number of convolution blocks employed
- NB_FILTERS: number of convolution filters employed in each layer of convolution blocks
- FILTER_WIDTH: width of convolution filters (e.g. 11 = 11x1 filter)
- DILATION: dilation factor employed on convolutions (set 1 for not dilation)
- DROP_PROB: dropout probability in dropout layers
- POOLING: boolean whether to employ a pooling layer after convolution layers
- BATCH_NORM: boolean whether to apply batch normalisation in convolution blocks
- REDUCE_LAYER: boolean whether to employ a reduce layer after convolution layers
- POOL_TYPE: type of pooling employed in pooling layer
- POOL_KERNEL_WIDTH: width of pooling kernel (e.g. 2 = 2x1 pooling kernel)
- REDUCE_LAYER_OUTPUT: size of the output after the reduce layer (i.e. what reduction is to be applied) 
"""

NETWORK = 'deepconvlstm'
NO_LSTM = False
NB_UNITS_LSTM = 128
NB_LAYERS_LSTM = 1
CONV_BLOCK_TYPE = 'normal'
NB_CONV_BLOCKS = 2
NB_FILTERS = 64
FILTER_WIDTH = 11
DILATION = 1
DROP_PROB = 0.5
POOLING = False
BATCH_NORM = False
REDUCE_LAYER = False
POOL_TYPE = 'max'
POOL_KERNEL_WIDTH = 2
REDUCE_LAYER_OUTPUT = 8

"""
TRAINING OPTIONS:
- SEED: random seed which is to be employed
- VALID_EPOCH: which epoch used for evaluation; either 'best' or 'last'
- BATCH_SIZE: size of the batches
- EPOCHS: number of epochs during training
- OPTIMIZER: optimizer to use; either 'rmsprop', 'adadelta' or 'adam'
- LR: learning rate to employ for optimizer
- WEIGHT_DECAY: weight decay to employ for optimizer
- WEIGHTS_INIT: weight initialization method to use to initialize network
- LOSS: loss to use ('cross_entropy')
- SMOOTHING: degree of label smoothing employed if cross-entropy used
- GPU: name of GPU to use (e.g. 'cuda:0')
- WEIGHTED: boolean whether to use weighted loss calculation based on support of each class
- SHUFFLING: boolean whether to use shuffling during training
- ADJ_LR: boolean whether to adjust learning rate if no improvement
- LR_SCHEDULER: type of learning rate scheduler to employ ('step_lr', 'reduce_lr_on_plateau')
- LR_STEP: step size of learning rate scheduler (patience if plateau).
- LR_DECAY: decay factor of learning rate scheduler.
- EARLY_STOPPING: boolean whether to stop the network training early if no improvement 
- ES_PATIENCE: patience (i.e. number of epochs) after which network training is stopped if no improvement
"""

SEED = 1
VALID_EPOCH = 'last'
BATCH_SIZE = 100
EPOCHS = 1
OPTIMIZER = 'adam'
LR = 1e-4
WEIGHT_DECAY = 1e-6
WEIGHTS_INIT = 'xavier_normal'
LOSS = 'cross_entropy'
SMOOTHING = 0.0
GPU = 'cuda:0'
WEIGHTED = True
SHUFFLING = False
ADJ_LR = False
LR_SCHEDULER = 'step_lr'
LR_STEP = 10
LR_DECAY = 0.9
EARLY_STOPPING = False
ES_PATIENCE = 10

"""
LOGGING OPTIONS:
- NAME: name of the experiment; used for logging purposes
- LOGGING: boolean whether to log console outputs in a text file
- PRINT_COUNTS: boolean whether to print the distribution of predicted labels after each epoch 
- VERBOSE: boolean whether to print batchwise results during epochs
- PRINT_FREQ: number of batches after which batchwise results are printed
- SAVE_TEST_PREDICTIONS: boolean whether to save test predictions
- SAVE_MODEL: boolean whether to save the model after last epoch as a checkpoint file
"""

NAME = None 
LOGGING = True
PRINT_COUNTS = False
VERBOSE = False
PRINT_FREQ = 100
SAVE_TEST_PREDICTIONS = False
SAVE_CHECKPOINTS = True
SAVE_ANALYSIS = True

"""
Machine learning training : SA
- INIT_TEMP_ITR: iterations used for the initial temperature calculation 
- MAX_STEP_SIZE_THR: upper limit of window threshold interval
- MAX_STEP_SIZE_SKIP: upper limit of window skip interval
- MAX_STEP_SIZE_TOL: upper limit of window tolerance interval
- ANN_RATE: annealing schedule; how fast the temperature will decrease
- ALGO_NAME: mateheuristic algorithm name 

"""
INIT_TEMP_ITR = 30
MAX_STEP_SIZE_THR = 50
MAX_STEP_SIZE_SKIP = 50
MAX_STEP_SIZE_TOL = 3
ANN_RATE = 0.5
ALGO_NAME = 'SA'

# settings for GA
# define range for input
bounds = [[1, 128], [1, 128], [1,10]]
# define the max iterations to stop getting stuck 
max_iter = 100
# termination iteration
termin_iter = 5
# bits per variable
n_bits = 7
# define the population size
n_pop = 50
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(bounds))




def main(args):
     # apply the chosen random seed to all relevant parts
     seed_torch(args.seed)

     # check if valid prediction type chosen for dataset
     if args.dataset == 'opportunity':
          if args.pred_type != 'gestures' and args.pred_type != 'locomotion':
               print('Did not choose a valid prediction type for Opportunity dataset!')
               exit()
     elif args.dataset == 'wetlab':
          if args.pred_type != 'actions' and args.pred_type != 'tasks':
               print('Did not choose a valid prediction type for Wetlab dataset!')
               exit()

     # parameters used to calculate runtime
     start = time.time()
     log_date = time.strftime('%Y%m%d')
     log_timestamp = time.strftime('%H%M%S')

     # saves logs to a file (standard output redirected)
     if args.logging:
          if args.name:
               sys.stdout = Logger(os.path.join('logs', log_date, log_timestamp, 'log_{}.txt'.format(args.name)))
          else:
               sys.stdout = Logger(os.path.join('logs', log_date, log_timestamp, 'log.txt'))

     print('Applied settings: ')
     print(args)

     ################################################## DATA LOADING ####################################################

     print('Loading data...')
     X, y, nb_classes, class_names, sampling_rate, has_null = \
          load_dataset(dataset=args.dataset,
                         pred_type=args.pred_type,
                         include_null=args.include_null,
                         saving_type=args.saving_type
                         )

     args.sampling_rate = sampling_rate
     args.nb_classes = nb_classes  # Number of classes (activities)
     args.class_names = class_names # activity labels 
     #! changing classes to 9 (pouring = pour cata) for the ML training 
     if args.dataset == 'wetlab':
        label_name = ['null_class', 'cutting', 'inverting', 'peeling', 'pestling',\
                       'pipetting', 'pouring', 'stirring', 'transfer']
        activity_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

     args.nb_classes = len(label_name)       # Number of classes (activities)
     args.class_names = label_name
     args.has_null = has_null

     ############################################# DEEP LEARNING TRAINING #############################################################

     # # re-create full dataset for splitting purposes
     data = np.concatenate((X, (np.array(y)[:, None])), axis=1)

     # custom_net = None
     # custom_loss = None
     # custom_opt = None

     # #leave-one-subject-out cross-validation 
     # trained_net, all_mod_eval_output = cross_participant_cv(data, custom_net, custom_loss, custom_opt, args, log_date, log_timestamp)

     # # calculate time data creation took
     # end = time.time()
     # hours, rem = divmod(end - start, 3600)
     # minutes, seconds = divmod(rem, 60)
     # print("\nFinal time elapsed: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))

     ################################################## Skip-heuristics on GT ####################################################
    # skip heuristic module with training disabled
    # skip_heuristic_gt(data, args) 
     ##### return variables to whoever needs them ####

     
     ################################################ Simulated Annealing ################################
     # seed the pseudorandom number generator #! comment this in when running with 5 different seeds
     np.random.seed(1)

     # settings for SA
     # setting range for hyperparameters to be optimised
     if DATASET == 'wetlab':
          window_threshold = np.array([1,100])
          skip_windows = np.array([1,100])
          tol_value = np.array([0, 10])
          init_temp = 0.01743# for act = 0 
     if DATASET == 'rwhar':
          window_threshold = np.array([1, 100])
          skip_windows = np.array([1, 100])
          tol_value = np.array([0, 10])
          init_temp = 0.083 

     # iterations at every Temperature value (for finding initial temperature )
     n_iter_init_temp = 30
     # define the maximum step size for threshold values
     max_step_size_win_thr = 50
     # define the maximum step size for skip windows
     max_step_size_skip_win = 50
     # define maximum step size for tolerance values
     max_step_size_tol_val = 3
     # annealing rate
     ann_rate_array = 0.5
     algo_name = 'SA'
#--------------- finding initial temperature (activity wise) --------------------------------

     find_initial_temperature(data, args, tol_value,\
                                window_threshold, skip_windows,
                                max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val, n_iter_init_temp)

#-----------------------------------------------------------------------------
     # args.name = args.dataset + '_' + algo_name 
     # SA training
     # sim_ann_activity_wise(args, window_threshold, skip_windows, tol_value, max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val, \
     #                          init_temp, ann_rate_array, log_date, log_timestamp, data)
     
     # taining on gt data (first winsow then tolerance) 
     # sim_ann_activity_wise_gt(args, window_threshold, skip_windows, tol_value, max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val, \
     #                          init_temp, ann_rate_array[0], log_date, log_timestamp, data)
    # validating best settings 
     # args.name = args.dataset + '_' + algo_name 
     # ml_validation(args, data, algo_name, log_date, log_timestamp)

     # evaluate best settings 
     # ml_evaluate(sbj, args, log_dir, algo_name, data, f_one_gt_mod_val_act_wise_lst, filename_best_csv)
     
#     simulated_annealing_all_validation_data(args, window_threshold, skip_windows, max_step_size_win_thr, max_step_size_skip_win,\
#                                              tol_value, init_temp, ann_rate_array, all_val_gt_two_times)

#--------------- evaluation on best parameters --------------------------------
# # Optimised hyper-parameters after training
#      best = np.array([[6, 57]]) 
     # best = np.array([[4, 100, 5]])
     # activity_labels = np.array([7])#, 1, 2, 3, 4, 5, 6, 7, 8, 9])
     # args.name = 'compare results' 
     # for activity in activity_labels:
     #      _, f_one, comp_saved_ratio = evaluate_sim_ann_on_best(DATASET, activity, args, best, all_eval_output)
     # args.name = 'tst_new_loss_fun_case1'
     # graph_gt_val_mod_val(data, args, 0, 0, log_date, log_timestamp)

#         if activity == 0:
#             # f1 score on full data 
#             print(f"f1 score full data: {f_one} \n", 
#                    f"accuracy on full data: {accuracy} \n",
#                    f"recall on full data: {recall} \n",
#                    f"precision on full data: {precision} \n",
#                    f"comp saved ratio on full data: {comp_saved_ratio}")
#-----------------------------------------------------------------------------
   
################################ reinforcement learning ####################################################

#     scores = train_reinforce(args, window_threshold, skip_windows, SAVING_TOLERANCE, all_val_gt_two_times)

################################ Genetic algorithm ####################################################
     # # settings for GA
     # # define range for input
     # bounds = [[1, 128], [1, 128], [1,10]]
     # # define the max iterations to stop getting stuck 
     # max_iter = 100
     # # termination iteration
     # termin_iter = 5
     # # bits per variable
     # n_bits = 7
     # # define the population size
     # n_pop = 50
     # # crossover rate
     # r_cross = 0.9
     # # mutation rate
     # r_mut = 1.0 / (float(n_bits) * len(bounds))
     algo_name = 'GA'
     # tournament selection
     # two-point crossover

     
     # # training with genetic algorithm 
     # args.name = args.dataset + algo_name 
     # ga_activity_wise(args, data,  bounds, n_bits, n_pop, r_cross, r_mut, termin_iter, max_iter, log_date, log_timestamp, data)
     # print('Done!')
    
     # args.name = args.dataset + algo_name 
     # ml_validation(args, data, algo_name, log_date, log_timestamp)

     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Bosch
    parser.add_argument('--saving_type', default=SAVING_TYPE, type=str,
                        help='Type of saving array (min or avg)')
    parser.add_argument('--saving_threshold', default=SAVING_THRESHOLD, type=float,
                        help='Saving threshold applied, i.e. how much percentage of an activtiy must be seen for skip to happen')
    parser.add_argument('--saving_tolerance', default=SAVING_TOLERANCE, type=int,
                        help='Saving threshold applied, i.e. how much percentage of an activtiy must be seen for skip to happen')
    parser.add_argument('--saving_skip_windows', default=SAVING_SKIP_WINDOWS, type=int,
                        help='Saving threshold applied, i.e. how much percentage of an activtiy must be seen for skip to happen')

    # Flags
    parser.add_argument('--save_test_preds', default=SAVE_TEST_PREDICTIONS, action='store_true',
                        help='Flag indicating to save predictions in separate file')
    parser.add_argument('--logging', default=LOGGING, action='store_true',
                        help='Flag indicating to log terminal output into text file')
    parser.add_argument('--verbose', default=VERBOSE, action='store_true',
                        help='Flag indicating to have verbose training output (batchwise)')
    parser.add_argument('--print_counts', default=PRINT_COUNTS, action='store_true',
                        help='Flag indicating to print class distribution of train and validation set after epochs')
    parser.add_argument('--include_null', default=INCLUDE_NULL, action='store_true',
                        help='Flag indicating to include null class (if dataset has one) in training/ prediction')
    parser.add_argument('--batch_norm', default=BATCH_NORM, action='store_true',
                        help='Flag indicating to use batch normalisation after each convolution')
    parser.add_argument('--reduce_layer', default=REDUCE_LAYER, action='store_true',
                        help='Flag indicating to use reduce layer after convolutions')
    parser.add_argument('--weighted', default=WEIGHTED, action='store_true',
                        help='Flag indicating to use weighted loss')
    parser.add_argument('--shuffling', default=WEIGHTED, action='store_true',
                        help='Flag indicating to use shuffling during training')
    parser.add_argument('--pooling', default=POOLING, action='store_true',
                        help='Flag indicating to apply pooling after convolutions')
    parser.add_argument('--adj_lr', default=ADJ_LR, action='store_true',
                        help='Flag indicating to adjust learning rate')
    parser.add_argument('--early_stopping', default=EARLY_STOPPING, action='store_true',
                        help='Flag indicating to employ early stopping')
    parser.add_argument('--save_checkpoints', default=SAVE_CHECKPOINTS, action='store_true',
                        help='Flag indicating to save the trained model as a checkpoint file')
    parser.add_argument('--no_lstm', default=NO_LSTM, action='store_true',
                        help='Flag indicating whether to omit LSTM from architecture')
    parser.add_argument('--save_analysis', default=SAVE_ANALYSIS, action='store_true',
                        help='Flag indicating whether to save analysis results.')

    # Strings
    parser.add_argument('--name', default=NAME, type=str,
                        help='Name of the experiment (visible in logging). Default: None')
    parser.add_argument('-d', '--dataset', default=DATASET, type=str,
                        help='Dataset to be used. Options: rwhar, sbhar, wetlab or hhar. '
                             'Default: rwhar')
    parser.add_argument('-p', '--pred_type', default=PRED_TYPE, type=str,
                        help='(If applicable) prediction type for dataset. See dataset documentation for options. '
                        'Default: gestures')
    parser.add_argument('-n', '--network', default=NETWORK, type=str,
                        help='Network to be used. Options: deepconvlstm. '
                             'Default: deepconvlstm')
    parser.add_argument('-ve', '--valid_epoch', default=VALID_EPOCH, type=str,
                        help='Which epoch to use for evaluation. Options: best, last'
                             'Default: best')
    parser.add_argument('-swu', '--sw_unit', default=SW_UNIT, type=str,
                        help='sliding window unit used. Options: units, seconds.'
                             'Default: seconds')
    parser.add_argument('-wi', '--weights_init', default=WEIGHTS_INIT, type=str,
                        help='weight initialization method used. Options: normal, orthogonal, xavier_uniform, '
                             'xavier_normal, kaiming_uniform, kaiming_normal. '
                             'Default: xavier_normal')
    parser.add_argument('-pt', '--pool_type', default=POOL_TYPE, type=str,
                        help='type of pooling applied. Options: max, average. '
                             'Default: max')
    parser.add_argument('-o', '--optimizer', default=OPTIMIZER, type=str,
                        help='Optimizer to be used. Options: adam, rmsprop, adadelta.'
                             'Default: adam')
    parser.add_argument('-l', '--loss', default=LOSS, type=str,
                        help='Loss to be used. Options: cross_entropy, maxup.'
                             'Default: cross_entropy')
    parser.add_argument('-g', '--gpu', default=GPU, type=str,
                        help='GPU to be used. Default: cuda:1')
    parser.add_argument('-lrs', '--lr_scheduler', default=LR_SCHEDULER, type=str,
                        help='Learning rate scheduler to use. Options: step_lr, reduce_lr_on_plateau. '
                             'Default: step_lr')
    parser.add_argument('-cbt', '--conv_block_type', default=CONV_BLOCK_TYPE, type=str,
                        help='type of convolution blocks used. Options: normal, skip, fixup.'
                             'Default: normal')

    # Integers
    parser.add_argument('-pf', '--print_freq', default=PRINT_FREQ, type=int,
                        help='If verbose, frequency of which is printed (batches).'
                             'Default: 100')
    parser.add_argument('-s', '--seed', default=SEED, type=int,
                        help='Seed to be employed. '
                             'Default: 1')
    parser.add_argument('-e', '--epochs', default=EPOCHS, type=int,
                        help='No. epochs to use during training.'
                             'Default: 30')
    parser.add_argument('-bs', '--batch_size', default=BATCH_SIZE, type=int,
                        help='Batch size to use during training.'
                             'Default: 100')
    parser.add_argument('-rlo', '--reduce_layer_output', default=REDUCE_LAYER_OUTPUT, type=int,
                        help='Size of reduce layer output. '
                             'Default: 8')
    parser.add_argument('-pkw', '--pool_kernel_width', default=POOL_KERNEL_WIDTH, type=int,
                        help='Size of pooling kernel.'
                             'Default: 2')
    parser.add_argument('-fw', '--filter_width', default=FILTER_WIDTH, type=int,
                        help='Filter size (convolutions).'
                             'Default: 11')
    parser.add_argument('-esp', '--es_patience', default=ES_PATIENCE, type=int,
                        help='Patience for early stopping (e.g. after 10 epochs of no improvement). '
                             'Default: 10')
    parser.add_argument('-nbul', '--nb_units_lstm', default=NB_UNITS_LSTM, type=int,
                        help='Number of units within each LSTM layer. '
                             'Default: 128')
    parser.add_argument('-nbll', '--nb_layers_lstm', default=NB_LAYERS_LSTM, type=int,
                        help='Number of layers in LSTM.'
                             'Default: 1')
    parser.add_argument('-nbcb', '--nb_conv_blocks', default=NB_CONV_BLOCKS, type=int,
                        help='Number of convolution blocks. '
                             'Default: 2')
    parser.add_argument('-nbf', '--nb_filters', default=NB_FILTERS, type=int,
                        help='Number of convolution filters.'
                             'Default: 64')
    parser.add_argument('-dl', '--dilation', default=DILATION, type=int,
                        help='Dilation applied in convolution filters.'
                             'Default: 1')
    parser.add_argument('-lrss', '--lr_step', default=LR_STEP, type=int,
                        help='Period of learning rate decay (patience if plateau scheduler).'
                             'Default: 10')

    # Floats
    parser.add_argument('-swl', '--sw_length', default=SW_LENGTH, type=float,
                        help='Length of sliding window. '
                             'Default: 1')
    parser.add_argument('-swo', '--sw_overlap', default=SW_OVERLAP, type=int,
                        help='Overlap employed between sliding windows.'
                             'Default: 60')
    parser.add_argument('-dp', '--drop_prob', default=DROP_PROB, type=float,
                        help='Dropout probability.'
                             'Default 0.5')
    parser.add_argument('-sm', '--smoothing', default=SMOOTHING, type=float,
                        help='Degree of label smoothing.'
                             'Default: 0.0')
    parser.add_argument('-lr', '--learning_rate', default=LR, type=float,
                        help='Learning rate to be used. '
                             'Default: 1e-04')
    parser.add_argument('-wd', '--weight_decay', default=WEIGHT_DECAY, type=float,
                        help='Weight decay to be used. '
                             'Default: 1e-06')
    parser.add_argument('-lrsd', '--lr_decay', default=LR_DECAY, type=float,
                        help='Multiplicative factor of learning rate decay. '
                             'Default: 0.9')

    args = parser.parse_args()

    main(args)
