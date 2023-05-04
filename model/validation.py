##################################################
# All functions related to validating a model
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

from data_processing.data_analysis import plot_sensordata_and_labels
from data_processing.sliding_window import apply_sliding_window
from misc.osutils import mkdir_if_missing
from model.DeepConvLSTM import DeepConvLSTM
from model.evaluate import evaluate_participant_scores, evaluate_mod_participant_scores
from model.train import train, init_optimizer, init_loss, init_scheduler
# from model.simulated_annealing import simulated_annealing


def cross_participant_cv(data, custom_net, custom_loss, custom_opt, args, log_date, log_timestamp):
    """
    Method to apply cross-participant cross-validation (also known as leave-one-subject-out cross-validation).

    :param data: numpy array
        Data used for applying cross-validation
    :param custom_net: pytorch model
        Custom network object
    :param custom_loss: loss object
        Custom loss object
    :param custom_opt: optimizer object
        Custom optimizer object
    :param args: dict
        Args object containing all relevant hyperparameters and settings
    :param log_date: string
        Date information needed for saving
    :param log_timestamp: string
        Timestamp information needed for saving
    :return pytorch model
        Trained network
    """

    print('\nCALCULATING CROSS-PARTICIPANT SCORES USING LOSO CV.\n')
    cp_scores = np.zeros((4, args.nb_classes, int(np.max(data[:, 0]) + 1)))
    mod_cp_scores = np.zeros((4, args.nb_classes, int(np.max(data[:, 0]) + 1)))
    train_val_gap = np.zeros((4, int(np.max(data[:, 0]) + 1)))
    mod_train_val_gap = np.zeros((4, int(np.max(data[:, 0]) + 1)))
    cp_savings = np.zeros((4, args.nb_classes, int(np.max(data[:, 0]) + 1)))
    all_eval_output = None
    all_mod_eval_output = None
    orig_lr = args.learning_rate
    log_dir = os.path.join('logs', log_date, log_timestamp)

    for i, sbj in enumerate(np.unique(data[:, 0])):
    # for sbj in [0]:
        print('\n VALIDATING FOR SUBJECT {0} OF {1}'.format(int(sbj) + 1, int(np.max(data[:, 0])) + 1))
        train_data = data[data[:, 0] != sbj]
        val_data = data[data[:, 0] == sbj]
        #train_data = data[data[:, 0] == 0]
        #val_data = data[data[:, 0] == 1]
        args.learning_rate = orig_lr

        # calculate concurrent windows
        curr_label = None
        curr_window = 0
        windows = []
        for sbj_id in np.unique(train_data[:, 0]):
            sbj_label = train_data[train_data[:, 0] == sbj_id][:, -1]
            for label in sbj_label:
                if label != curr_label and curr_label is not None:
                    windows.append([curr_label, curr_window / args.sampling_rate, curr_window])
                    curr_label = label
                    curr_window = 1
                elif label == curr_label:
                    curr_window += 1
                else:
                    curr_label = label
                    curr_window += 1
        windows = np.array(windows)

        # calculate savings array
        saving_array = np.zeros(args.nb_classes)
        for label in range(args.nb_classes):
            label_windows = windows[windows[:, 0] == label]
            if label_windows.size != 0:
                if args.saving_type == 'mean':
                    saving_array[int(label)] = np.mean(label_windows[:, 1].astype(float))
                elif args.saving_type == 'median':
                    saving_array[int(label)] = np.median(label_windows[:, 1].astype(float))
                elif args.saving_type == 'min':
                    saving_array[int(label)] = np.min(label_windows[:, 1].astype(float))
                elif args.saving_type == 'max':
                    saving_array[int(label)] = np.max(label_windows[:, 1].astype(float))
                elif args.saving_type == 'first_quartile':
                    saving_array[int(label)] = np.percentile(label_windows[:, 1].astype(float), 25)

        args.saving_array = saving_array
        print(saving_array)

        # Sensor data is segmented using a sliding window mechanism
        X_train, y_train = apply_sliding_window(train_data[:, :-1], train_data[:, -1],
                                                sliding_window_size=args.sw_length,
                                                unit=args.sw_unit,
                                                sampling_rate=args.sampling_rate,
                                                sliding_window_overlap=args.sw_overlap,
                                                )

        X_val, y_val = apply_sliding_window(val_data[:, :-1], val_data[:, -1],
                                            sliding_window_size=args.sw_length,
                                            unit=args.sw_unit,
                                            sampling_rate=args.sampling_rate,
                                            sliding_window_overlap=args.sw_overlap,
                                            )

        X_train, X_val = X_train[:, :, 1:], X_val[:, :, 1:]

        args.window_size = X_train.shape[1]
        args.nb_channels = X_train.shape[2]

        # network initialization
        if args.network == 'deepconvlstm':
            net = DeepConvLSTM(config=vars(args))
        elif args.network == 'custom':
            net = custom_net
        else:
            print("Did not provide a valid network name!")

        # optimizer initialization
        if args.optimizer != 'custom':
            opt = init_optimizer(net, args)
        else:
            opt = custom_opt

        # optimizer initialization
        if args.loss != 'custom':
            loss = init_loss(args)
        else:
            loss = custom_loss

        # lr scheduler initialization
        if args.adj_lr:
            print('Adjusting learning rate according to scheduler: ' + args.lr_scheduler)
            scheduler = init_scheduler(opt, args)
        else:
            scheduler = None
       
        #! here mod_val_output and val_output are same as no modification is made in train()
        net, checkpoint, val_output, mod_val_output, train_output, comp_saved, data_saved, comp_windows, data_windows = \
            train(X_train, y_train, X_val, y_val,
                  network=net, optimizer=opt, loss=loss, lr_scheduler=scheduler, config=vars(args),
                  log_date=log_date, log_timestamp=log_timestamp)
        
        mkdir_if_missing(log_dir)
        #saving validation predictions for eaach subject
        print(f'Saving validation predictions for subj {int(sbj) + 1}')
        val_pred_sbj = pd.DataFrame(val_output)
        val_pred_sbj.to_csv(os.path.join(log_dir, f'val_pred_sbj_{int(sbj)+1}.csv'), index=None, header = None)

        #saving training predictions for eaach subject
        print(f'Saving training predictions for subj {int(sbj) + 1}')
        train_pred_sbj = pd.DataFrame(train_output)
        train_pred_sbj.to_csv(os.path.join(log_dir, f'train_pred_sbj_{int(sbj)+1}.csv'), index=None, header = None)

        if args.save_checkpoints:
            mkdir_if_missing(log_dir)
            print('Saving checkpoint...')
            if args.valid_epoch == 'last':
                if args.name:
                    c_name = os.path.join(log_dir, "checkpoint_last_{}_{}.pth".format(str(sbj), str(args.name)))
                else:
                    c_name = os.path.join(log_dir, "checkpoint_last_{}.pth".format(str(sbj)))
            else:
                if args.name:
                    c_name = os.path.join(log_dir, "checkpoint_best_{}_{}.pth".format(str(sbj), str(args.name)))
                else:
                    c_name = os.path.join(log_dir, "checkpoint_best_{}.pth".format(str(sbj)))
            torch.save(checkpoint, c_name)

        #saving validation predictions for eaach subject
        print(f'Saving validation predictions for subj {int(sbj) + 1}')
        val_pred_sbj = pd.DataFrame(val_output)
        val_pred_sbj.to_csv(os.path.join(log_dir, f'val_pred_sbj_{int(sbj)+1}.csv'), index=None)

        if all_eval_output is None:
            all_eval_output = val_output
        else:
            all_eval_output = np.concatenate((all_eval_output, val_output), axis=0)

        # all_mod_eval_output and all_eval_output are the same here
        if all_mod_eval_output is None:
            all_mod_eval_output = mod_val_output
        else:
            all_mod_eval_output = np.concatenate((all_mod_eval_output, mod_val_output), axis=0)


        # fill values for normal evaluation
        labels = list(range(0, args.nb_classes))
        cp_scores[0, :, int(sbj)] = jaccard_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)
        cp_scores[1, :, int(sbj)] = precision_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)
        cp_scores[2, :, int(sbj)] = recall_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)
        cp_scores[3, :, int(sbj)] = f1_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)

        # fill values for train val gap evaluation
        train_val_gap[0, int(sbj)] = jaccard_score(train_output[:, 1], train_output[:, 0], average='macro', labels=labels) - \
                                     jaccard_score(val_output[:, 1], val_output[:, 0], average='macro', labels=labels)
        train_val_gap[1, int(sbj)] = precision_score(train_output[:, 1], train_output[:, 0], average='macro', labels=labels) - \
                                     precision_score(val_output[:, 1], val_output[:, 0], average='macro', labels=labels)
        train_val_gap[2, int(sbj)] = recall_score(train_output[:, 1], train_output[:, 0], average='macro', labels=labels) - \
                                     recall_score(val_output[:, 1], val_output[:, 0], average='macro', labels=labels)
        train_val_gap[3, int(sbj)] = f1_score(train_output[:, 1], train_output[:, 0], average='macro', labels=labels) - \
                                     f1_score(val_output[:, 1], val_output[:, 0], average='macro', labels=labels)

        # fill values for normal evaluation
        labels = list(range(0, args.nb_classes))
        mod_cp_scores[0, :, int(sbj)] = jaccard_score(mod_val_output[:, 1], mod_val_output[:, 0], average=None, labels=labels)
        mod_cp_scores[1, :, int(sbj)] = precision_score(mod_val_output[:, 1], mod_val_output[:, 0], average=None, labels=labels)
        mod_cp_scores[2, :, int(sbj)] = recall_score(mod_val_output[:, 1], mod_val_output[:, 0], average=None, labels=labels)
        mod_cp_scores[3, :, int(sbj)] = f1_score(mod_val_output[:, 1], mod_val_output[:, 0], average=None, labels=labels)

        # fill values for train val gap evaluation
        mod_train_val_gap[0, int(sbj)] = jaccard_score(train_output[:, 1], train_output[:, 0], average='macro',
                                                   labels=labels) - \
                                     jaccard_score(mod_val_output[:, 1], mod_val_output[:, 0], average='macro', labels=labels)
        mod_train_val_gap[1, int(sbj)] = precision_score(train_output[:, 1], train_output[:, 0], average='macro', labels=labels) - \
                                         precision_score(mod_val_output[:, 1], mod_val_output[:, 0], average='macro', labels=labels)
        mod_train_val_gap[2, int(sbj)] = recall_score(train_output[:, 1], train_output[:, 0], average='macro', labels=labels) - \
                                     recall_score(mod_val_output[:, 1], mod_val_output[:, 0], average='macro', labels=labels)
        mod_train_val_gap[3, int(sbj)] = f1_score(train_output[:, 1], train_output[:, 0], average='macro', labels=labels) - \
                                     f1_score(mod_val_output[:, 1], mod_val_output[:, 0], average='macro', labels=labels)

        cp_savings[0, :, int(sbj)] = data_saved
        cp_savings[1, :, int(sbj)] = data_windows
        cp_savings[2, :, int(sbj)] = comp_saved
        cp_savings[3, :, int(sbj)] = comp_windows

        print("SUBJECT {0} VALIDATION RESULTS: ".format(int(sbj) + 1))
        print("Accuracy: {0}".format(jaccard_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)))
        print("Precision: {0}".format(precision_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)))
        print("Recall: {0}".format(recall_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)))
        print("F1: {0}".format(f1_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)))

        print("SUBJECT {0} MODIFIED VALIDATION RESULTS: ".format(int(sbj) + 1))
        print("Accuracy: {0}".format(jaccard_score(mod_val_output[:, 1], mod_val_output[:, 0], average=None, labels=labels)))
        print("Precision: {0}".format(precision_score(mod_val_output[:, 1], mod_val_output[:, 0], average=None, labels=labels)))
        print("Recall: {0}".format(recall_score(mod_val_output[:, 1], mod_val_output[:, 0], average=None, labels=labels)))
        print("F1: {0}".format(f1_score(mod_val_output[:, 1], mod_val_output[:, 0], average=None, labels=labels)))

        print("SUBJECT {} COMPUTATION SAVINGS: {:.2%}".format(int(sbj) + 1, sum(comp_saved) / sum(comp_windows)))
        print("SUBJECT {} DATA SAVINGS: {:.2%}".format(int(sbj) + 1, sum(data_saved) / sum(data_windows)))


    if args.save_analysis:
        mkdir_if_missing(log_dir)
        cp_score_acc = pd.DataFrame(cp_scores[0, :, :], index=None)
        cp_score_acc.index = args.class_names
        cp_score_prec = pd.DataFrame(cp_scores[1, :, :], index=None)
        cp_score_prec.index = args.class_names
        cp_score_rec = pd.DataFrame(cp_scores[2, :, :], index=None)
        cp_score_rec.index = args.class_names
        cp_score_f1 = pd.DataFrame(cp_scores[3, :, :], index=None)
        cp_score_f1.index = args.class_names
        tv_gap = pd.DataFrame(train_val_gap, index=None)
        tv_gap.index = ['accuracy', 'precision', 'recall', 'f1']
        if args.name:
            cp_score_acc.to_csv(os.path.join(log_dir, 'cp_scores_acc_{}.csv'.format(args.name)))
            cp_score_prec.to_csv(os.path.join(log_dir, 'cp_scores_prec_{}.csv').format(args.name))
            cp_score_rec.to_csv(os.path.join(log_dir, 'cp_scores_rec_{}.csv').format(args.name))
            cp_score_f1.to_csv(os.path.join(log_dir, 'cp_scores_f1_{}.csv').format(args.name))
            tv_gap.to_csv(os.path.join(log_dir, 'train_val_gap_{}.csv').format(args.name))
        else:
            cp_score_acc.to_csv(os.path.join(log_dir, 'cp_scores_acc.csv'))
            cp_score_prec.to_csv(os.path.join(log_dir, 'cp_scores_prec.csv'))
            cp_score_rec.to_csv(os.path.join(log_dir, 'cp_scores_rec.csv'))
            cp_score_f1.to_csv(os.path.join(log_dir, 'cp_scores_f1.csv'))
            tv_gap.to_csv(os.path.join(log_dir, 'train_val_gap.csv'))

        mod_cp_score_acc = pd.DataFrame(mod_cp_scores[0, :, :], index=None)
        mod_cp_score_acc.index = args.class_names
        mod_cp_score_prec = pd.DataFrame(mod_cp_scores[1, :, :], index=None)
        mod_cp_score_prec.index = args.class_names
        mod_cp_score_rec = pd.DataFrame(mod_cp_scores[2, :, :], index=None)
        mod_cp_score_rec.index = args.class_names
        mod_cp_score_f1 = pd.DataFrame(mod_cp_scores[3, :, :], index=None)
        mod_cp_score_f1.index = args.class_names
        mod_tv_gap = pd.DataFrame(mod_train_val_gap, index=None)
        mod_tv_gap.index = ['accuracy', 'precision', 'recall', 'f1']
        if args.name:
            mod_cp_score_acc.to_csv(os.path.join(log_dir, 'mod_cp_scores_acc_{}.csv'.format(args.name)))
            mod_cp_score_prec.to_csv(os.path.join(log_dir, 'mod_cp_scores_prec_{}.csv').format(args.name))
            mod_cp_score_rec.to_csv(os.path.join(log_dir, 'mod_cp_scores_rec_{}.csv').format(args.name))
            mod_cp_score_f1.to_csv(os.path.join(log_dir, 'mod_cp_scores_f1_{}.csv').format(args.name))
            mod_tv_gap.to_csv(os.path.join(log_dir, 'mod_train_val_gap_{}.csv').format(args.name))
        else:
            mod_cp_score_acc.to_csv(os.path.join(log_dir, 'mod_cp_scores_acc.csv'))
            mod_cp_score_prec.to_csv(os.path.join(log_dir, 'mod_cp_scores_prec.csv'))
            mod_cp_score_rec.to_csv(os.path.join(log_dir, 'mod_cp_scores_rec.csv'))
            mod_cp_score_f1.to_csv(os.path.join(log_dir, 'mod_cp_scores_f1.csv'))
            mod_tv_gap.to_csv(os.path.join(log_dir, 'mod_train_val_gap.csv'))

    evaluate_participant_scores(participant_scores=cp_scores,
                                gen_gap_scores=train_val_gap,
                                input_cm=all_eval_output,
                                class_names=args.class_names,
                                nb_subjects=int(np.max(data[:, 0]) + 1),
                                filepath=os.path.join('logs', log_date, log_timestamp),
                                filename='cross-participant',
                                args=args
                                )

    # evaluate_mod_participant_scores(savings_scores=cp_savings,
    #                                 participant_scores=mod_cp_scores,
    #                                 gen_gap_scores=mod_train_val_gap,
    #                                 input_cm=all_mod_eval_output,
    #                                 class_names=args.class_names,
    #                                 nb_subjects=int(np.max(data[:, 0]) + 1),
    #                                 filepath=os.path.join('logs', log_date, log_timestamp),
    #                                 filename='mod-cross-participant',
    #                                 args=args
    #                                 )
    

    # all_mod_eval_output and all_eval_output are the same
    return net, all_eval_output 
