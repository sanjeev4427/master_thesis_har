 # apply best settings on validation data
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from data_processing.data_analysis import plot_sensordata_and_labels
from misc.osutils import mkdir_if_missing
from model.evaluate import evaluate_mod_participant_scores
from skip_heuristics_scripts.data_skipping import data_skipping


def  apply_best_settings(sbj, args,*arg):  
    """
    Apply the best settings to the given subject's training and validation data.

    Args:
        sbj (str): The subject ID.
        args (argparse.Namespace): The command-line arguments passed to the script.
        *arg (str): Variable length argument list; the first argument should be the name of the file containing the best settings.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following six elements:
            mod_train_preds (np.ndarray): The modified predictions for the subject's training data after applying the best settings.
            mod_val_preds (np.ndarray): The modified predictions for the subject's validation data after applying the best settings.
            train_output (np.ndarray): The original predictions for the subject's training data.
            val_output (np.ndarray): The original predictions for the subject's validation data.
            mod_val_comp_saved (np.ndarray): The computation saved for each activity in the subject's validation data after applying the best settings.
            mod_val_data_saved (np.ndarray): The data saved for each activity in the subject's validation data after applying the best settings.
    """
        
    config=vars(args)   
    # comapare f1 after applying best setting
    if args.dataset == 'rwhar':
        train_output = np.loadtxt(rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\ml_training_data\rwhar\train_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')
        val_output = np.loadtxt(rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\ml_training_data\rwhar\val_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')
        activity_labels = range(8)
    elif args.dataset == 'wetlab':
        train_output = np.loadtxt(rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\ml_training_data\wetlab\train_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')
        val_output = np.loadtxt(rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\ml_training_data\wetlab\val_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')
        activity_labels = range(10)

    computations_saved = np.zeros(args.nb_classes)
    data_saved = np.zeros(args.nb_classes)

    # activate apply_best mode in dat_skipping
    best_filename = arg[0]
    apply_best = True

    # apply best on training data
    mod_train_preds,train_data_saved,train_comp_saved= data_skipping(-2, np.copy(train_output[:,0]), config, data_saved, computations_saved, apply_best, best_filename)

    # apply best on validation data
    mod_val_preds,mod_val_data_saved,mod_val_comp_saved= data_skipping(-2, np.copy(val_output[:,0]), config, data_saved, computations_saved, apply_best, best_filename)

    return mod_train_preds, mod_val_preds, train_output, val_output, mod_val_comp_saved, mod_val_data_saved


def ml_validation(args, data, algo_name, log_date, log_timestamp):
    """
    Calculate cross-participant scores using Leave-One-Subject-Out Cross Validation (LOSO CV).
    
    Args:
        args: A Namespace object containing the parsed command line arguments.
        data: A numpy array, where each row contains the label and features of a sample.
        algo_name: A string representing the name of the algorithm being used.
        log_date: A string representing the date of logging.
        log_timestamp: A string representing the timestamp of logging.
        
    Returns:
        None
    """
    print('\nCALCULATING CROSS-PARTICIPANT SCORES USING LOSO CV.\n')
    # mod_train_val_gap = np.zeros((4, int(np.max(data[:, 0]) + 1)))
    # all_mod_eval_output = None
    # mod_cp_scores = np.zeros((4, args.nb_classes, int(np.max(data[:, 0]) + 1)))
    # cp_scores = np.zeros((4, args.nb_classes, int(np.max(data[:, 0]) + 1)))
    # log_dir = os.path.join('logs', log_date, log_timestamp)
    # mod_cp_savings = np.zeros((1, args.nb_classes, int(np.max(data[:, 0]) + 1)))

    # for _,sbj in enumerate(np.unique(data[:, 0])):
    lst_sbj = np.unique(data[:, 0])
    
    # lst_sbj = range(1) # change number of subjects here
    mod_train_val_gap = np.zeros((4, len(lst_sbj)))
    all_mod_eval_output = None
    mod_cp_scores = np.zeros((4, args.nb_classes, len(lst_sbj)))
    cp_scores = np.zeros((4, args.nb_classes, len(lst_sbj)))
    log_dir = os.path.join('logs', log_date, log_timestamp)
    mod_cp_savings = np.zeros((2, args.nb_classes, len(lst_sbj)))

    for sbj in lst_sbj:
        exp1_1_0 = False
        exexp1_2_0 = False
        exp_less_sbj = False
        exp_data_included = True

        if exexp1_2_0 == True:
            filename_best_csv = fr'.\best_files_exp\{args.dataset}_exp_1_2_0.csv'

        elif exp_less_sbj == True:
            filename_best_csv = fr".\best_files_exp\best_less_sbjs\{algo_name}\{args.dataset}\best_results_for_{args.dataset}_{algo_name}_{int(sbj)+1}.csv"
        
        elif exp_data_included == True:
            filename_best_csv = fr"best_files_exp\best_files\{algo_name}\best_results_for_{args.dataset}_{algo_name}_{int(sbj)+1}.csv"

        else:
            filename_best_csv = fr'.\best_files_{algo_name}\{args.dataset}\best_results_for_{args.dataset}_{algo_name}_{int(sbj)+1}.csv'
        
        # for plotting sensor data.
        val_data = data[data[:, 0] == sbj]

        # modified training and validation predictions
        mod_train_preds, mod_val_preds, train_output, val_output, mod_val_comp_saved, mod_val_data_saved \
                                                 = apply_best_settings(sbj, args,filename_best_csv)

        val_gt = val_output[:,1]
        val_pred = np.copy(val_output[:,0])

        train_gt = train_output[:,1]
        train_pred = np.copy(train_output[:,0])

        # for activity in range(len(args.class_names)):

        #     f_one_gt_mod_val = f1_score(val_gt, mod_val_preds, labels = np.array([activity]), average= None) 
        #     f_one_gt_val = f1_score(val_gt, val_pred, labels = np.array([activity]), average= None) 
        #     f_one_gt_mod_train = f1_score(train_gt, mod_train_preds, labels = np.array([activity]), average= None) 
        #     f_one_gt_train = f1_score(train_gt, train_pred, labels = np.array([activity]), average= None) 

        #     print(f'Activity {args.class_names[activity]} f1: {f_one_gt_train}')
        #     print(f'f1 Modified: {f_one_gt_mod_train}', '\n')

        #     print(f'Activity {args.class_names[activity]} f1: {f_one_gt_val}')
        #     print(f'f1 Modified: {f_one_gt_mod_val}', '\n')


        if all_mod_eval_output is None:
            all_mod_eval_output = np.vstack((mod_val_preds, val_gt)).T
        else:
            all_mod_eval_output = np.concatenate((all_mod_eval_output, np.vstack((mod_val_preds, val_gt)).T), axis=0)

        # plot sensor data and subject wise activity and modified activity
        # to create activity plot wrt time from best settings 
        mkdir_if_missing(log_dir)
        if args.name:
            plot_sensordata_and_labels(val_data, sbj, val_gt, args.class_names, val_pred,
                                        mod_val_preds,
                                        figname=os.path.join(log_dir, 'sbj_' + str(int(sbj)) + '_' + args.name + '.png'))
        else:
            plot_sensordata_and_labels(val_data, sbj, val_gt, args.class_names, val_pred,
                                    mod_val_preds,
                                    figname=os.path.join(log_dir, 'sbj_' + str(int(sbj)) + '.png')) 

        # fill values for normal evaluation
        labels = list(range(0, args.nb_classes))
        cp_scores[0, :, int(sbj)] = jaccard_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)
        cp_scores[1, :, int(sbj)] = precision_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)
        cp_scores[2, :, int(sbj)] = recall_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)
        cp_scores[3, :, int(sbj)] = f1_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)

        mod_cp_scores[0, :, int(sbj)] = jaccard_score(val_gt, mod_val_preds, average=None, labels=labels)
        mod_cp_scores[1, :, int(sbj)] = precision_score(val_gt, mod_val_preds, average=None, labels=labels)
        mod_cp_scores[2, :, int(sbj)] = recall_score(val_gt, mod_val_preds, average=None, labels=labels)
        mod_cp_scores[3, :, int(sbj)] = f1_score(val_gt, mod_val_preds, average=None, labels=labels)

        # fill values for train val gap evaluation
        mod_train_val_gap[0, int(sbj)] = jaccard_score(train_gt, mod_train_preds, average='macro',
                                                   labels=labels) - \
                                     jaccard_score(val_gt, mod_val_preds, average='macro', labels=labels)
        mod_train_val_gap[1, int(sbj)] = precision_score(train_gt, mod_train_preds, average='macro', labels=labels) - \
                                         precision_score(val_gt, mod_val_preds, average='macro', labels=labels)
        mod_train_val_gap[2, int(sbj)] = recall_score(train_gt, mod_train_preds, average='macro', labels=labels) - \
                                     recall_score(val_gt, mod_val_preds, average='macro', labels=labels)
        mod_train_val_gap[3, int(sbj)] = f1_score(train_gt, mod_train_preds, average='macro', labels=labels) - \
                                     f1_score(val_gt, mod_val_preds, average='macro', labels=labels)
        
        mod_cp_savings[0, :, int(sbj)] = mod_val_comp_saved
        mod_cp_savings[1, :, int(sbj)] = mod_val_data_saved


        print("SUBJECT {0} VALIDATION RESULTS: ".format(int(sbj) + 1))
        # print("Accuracy: {0}".format(jaccard_score(val_gt, val_pred, average=None, labels=labels)))
        # print("Precision: {0}".format(precision_score(val_gt, val_pred, average=None, labels=labels)))
        # print("Recall: {0}".format(recall_score(val_gt, val_pred, average=None, labels=labels)))
        print("F1: {0}".format(f1_score(val_gt, val_pred, average=None, labels=labels)))
        print("Average F1: {0}".format(f1_score(val_gt, val_pred, average='macro')), '\n')


        print("SUBJECT {0} MODIFIED VALIDATION RESULTS: ".format(int(sbj) + 1))
        # print("Accuracy: {0}".format(jaccard_score(val_gt, mod_val_preds, average=None, labels=labels)))
        # print("Precision: {0}".format(precision_score(val_gt, mod_val_preds, average=None, labels=labels)))
        # print("Recall: {0}".format(recall_score(val_gt, mod_val_preds, average=None, labels=labels)))
        print("F1: {0}".format(f1_score(val_gt, mod_val_preds, average=None, labels=labels)))
        print("Average F1: {0}".format(f1_score(val_gt, mod_val_preds, average='macro')), '\n')

        print("SUBJECT {0} ML TRAINING RESULTS: ".format(int(sbj) + 1))
        print("F1: {0}".format(f1_score(train_gt, train_pred, average=None, labels=labels)))
        print("Average F1: {0}".format(f1_score(train_gt, train_pred, average='macro')), '\n')

        print("SUBJECT {0} ML MODIFIED TRAINING RESULTS: ".format(int(sbj) + 1))
        print("F1: {0}".format(f1_score(train_gt, mod_train_preds, average=None, labels=labels)))
        print("Average F1: {0}".format(f1_score(train_gt, mod_train_preds, average='macro')), '\n')

        print("SUBJECT {} COMPUTATION SAVINGS: {} %".format(int(sbj) + 1, mod_val_comp_saved))

    if args.save_analysis:
        mkdir_if_missing(log_dir)
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
        mod_cp_savings_comp = pd.DataFrame(mod_cp_savings[0, :, :], index=None)
        mod_cp_savings_comp.index = args.class_names
        mod_cp_savings_data = pd.DataFrame(mod_cp_savings[1, :, :], index=None)
        mod_cp_savings_data.index = args.class_names

        if args.name:
            mod_cp_score_acc.to_csv(os.path.join(log_dir, 'mod_cp_scores_acc_{}.csv'.format(args.name)))
            mod_cp_score_prec.to_csv(os.path.join(log_dir, 'mod_cp_scores_prec_{}.csv').format(args.name))
            mod_cp_score_rec.to_csv(os.path.join(log_dir, 'mod_cp_scores_rec_{}.csv').format(args.name))
            mod_cp_score_f1.to_csv(os.path.join(log_dir, 'mod_cp_scores_f1_{}.csv').format(args.name))
            mod_tv_gap.to_csv(os.path.join(log_dir, 'mod_train_val_gap_{}.csv').format(args.name))
            mod_cp_savings_comp.to_csv(os.path.join(log_dir, 'mod_cp_savings_comp{}.csv').format(args.name))
            mod_cp_savings_data.to_csv(os.path.join(log_dir, 'mod_cp_savings_data{}.csv').format(args.name))
        else:
            mod_cp_score_acc.to_csv(os.path.join(log_dir, 'mod_cp_scores_acc.csv'))
            mod_cp_score_prec.to_csv(os.path.join(log_dir, 'mod_cp_scores_prec.csv'))
            mod_cp_score_rec.to_csv(os.path.join(log_dir, 'mod_cp_scores_rec.csv'))
            mod_cp_score_f1.to_csv(os.path.join(log_dir, 'mod_cp_scores_f1.csv'))
            mod_tv_gap.to_csv(os.path.join(log_dir, 'mod_train_val_gap.csv'))
            mod_cp_savings_comp.to_csv(os.path.join(log_dir, 'mod_cp_savings_comp.csv'))
            mod_cp_savings_data.to_csv(os.path.join(log_dir, 'mod_cp_savings_data.csv'))
    evaluate_mod_participant_scores(algo_name, savings_scores=mod_cp_savings,
                            participant_scores=mod_cp_scores,
                            participant_scores_unmod=cp_scores,
                            gen_gap_scores=mod_train_val_gap,
                            input_cm=all_mod_eval_output,
                            class_names=args.class_names,
                            # nb_subjects=int(np.max(data[:, 0]) + 1),
                            nb_subjects=len(lst_sbj),
                            filepath=os.path.join('logs', log_date, log_timestamp),
                            filename='mod-cross-participant',
                            args=args
                            )
