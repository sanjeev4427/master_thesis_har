##################################################
# All functions related to evaluating training and testing results
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

import matplotlib.pyplot as plt
import numpy as np
import itertools

import os
from sklearn.metrics import confusion_matrix

from misc.osutils import mkdir_if_missing
from ml_evaluate import mod_bar_plot_activity, mod_bar_plot_sbj


def plot_confusion_matrix(input, target_names, title='Confusion matrix', cmap=None, normalize=True, output_path=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    input:        confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    cm = confusion_matrix(input[:, 1], input[:, 0])
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if output_path is not None:
        plt.savefig(output_path)


def evaluate_participant_scores(participant_scores, gen_gap_scores, input_cm, class_names, nb_subjects, filepath, filename, args):
    """
    Function which prints evaluation metrics of each participant, overall average and saves confusion matrix

    :param participant_scores: numpy array
        Array containing all results
    :param gen_gap_scores:
        Array containing generalization gap results
    :param input_cm: confusion matrix
        Confusion matrix of overall results
    :param class_names: list of strings
        Class names
    :param nb_subjects: int
        Number of subjects in dataset
    :param filepath: str
        Directory where to save plots to
    :param filename: str
        Name of plot
    :param args: dict
        Overall settings dict
    """
    print('\nPREDICTION RESULTS')
    print('-------------------')
    print('Average results')
    avg_acc = np.mean(participant_scores[0, :, :])
    std_acc = np.std(participant_scores[0, :, :])
    avg_prc = np.mean(participant_scores[1, :, :])
    std_prc = np.std(participant_scores[1, :, :])
    avg_rcll = np.mean(participant_scores[2, :, :])
    std_rcll = np.std(participant_scores[2, :, :])
    avg_f1 = np.mean(participant_scores[3, :, :])
    std_f1 = np.std(participant_scores[3, :, :])
    print('Avg. Accuracy {:.4f} (±{:.4f}), '.format(avg_acc, std_acc),
          'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
          'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
          'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    if args.include_null:
        print('Average results (no null)')
        avg_acc = np.mean(participant_scores[0, 1:, :])
        std_acc = np.std(participant_scores[0, 1:, :])
        avg_prc = np.mean(participant_scores[1, 1:, :])
        std_prc = np.std(participant_scores[1, 1:, :])
        avg_rcll = np.mean(participant_scores[2, 1:, :])
        std_rcll = np.std(participant_scores[2, 1:, :])
        avg_f1 = np.mean(participant_scores[3, 1:, :])
        std_f1 = np.std(participant_scores[3, 1:, :])
        print('Avg. Accuracy {:.4f} (±{:.4f}), '.format(avg_acc, std_acc),
              'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
              'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
              'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    print('Average class results')
    for i, class_name in enumerate(class_names):
        avg_acc = np.mean(participant_scores[0, i, :])
        std_acc = np.std(participant_scores[0, i, :])
        avg_prc = np.mean(participant_scores[1, i, :])
        std_prc = np.std(participant_scores[1, i, :])
        avg_rcll = np.mean(participant_scores[2, i, :])
        std_rcll = np.std(participant_scores[2, i, :])
        avg_f1 = np.mean(participant_scores[3, i, :])
        std_f1 = np.std(participant_scores[3, i, :])
        print('Class {}: Avg. Accuracy {:.4f} (±{:.4f}), '.format(class_name, avg_acc, std_acc),
              'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
              'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
              'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    print('Subject-wise results')
    for subject in range(nb_subjects):
        print('Subject ', subject + 1, ' results: ')
        for i, class_name in enumerate(class_names):
            acc = participant_scores[0, i, subject]
            prc = participant_scores[1, i, subject]
            rcll = participant_scores[2, i, subject]
            f1 = participant_scores[3, i, subject]
            print('Class {}: Accuracy {:.4f}, '.format(class_name, acc),
                  'Precision {:.4f}, '.format(prc),
                  'Recall {:.4f}, '.format(rcll),
                  'F1-Score {:.4f}'.format(f1))

    print('\nGENERALIZATION GAP ANALYSIS')
    print('-------------------')
    print('Average results')
    avg_acc = np.mean(gen_gap_scores[0, :])
    std_acc = np.std(gen_gap_scores[0, :])
    avg_prc = np.mean(gen_gap_scores[1, :])
    std_prc = np.std(gen_gap_scores[1, :])
    avg_rcll = np.mean(gen_gap_scores[2, :])
    std_rcll = np.std(gen_gap_scores[2, :])
    avg_f1 = np.mean(gen_gap_scores[3, :])
    std_f1 = np.std(gen_gap_scores[3, :])
    print('Avg. Accuracy {:.4f} (±{:.4f}), '.format(avg_acc, std_acc),
          'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
          'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
          'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    print('Subject-wise results')
    for subject in range(nb_subjects):
        print('Subject ', subject + 1, ' results: ')
        acc = gen_gap_scores[0, subject]
        prc = gen_gap_scores[1, subject]
        rcll = gen_gap_scores[2, subject]
        f1 = gen_gap_scores[3, subject]
        print('Accuracy {:.4f}, '.format(acc),
              'Precision {:.4f}, '.format(prc),
              'Recall {:.4f}, '.format(rcll),
              'F1-Score {:.4f}'.format(f1))

    # create boxplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.suptitle('Average Participant Results', size=16)
    axs[0, 0].set_title('Accuracy')
    axs[0, 0].boxplot(participant_scores[0, :, :].T, labels=class_names, showmeans=True)
    axs[0, 1].set_title('Precision')
    axs[0, 1].boxplot(participant_scores[1, :, :].T, labels=class_names, showmeans=True)
    axs[1, 0].set_title('Recall')
    axs[1, 0].boxplot(participant_scores[2, :, :].T, labels=class_names, showmeans=True)
    axs[1, 1].set_title('F1-Score')
    axs[1, 1].boxplot(participant_scores[3, :, :].T, labels=class_names, showmeans=True)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)
    fig.subplots_adjust(hspace=0.5)
    mkdir_if_missing(filepath)
    if args.name:
        plt.savefig(os.path.join(filepath, filename + '_bx_{}.png'.format(args.name)))
        plot_confusion_matrix(input_cm, class_names, normalize=False,
                              output_path=os.path.join(filepath, filename + '_cm_{}.png'.format(args.name)))
    else:
        plt.savefig(os.path.join(filepath, filename + '_bx.png'))
        plot_confusion_matrix(input_cm, class_names, normalize=False,
                              output_path=os.path.join(filepath, filename + '_cm.png'))


def evaluate_mod_participant_scores(algo_name, savings_scores, participant_scores, participant_scores_unmod, gen_gap_scores, input_cm, class_names, nb_subjects, filepath, filename, args):
    """
    Function which prints evaluation metrics of each participant, overall average and saves confusion matrix

    :param participant_scores: numpy array
        Array containing all results
    :param gen_gap_scores:
        Array containing generalization gap results
    :param input_cm: confusion matrix
        Confusion matrix of overall results
    :param class_names: list of strings
        Class names
    :param nb_subjects: int
        Number of subjects in dataset
    :param filepath: str
        Directory where to save plots to
    :param filename: str
        Name of plot
    :param args: dict
        Overall settings dict
    """
    print('\nSAVINGS RESULTS')
    print('-------------------')
    print('Average computations savings')
    # avg_sbj_data_savings = np.mean(savings_scores, 1)[0, :] 
    avg_sbj_comp_savings = np.mean(savings_scores, 1)[0, :] 
    avg_sbj_data_savings = np.mean(savings_scores, 1)[1, :] 
    # print('Avg. Data Savings {:.4f} (±{:.4f}), '.format(np.mean(avg_sbj_data_savings), np.std(avg_sbj_data_savings)))
    print('Avg. Computation Savings {:.4f} (±{:.4f}), '.format(np.mean(avg_sbj_comp_savings), np.std(avg_sbj_comp_savings)))
    print('Avg. Data Savings {:.4f} (±{:.4f}), '.format(np.mean(avg_sbj_data_savings), np.std(avg_sbj_data_savings)))
    # if args.include_null:
    #     print('Average results (no null)')
    #     # print('Avg. Data Savings {:.4f} (±{:.4f})'.format(np.mean(avg_sbj_data_savings[1:]), np.std(avg_sbj_data_savings[1:])))
    #     print('Avg. Computation Savings {:.4f} (±{:.4f})'.format(np.mean(avg_sbj_comp_savings[1:]), np.std(avg_sbj_comp_savings[1:])))

    print('Subject-wise results')
    avg_comp_saved_subj = []
    for i, (sbj_comp_sav) in enumerate((avg_sbj_comp_savings)):
        print('Subject ', i + 1, ' results: ')
        print('Avg. Computation Savings {:.4f}'.format(sbj_comp_sav))
        avg_comp_saved_subj.append(sbj_comp_sav)

    avg_data_saved_subj = []
    for i, (sbj_data_sav) in enumerate((avg_sbj_data_savings)):
        print('Subject ', i + 1, ' results: ')
        print('Avg. Computation Savings {:.4f}'.format(sbj_data_sav))
        avg_data_saved_subj.append(sbj_data_sav)


    print('\nMODIFIED PREDICTION RESULTS')
    print('-------------------')
    print('Average results...')
    avg_acc = np.mean(participant_scores[0, :, :])
    std_acc = np.std(participant_scores[0, :, :])
    avg_prc = np.mean(participant_scores[1, :, :])
    std_prc = np.std(participant_scores[1, :, :])
    avg_rcll = np.mean(participant_scores[2, :, :])
    std_rcll = np.std(participant_scores[2, :, :])
    avg_f1_all = np.mean(participant_scores[3, :, :])
    std_f1 = np.std(participant_scores[3, :, :])
    avg_f1_unmod_all = np.mean(participant_scores_unmod[3, :, :])
    std_f1_unmod = np.std(participant_scores_unmod[3, :, :])
    avg_comp_saved_all = np.mean(savings_scores[0,:,:])
    avg_data_saved_all = np.mean(savings_scores[1,:,:])

    print('Avg. Accuracy {:.4f} (±{:.4f}), '.format(avg_acc, std_acc),
          'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
          'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll), '\n',
          'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1_all, std_f1),'\n',
          'Avg. F1-Score unmodified {:.4f} (±{:.4f})'.format(avg_f1_unmod_all, std_f1_unmod),'\n',
          f'Avg. comp saved: {avg_comp_saved_all}','\n',
          f'Avg. data saved: {avg_data_saved_all}' )
    if args.include_null:
        print('Average results (no null)')
        avg_acc = np.mean(participant_scores[0, 1:, :])
        std_acc = np.std(participant_scores[0, 1:, :])
        avg_prc = np.mean(participant_scores[1, 1:, :])
        std_prc = np.std(participant_scores[1, 1:, :])
        avg_rcll = np.mean(participant_scores[2, 1:, :])
        std_rcll = np.std(participant_scores[2, 1:, :])
        avg_f1 = np.mean(participant_scores[3, 1:, :])
        std_f1 = np.std(participant_scores[3, 1:, :])
        print('Avg. Accuracy {:.4f} (±{:.4f}), '.format(avg_acc, std_acc),
              'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
              'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
              'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1),'\n')
        
    print('Average class results...')
    avg_f1_activity = []
    avg_f1_unmod_activity = []
    avg_comp_saved_activity = []
    avg_data_saved_activity = []
    for i, class_name in enumerate(class_names):
        avg_acc = np.mean(participant_scores[0, i, :])
        std_acc = np.std(participant_scores[0, i, :])
        avg_prc = np.mean(participant_scores[1, i, :])
        std_prc = np.std(participant_scores[1, i, :])
        avg_rcll = np.mean(participant_scores[2, i, :])
        std_rcll = np.std(participant_scores[2, i, :])
        avg_f1 = np.mean(participant_scores[3, i, :])
        std_f1 = np.std(participant_scores[3, i, :])
        avg_f1_unmod = np.mean(participant_scores_unmod[3, i, :])
        std_f1_unmod = np.std(participant_scores_unmod[3, i, :])
        avg_comp_saved = np.mean(savings_scores[0, i, :])
        std_comp_saved = np.std(savings_scores[0, i, :])
        avg_data_saved = np.mean(savings_scores[1, i, :])
        std_data_saved = np.std(savings_scores[1, i, :])
        print('Class {}: Avg. Accuracy {:.4f} (±{:.4f}), '.format(class_name, avg_acc, std_acc),
              'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
              'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),'\n',
              'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1), '\n',
              'Avg. F1-Score unmodified {:.4f} (±{:.4f})'.format(avg_f1_unmod, std_f1_unmod),'\n',
              'Avg. comp saved {:.4f} (±{:.4f})'.format(avg_comp_saved, std_comp_saved),'\n',
              'Avg. data saved {:.4f} (±{:.4f})'.format(avg_data_saved, std_data_saved))
        # append avg. f1 score activity wise
        avg_f1_activity.append(avg_f1)
        avg_f1_unmod_activity.append(avg_f1_unmod)
        avg_comp_saved_activity.append(avg_comp_saved)
        avg_data_saved_activity.append(avg_data_saved)
    # appending overall avg. scores
    avg_f1_activity.append(avg_f1_all)
    avg_f1_unmod_activity.append(avg_f1_unmod_all)
    avg_comp_saved_activity.append(avg_comp_saved_all)
    avg_data_saved_activity.append(avg_data_saved_all)

    # plot bar-graph for activity wise results
    mod_bar_plot_activity(algo_name, avg_f1_activity, avg_f1_unmod_activity, avg_comp_saved_activity, avg_data_saved_activity, filepath, args)


    print('Subject-wise results')
    for subject in range(nb_subjects):
        print('Subject ', subject + 1, ' results: ')
        for i, class_name in enumerate(class_names):
            acc = participant_scores[0, i, subject]
            prc = participant_scores[1, i, subject]
            rcll = participant_scores[2, i, subject]
            f1 = participant_scores[3, i, subject]
            print('Class {}: Accuracy {:.4f}, '.format(class_name, acc),
                  'Precision {:.4f}, '.format(prc),
                  'Recall {:.4f}, '.format(rcll),
                  'F1-Score {:.4f}'.format(f1))
    # saving subj wise average results for bar graph
    avg_f1_subj = []
    avg_f1_unmod_subj = []
    for subject in range(nb_subjects):
        avg_f1_sub = np.mean(participant_scores[3, :, subject])
        avg_f1_unmod_sub = np.mean(participant_scores_unmod[3, :, subject])
        avg_f1_subj.append(avg_f1_sub)
        avg_f1_unmod_subj.append(avg_f1_unmod_sub)

    #plotting bar graph sub wise
    mod_bar_plot_sbj(algo_name, avg_f1_subj, avg_f1_unmod_subj, avg_comp_saved_subj, avg_data_saved_subj, filepath, args)
        

    print('\nGENERALIZATION GAP ANALYSIS')
    print('-------------------')
    print('Average results')
    avg_acc = np.mean(gen_gap_scores[0, :])
    std_acc = np.std(gen_gap_scores[0, :])
    avg_prc = np.mean(gen_gap_scores[1, :])
    std_prc = np.std(gen_gap_scores[1, :])
    avg_rcll = np.mean(gen_gap_scores[2, :])
    std_rcll = np.std(gen_gap_scores[2, :])
    avg_f1 = np.mean(gen_gap_scores[3, :])
    std_f1 = np.std(gen_gap_scores[3, :])
    print('Avg. Accuracy {:.4f} (±{:.4f}), '.format(avg_acc, std_acc),
          'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
          'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
          'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    print('Subject-wise results')
    for subject in range(nb_subjects):
        print('Subject ', subject + 1, ' results: ')
        acc = gen_gap_scores[0, subject]
        prc = gen_gap_scores[1, subject]
        rcll = gen_gap_scores[2, subject]
        f1 = gen_gap_scores[3, subject]
        print('Accuracy {:.4f}, '.format(acc),
              'Precision {:.4f}, '.format(prc),
              'Recall {:.4f}, '.format(rcll),
              'F1-Score {:.4f}'.format(f1))

    # create boxplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.suptitle('Modified Average Participant Results', size=16)
    activity = args.class_names
    capital_activity = [word.replace('_', ' ').title() for word in activity]
    axs[0, 0].set_title('Accuracy')
    axs[0, 0].boxplot(participant_scores[0, :, :].T, labels=capital_activity, showmeans=True)
    axs[0, 1].set_title('Precision')
    axs[0, 1].boxplot(participant_scores[1, :, :].T, labels=capital_activity, showmeans=True)
    axs[1, 0].set_title('Recall')
    axs[1, 0].boxplot(participant_scores[2, :, :].T, labels=capital_activity, showmeans=True)
    axs[1, 1].set_title('F1-Score')
    axs[1, 1].boxplot(participant_scores[3, :, :].T, labels=capital_activity, showmeans=True)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)
    fig.subplots_adjust(hspace=0.5)
    mkdir_if_missing(filepath)
    if args.name:
        plt.savefig(os.path.join(filepath, filename + '_bx_mod_{}.png'.format(args.name)))
        plot_confusion_matrix(input_cm, capital_activity, normalize=True, title='Mod. Confusion Matrix',
                              output_path=os.path.join(filepath, filename + '_cm_mod_{}.png'.format(args.name)))
    else:
        plt.savefig(os.path.join(filepath, filename + '_bx_mod.png'))
        plot_confusion_matrix(input_cm, capital_activity, title='Mod. Confusion Matrix', normalize=True,
                              output_path=os.path.join(filepath, filename + '_cm_mod.png'))