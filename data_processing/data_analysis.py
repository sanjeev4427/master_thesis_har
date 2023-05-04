##################################################
# All functions related to analysing sensor data
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import StrMethodFormatter


class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=height + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def plot_sensordata_and_labels(sensordata, sbj, gt, class_names, predictions, mod_predictions, figname='test.png'):
    """
    Plots the sensor data and corresponding ground truth, predictions, and modified predictions for a given subject.

    Args:
    sensordata (pandas.DataFrame or numpy.ndarray): The sensor data to be plotted. If a pandas DataFrame, it should have columns "subject", "acc_x", "acc_y", "acc_z", and "label". If a numpy array, it should have shape (n_samples, 4) and columns [subject, acc_x, acc_y, acc_z].
    sbj (int): The subject number for the data being plotted.
    gt (numpy.ndarray): The ground truth labels for the sensor data.
    class_names (list): The list of class names for the activities being predicted.
    predictions (numpy.ndarray): The predicted labels for the sensor data.
    mod_predictions (numpy.ndarray): The modified predicted labels for the sensor data.
    figname (str): The filename to save the generated plot. Default is "test.png".

    Returns:
    - None
    """  
    figtitle=f'Activities (Subject: {int(sbj) + 1})'
    # acc_x = []
    # acc_y = []
    # acc_z = []
    # if sensordata is None:
    #     print("input data is empty, exiting the program.")
    #     exit(0)
    # if isinstance(sensordata, pd.DataFrame):
    #     sensordata.columns = ['subject', 'acc_x', 'acc_y', 'acc_z', 'label']
    #     acc_x = sensordata["acc_x"].to_numpy()
    #     acc_y = sensordata["acc_y"].to_numpy()
    #     acc_z = sensordata["acc_z"].to_numpy()
    # if isinstance(sensordata, np.ndarray):
    #     acc_x = sensordata[:, 1]
    #     acc_y = sensordata[:, 2]
    #     acc_z = sensordata[:, 3]

    n_classes = len(np.unique(gt))
    # # plot 1:
    fig, (ax2) = plt.subplots(1, 1, figsize=(18, 7))
    # ax1.plot(acc_x, color='blue')
    # ax1.plot(acc_y, color='green')
    # ax1.plot(acc_z, color='red')
    # ax1.set_ylabel("Acceleration (mg)")
    # ax1.legend(["acc x", "acc y", "acc z"])
    # ax1.set_xmargin(0)

    unordered_unique_labels, first_occurences_labels, labels_onehot = np.unique(gt, return_inverse=True,
                                                                         return_index=True)
    order = []

    ordered_unique_onehot_labels = first_occurences_labels.copy()
    ordered_unique_onehot_labels.sort()
    ordered_labels = []

    for index in ordered_unique_onehot_labels:
        ordered_labels.append(unordered_unique_labels[np.where(first_occurences_labels == index)[0][0]])
        order.append(np.where(first_occurences_labels == index)[0][0])

    cmap1 = plt.cm.get_cmap('Set1', n_classes).reversed()
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_ylabel("Mod. Predictions  vs.  Predictions  vs.  Ground Truth ", fontsize=12)
    # ax2.set_xlabel("time")
    ax2.pcolor([mod_predictions, predictions, gt], cmap=cmap1)

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.suptitle(figtitle, fontsize=16)

    c = [mpatches.Circle((0.5, 0.5), radius=0.25, color=cmap1(i), edgecolor="none") for i in
         range(n_classes)]

    plt.legend(c, class_names, bbox_to_anchor=(0.5, -0.15), loc='lower center', ncol=n_classes,
               fancybox=True, shadow=True,
               handler_map={mpatches.Circle: HandlerEllipse()}).get_frame()

    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # No decimal places
    plt.tight_layout()
    plt.savefig(figname)


def get_cmap(n, name='hsv'):
    import matplotlib.pyplot as plt

    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def max_min_values(data):
    """
    Function which prints the min and max value for each column of a dataset

    :param data: dataframe
        Data to be analysed
    """
    for i, column in enumerate(data.T):
        print('Column {}:'.format(i), 'min: {} max: {}'.format(min(column), max(column)))


def analyze_window_lengths(labels, subject_idx, sampling_rate):
    """
    Function which analyses and prints the avg. window lengths (of each label)

    :param labels: list
        List of label names
    :param subject_idx: list
        List of subject identifiers
    :param sampling_rate: int
         Sampling rate of dataset
    """
    curr_label = None
    curr_window = 0
    curr_subject = -1
    windows = []
    for i, (label, subject_id) in enumerate(zip(labels, subject_idx)):
        if label != curr_label and i > 0:
            windows.append([int(curr_subject), curr_label, curr_window / sampling_rate, curr_window])
            curr_label = label
            curr_subject = subject_id
            curr_window = 1
        elif label == curr_label:
            curr_window += 1
        else:
            curr_label = label
            curr_subject = subject_id
            curr_window += 1
    windows = np.array(windows)
    # per subject and label
    unique_subjects = np.unique(windows[:, 0])
    unique_labels = np.unique(windows[:, 1])
    print('\n#### ACTIVITY TIMES ####')
    for subject in unique_subjects:
        subject_windows = windows[windows[:, 0] == subject]
        print('\n#### SUBJECT {} #####'.format(int(subject)))
        curr_datapoint = 0
        for window in subject_windows:
            print('Label {}: '.format(window[1]),
                  'Duration: {}'.format(window[2]),
                  'Datapoints: {} - {}'.format(curr_datapoint, curr_datapoint + int(window[3]))
                  )
            curr_datapoint += int(window[3]) + 1
    print('\n#### PER SUBJECT-LABEL AVERAGES #####')
    for subject in unique_subjects:
        subject_windows = windows[windows[:, 0] == subject]
        print('\n#### SUBJECT {} #####'.format(int(subject)))
        for label in unique_labels:
            subject_label_windows = subject_windows[subject_windows[:, 1] == label]
            if subject_label_windows.size == 0:
                print('NO LABELS FOUND FOR: {}'.format(label))
            else:
                print('Label {}: '.format(label),
                      'avg. window length {:.1f} seconds, '.format(np.mean(subject_label_windows[:, 2].astype(float))),
                      'median window length {:.1f} seconds, '.format(np.median(subject_label_windows[:, 2].astype(float))),
                      'min. window length {:.1f} seconds, '.format(np.min(subject_label_windows[:, 2].astype(float))),
                      'max. window length {:.1f} seconds'.format(np.max(subject_label_windows[:, 2].astype(float)))
                      )
    print('\n#### PER LABEL AVERAGES #####')
    for label in unique_labels:
        label_windows = windows[windows[:, 1] == label]
        if label_windows.size == 0:
            print('LABEL NON-EXISTENT!')
        else:
            print('Label {}: '.format(label),
                  'avg. window length {:.1f} seconds, '.format(np.mean(label_windows[:, 2].astype(float))),
                  'median window length {:.1f} seconds, '.format(np.median(label_windows[:, 2].astype(float))),
                  'min. window length {:.1f} seconds, '.format(np.min(label_windows[:, 2].astype(float))),
                  'max. window length {:.1f} seconds'.format(np.max(label_windows[:, 2].astype(float)))
                  )


if __name__ == '__main__':
    print('\n WETLAB')
    wetlab_data = pd.read_csv('../data/wetlab_data.csv', header=None)
    analyze_window_lengths(wetlab_data.iloc[:, -2], wetlab_data.iloc[:, 0], 50)
    plot_sensordata_and_labels(wetlab_data, [], datasetname='WETLAB')