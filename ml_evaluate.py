
import os
from matplotlib import pyplot as plt
import numpy as np
from Other_helpful_scripts.bar_plot_act_f1_comp import bar_plot_act_f1_comp
from Other_helpful_scripts.graph_activities_gt_val_mod_val import graph_gt_val_mod_val
import csv

def mod_bar_plot_activity(algo_name, f1_score, f1_score_unmod, comp_saved, data_saved, filepath, args):

    # if args.dataset == 'rwhar':
    #     activity = ['Climbing down', 'Climbing_up', 'Jumping', 'lying',\
    #                    'running', 'sitting', 'standing', 'walking']
        
    # if args.dataset == 'wetlab':
    #     activity = ['null_class', 'cutting', 'inverting', 'peeling', 'pestling',\
    #                    'pipetting', 'pouring', 'pour catalysator', 'stirring', 'transfer']

    activity = args.class_names
    capital_activity = [word.replace('_', ' ').title() for word in activity]    
    activities = capital_activity + ['Average']
    # activities = ['null_class','cutting','inverting','peeling','pestling','pipetting','pouring','pour catalysator','stirring','transfer']

    f1_score = f1_score
    f1_gt_val = f1_score_unmod
    comp_saved = np.array(comp_saved)/100
    data_saved = np.array(data_saved)/100

    # create a list of floats from list of arrays
    # f1_score = [float(x) for sublist in f1_score for x in sublist]
    
    # Calculate percent difference
    percent_diff = [(f1 - f1_gt) / f1_gt * 100 for f1, f1_gt in zip(f1_score,f1_gt_val)]
    # percent_diff = [float(x) for sublist in percent_diff for x in sublist]

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the title and axis labels
    ax.set_title('Activities vs. F1 Score (mod. predictions), F1 Score (predictions), Computation saved, Data saved') #val_pred w.r.t GT
    ax.set_xlabel('Activities')
    ax.set_ylabel('F1 Score / Computaion saved/ Data saved')

    # Set the x-axis tick labels
    ax.set_xticklabels(activities, rotation=45, ha='right',fontsize = 'x-small')

    # Set bar positions and width
    x_pos = np.arange(len(activities))
    width = 0.20
  
    # Plot bars for f1 score and computation saved
    bar1 = ax.bar(x_pos, f1_score, width, label='F1 Score (mod. predictions)')
    bar2 = ax.bar(x_pos + width, f1_gt_val, width, label='F1 Score (predictions)')
    bar3 = ax.bar(x_pos + 2*width, comp_saved, width, label='Computation saved')
    bar4 = ax.bar(x_pos + 3*width, data_saved, width, label='Data saved')

    # Add text on bars
    for i,bar in enumerate([bar1, bar2]):
        hloc = []
        xloc = []
        for rect in bar:
            width, height = rect.get_width(), rect.get_height()
            x, y = rect.get_xy()
            ax.text(x + width / 2, y + height/2, f"{height * 100:.2f}%", ha='center', va='center', rotation=90,fontsize = 'small')
            xloc.append(x) 
            hloc.append(height)
        if i == 0:
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            for act,rect in enumerate(bar):
                width, height = rect.get_width(), rect.get_height()
                x, y = rect.get_xy()
                ax.text(x + width / 2, y + height + 0.08, f"{percent_diff[act]:.2f}%", ha='center', va='center', rotation=90, bbox=props, color='red')

        # ax.text(np.mean(xloc), max(hloc), f"{percent_diff[act]:.2f}%", ha='center', va='center', rotation=0)

    # Add text on bar comp saved and data saved
    for bar in [bar3]:
        for rect in bar:
            width, height = rect.get_width(), rect.get_height()
            x, y = rect.get_xy()
            ax.text(x + width / 2, y + height/2, f"{height*100:.2f}%", ha='center', va='center', rotation=90, fontsize = 'small')

    # Add text on bar data saved
    for bar in [bar4]:
        for rect in bar:
            width, height = rect.get_width(), rect.get_height()
            x, y = rect.get_xy()
            ax.text(x + width / 2, y + height/2, f"{height*100:.2f}%", ha='center', va='center', rotation=90, fontsize = 'small')
    # add horizontal line 
    plt.axhline(y=1,linewidth=0.25, color='black')

    # Set x-axis tick labels to activities
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(activities, rotation=45)
    ax.yaxis.grid(color='gray', linestyle='--', linewidth=0.5)

    # Show legend
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize='small')

    if args.name:
        filename=os.path.join(filepath, f'bar_plot_activity_{args.dataset}_{algo_name}' + args.name + '.png')
    else: 
        filename=os.path.join(filepath, f'bar_plot_activity_{args.dataset}_{algo_name}' + '.png')
    
    plt.savefig(
        filename, format="png", bbox_inches="tight"
    )



def mod_bar_plot_sbj(algo_name, f1_score, f1_score_unmod, comp_saved, data_saved, filepath, args):

    
    # subjects.append('Average')
    # activities = ['null_class','cutting','inverting','peeling','pestling','pipetting','pouring','pour catalysator','stirring','transfer']

    f1_score = f1_score
    f1_gt_val = f1_score_unmod
    comp_saved = np.array(comp_saved)/100
    data_saved = np.array(data_saved)/100
    subjects = [f'Subject {x+1}' for x in range(len(f1_score))]

    # create a list of floats from list of arrays
    # f1_score = [float(x) for sublist in f1_score for x in sublist]
    
    # Calculate percent difference
    percent_diff = [(f1 - f1_gt) / f1_gt * 100 for f1, f1_gt in zip(f1_score,f1_gt_val)]
    # percent_diff = [float(x) for sublist in percent_diff for x in sublist]

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the title and axis labels
    ax.set_title('Subjects vs. vs. F1 Score (mod. predictions) and F1 Score (predictions) and Data saved') #val_pred w.r.t GT
    ax.set_xlabel('Subjects')
    ax.set_ylabel('F1 Score / comp saved / Data saved')

    # Set the x-axis tick labels
    ax.set_xticklabels(subjects, rotation=45, ha='right',fontsize = 'x-small')

    # Set bar positions and width
    x_pos = np.arange(len(subjects))
    width = 0.15
  
    # Plot bars for f1 score and computation saved
    bar1 = ax.bar(x_pos, f1_score, width, label='F1 Score (mod. predictions)')
    bar2 = ax.bar(x_pos + width, f1_gt_val, width, label='F1 Score (predictions)')
    bar3 = ax.bar(x_pos + 2*width, comp_saved, width, label='Comp saved')
    bar4 = ax.bar(x_pos + 3*width, data_saved, width, label='Data saved')
    # Add text on bars
    for i,bar in enumerate([bar1, bar2]):
        hloc = []
        xloc = []
        for rect in bar:
            width, height = rect.get_width(), rect.get_height()
            x, y = rect.get_xy()
            # ax.text(x + width / 2, y + height/2, f"{height * 100:.2f}%", ha='center', va='center', rotation=90,fontsize = 'small')
            xloc.append(x) 
            hloc.append(height)
        if i == 0:
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            for act,rect in enumerate(bar):
                width, height = rect.get_width(), rect.get_height()
                x, y = rect.get_xy()
                # ax.text(x + width / 2, y + height + 0.08, f"{percent_diff[act]:.2f}%", ha='center', va='center', rotation=90, bbox=props, color='red')

        # ax.text(np.mean(xloc), max(hloc), f"{percent_diff[act]:.2f}%", ha='center', va='center', rotation=0)

    # Add text on bar comp saved
    for bar in [bar3]:
        for rect in bar:
            width, height = rect.get_width(), rect.get_height()
            x, y = rect.get_xy()
            # ax.text(x + width / 2, y + height/2, f"{height*100:.2f}%", ha='center', va='center', rotation=90, fontsize = 'small')
    # Add text on bar data saved
    for bar in [bar4]:
        for rect in bar:
            width, height = rect.get_width(), rect.get_height()
            x, y = rect.get_xy()
            # ax.text(x + width / 2, y + height/2, f"{height*100:.2f}%", ha='center', va='center', rotation=90, fontsize = 'small')

    # add horizontal line 
    plt.axhline(y=1,linewidth=0.25, color='black')

    # Set x-axis tick labels to activities
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(subjects, rotation=45)
    ax.yaxis.grid(color='gray', linestyle='--', linewidth=0.5)

    # Show legend
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize='small')

    if args.name:
        filename=os.path.join(filepath, f'bar_plot_subj_{args.dataset}_{algo_name}' + args.name + '.png')
    else: 
        filename=os.path.join(filepath, f'bar_plot_subj_{args.dataset}_{algo_name}' + '.png')
    
    plt.savefig(
        filename, format="png", bbox_inches="tight"
    )



def mod_bar_graph(sbj, args, log_dir, algo_name, data, f_one_gt_mod_val_act_wise_lst, filename_best_csv):

    config = vars(args)
    
    activity = []
    threshold = []
    skip_windows = []
    tolerance_value = []
    f1_score = []
    target_f1 = []
    computation_saved = []
    lowest_loss = []
    elapsed_time = []
    f1_gt_val = []
    f1_gt_mod_val = []
    f1_val_mod_val = []
    f_one_gt_mod_val_avg = []
    f_one_gt_val_avg = []

    with open(fr'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\logs\20230417\234814\best_results_for_rwhar_SA_rwhar_sbj_1_training.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # skip header row
        for row in reader:
            activity.append(row[0])
            threshold.append(float(row[1]))
            skip_windows.append(float(row[2]))
            tolerance_value.append(float(row[3]))
            f1_score.append(float(row[4].strip('[]')))
            target_f1.append(float(row[5]))
            computation_saved.append(float(row[6]))
            lowest_loss.append(float(row[7]))
            elapsed_time.append(row[8])
            f1_gt_val.append(float(row[9]))
            f1_gt_mod_val.append(float(row[10]))
            f1_val_mod_val.append(float(row[11]))
            f_one_gt_mod_val_avg.append(float(row[12]))
            f_one_gt_val_avg.append(float(row[13]))

        acitivity_name_lst = activity
        f_one_gt_val_lst = f1_gt_val
        best_comp_saved_for_activity_lst = computation_saved
        f_one_gt_val_avg = f_one_gt_val_avg
        f_one_gt_mod_val_avg = f_one_gt_mod_val_avg
        best_thrs_for_activity_lst = threshold
        best_skip_win_for_activity_lst = skip_windows

    # to create bar plot of of f1 score from best settings (applied to whole data set)
    bar_plot_act_f1_comp(sbj, acitivity_name_lst, f_one_gt_mod_val_act_wise_lst, f_one_gt_val_lst, best_comp_saved_for_activity_lst, log_dir, config, algo_name, True, f_one_gt_mod_val_avg, f_one_gt_val_avg,
                        best_thrs_for_activity_lst, best_skip_win_for_activity_lst)
    
    

    return None