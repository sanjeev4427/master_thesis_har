import numpy as np
import torch
from torch.utils.data import DataLoader
from misc.torchutils import seed_worker
from data_processing.sliding_window import apply_sliding_window
import pandas as pd

def ml_generate_train_data(data, args, sbj):

    # config dictionary containing setting parameters
    config= vars(args)   
    
    

    # all_val_gt = []
    # for i, sbj in enumerate(np.unique(data[:, 0])):
    #     # loading data
    #     train_data = data[data[:, 0] != sbj] # training data from all but one subject
    #     val_data = data[data[:, 0] == sbj]  # validaaton data from one subject

        ############ generating config['saving_array'] ##################################

        # # calculate concurrent windows
        # curr_label = None
        # curr_window = 0
        # windows = []
        # for sbj_id in np.unique(train_data[:, 0]):  # first column is subject id
        #     sbj_label = train_data[train_data[:, 0] == sbj_id][:, -1]  # label column for each subject in training data
        #     for label in sbj_label:
        #         if label != curr_label and curr_label is not None:
        #             windows.append(
        #                 [curr_label, curr_window / args.sampling_rate, curr_window]
        #             )  # store training duration in terms of number of windows, curr_window is actually data point here
        #                # 'curr_window / args.sampling_rate' calculates number of windows without overlapping
        #             curr_label = label                    
        #             curr_window = 1   # reset curr_window to 1
        #         elif label == curr_label:
        #             curr_window += 1 # curr_window is actually data point and not the window as apply_window function has not been used yet
        #         else:
        #             curr_label = label
        #             curr_window += 1
        # windows = np.array(
        #     windows
        # )  

        # calculate savings array, calculates activity duration 
        # (in terms of number of windows without overlap) for each class/activity
        # saving_array = np.zeros(args.nb_classes)
        # for label in range(args.nb_classes):
        #     label_windows = windows[
        #         windows[:, 0] == label #! this label should be subj id but in for loop label is actually classes
        #     ]  # accessing windows label wise
        #     if label_windows.size != 0:
        #         if args.saving_type == "mean":
        #             saving_array[int(label)] = np.mean(
        #                 label_windows[:, 1].astype(float)
        #             )  # mean of activity duration of each activity across all subjects
        #         elif args.saving_type == "median":
        #             saving_array[int(label)] = np.median(
        #                 label_windows[:, 1].astype(float)
        #             )  # median of activity duration of each activity across all subjects
        #         elif args.saving_type == "min":
        #             saving_array[int(label)] = np.min(label_windows[:, 1].astype(float))
        #         elif args.saving_type == "max":
        #             saving_array[int(label)] = np.max(label_windows[:, 1].astype(float))
        #         elif args.saving_type == "first_quartile":
        #             saving_array[int(label)] = np.percentile(
        #                 label_windows[:, 1].astype(float), 25
        #             )

        # args.saving_array = saving_array
        # print(f'{args.saving_type} activity duaration for all activities: \n',
        # saving_array)
        ################################################################################

        # # Sensor data is segmented using a sliding window mechanism
        # X_val, y_val = apply_sliding_window(val_data[:, :-1], val_data[:, -1],
        #                                     sliding_window_size=args.sw_length,
        #                                     unit=args.sw_unit,
        #                                     sampling_rate=args.sampling_rate,
        #                                     sliding_window_overlap=args.sw_overlap,
        #                                     )

        # X_val = X_val[:, :, 1:] # removing subj_id from X_val
        
        # val_features, val_labels = X_val, y_val
        
        # g = torch.Generator()
        # g.manual_seed(config['seed'])

        # dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_labels))
        # valloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
        #                 worker_init_fn=seed_worker, generator=g, pin_memory=True)

        # # helper objects
        # val_gt = []
        # with torch.no_grad():
        #     # iterate over validation dataset
        #     for i, (x, y) in enumerate(valloader):
        #         # send x and y to GPU
        #         inputs, targets = x.to(config['gpu']), y.to(config['gpu'])
        #         y_true = targets.cpu().numpy().flatten()
        #         val_gt = np.concatenate((np.array(val_gt, int), np.array(y_true, int)))
        # all_val_gt = np.concatenate((np.array(all_val_gt, int), np.copy(val_gt)))  
    # df = pd.read_csv(r'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\logs\222255\val_pred_all_sbj.csv')
    
    if args.dataset == 'wetlab':
        ml_train_data = np.loadtxt(rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\ml_training_data\wetlab\train_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')
    elif args.dataset == 'rwhar':
        ml_train_data = np.loadtxt(fr'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\ml_training_data\rwhar\train_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')

    train_pred = ml_train_data[:,0]
    train_gt = ml_train_data[:,1] 
    ml_train_gt_pred = np.vstack((train_pred, train_gt)).T
    return ml_train_gt_pred


def ml_generate_train_data_exp_gt(data, args, sbj):

    # config dictionary containing setting parameters
    config= vars(args)   
    
    if args.dataset == 'wetlab':
        ml_train_data = np.loadtxt(rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\ml_training_data\wetlab\train_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')
    elif args.dataset == 'rwhar':
        ml_train_data = np.loadtxt(fr'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\ml_training_data\rwhar\train_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')

    train_pred = ml_train_data[:,0]
    train_gt = ml_train_data[:,1] 
    ml_train_gt_gt = np.vstack((train_gt, train_gt)).T
    return ml_train_gt_gt


