import matplotlib.pyplot as plt
import numpy as np

def activity_plot_loss_ga(best_loss_list, best_gen_list, config, activity_name):
    plt.figure()
    plt.plot(range(len(best_gen_list)), best_loss_list)
    plt.xlabel("Generations")
    plt.ylabel("Loss ")
    plt.title(f"Loss vs Generations \n activity: {activity_name}")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\figures\loss_iter_GA_{config["dataset"]}_{activity_name}.png', format="png", bbox_inches="tight"
    )
    return None


def activity_plot_f1_gen(best_f1_list, best_gen_list, config, activity_name):
    plt.figure()
    plt.plot(range(len(best_gen_list)), best_f1_list)
    plt.xlabel("Generations")
    plt.ylabel("F1 score")
    plt.title(f"F1 score vs Generations \n activity: {activity_name}")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\figures\f1score_iter_GA_{config["dataset"]}_{activity_name}.png', format="png", bbox_inches="tight"
    )
    #plt.show()
    return None

def activity_plot_comp_saved_gen(best_comp_saved_ratio_list, best_gen_list, config, activity_name):
    plt.figure()
    plt.plot(range(len(best_gen_list)), best_comp_saved_ratio_list)
    plt.xlabel("Generations")
    plt.ylabel("Average computation saved")
    plt.title(f"Average computation saved vs Generations \n activity: {activity_name}")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\figures\comp_saved_iter_GA_{config["dataset"]}_{activity_name}.png', format="png", bbox_inches="tight"
    )
    #plt.show()
    return None

def activity_plot_threshold_gen(win_thrs_list, best_gen_list, config, activity_name):
    plt.figure()
    plt.plot(range(len(best_gen_list)), win_thrs_list)
    plt.xlabel("Generations")
    plt.ylabel("Threshold value")
    plt.title(f"Threshold value vs Generations \n activity: {activity_name}")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\figures\threshold_iter_GA_{config["dataset"]}_{activity_name}.png', format="png", bbox_inches="tight"
    )
    #plt.show()
    return None

def activity_plot_skip_windows_gen(skip_win_list, best_gen_list, config, activity_name):
    plt.figure()
    plt.plot(range(len(best_gen_list)), skip_win_list)
    plt.xlabel("Generations")
    plt.ylabel("Skip windows value")
    plt.title(f"Skip_windows value vs Generations \n activity: {activity_name}")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\figures\skip_windows_iter_GA_{config["dataset"]}_{activity_name}.png', format="png", bbox_inches="tight"
    )
    #plt.show()
    return None