import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

palette = 'Set2'

###################
## train log report

def save_heatmap(data, path, fmt='.2g', annot_kws=None, title=None, cbar_kws=None):
    fig = sns.heatmap(data, annot=True, fmt=fmt, annot_kws=annot_kws, cbar_kws=cbar_kws)
    if title is not None:
        plt.title(title)
    plt.savefig(path)
    plt.clf()
    plt.close()

def save_lineplot(data, path, title, xlabel, ylabel):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.clf()
    plt.close()

def log_training_process(experiment_log_dir, states_visits_mean, episodes_steps, episodes_rewards):
    if not os.path.exists(experiment_log_dir):
        os.makedirs(experiment_log_dir)

    save_heatmap(data=states_visits_mean.T, path=f'{experiment_log_dir}/states_visits_mean.png', fmt='.1f', annot_kws={"size": 6})

    save_lineplot(data=episodes_steps, path=f'{experiment_log_dir}/Convergence_Graph__Episodes_steps.png',
                  title='Convergence Graph: Episodes steps', xlabel='Episode number', ylabel='Steps')
    save_lineplot(data=episodes_rewards, path=f'{experiment_log_dir}/Convergence_Graph__Episodes_reward.png',
                  title='Convergence Graph: Episodes reward', xlabel='Episode number', ylabel='Reward')