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

#########################
## full report generation
def create_header(subplot, header):
    subplot.set_title(header)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.spines.clear()

def create_table(subplot, header, df):
    df.columns = df.columns.str.replace('_', ' ')
    table_data = [df.columns] + df.values.tolist()

    table = subplot.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    subplot.set_title(header)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.spines.clear()

    for spine in subplot.spines.values(): # remove redundent spines
        spine.set_visible(False)

def plot_mean_steps(df, tested_parameter, ax):
    df = df[[tested_parameter, 'total_steps_avg']]
    sns.barplot(data=df, x=tested_parameter, y='total_steps_avg', palette=palette, ax=ax)
    ax.set(xlabel=ax.get_xlabel().replace('_', ' '))
    ax.set(ylabel='')

def plot_done_episodes_count(df, tested_parameter, ax):
    df = df[[tested_parameter, 'done_episodes_count']]
    sns.barplot(data=df, x=tested_parameter, y='done_episodes_count', palette=palette, ax=ax)
    ax.set(xlabel=ax.get_xlabel().replace('_', ' '))
    ax.set(ylabel='')

def create_tabular_method_report(plot_path, tested_parameter, train_result_file, test_result_file):
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(5, 6, height_ratios=[0.05,
                                       1,
                                       0.05, 1,
                                       0.1], hspace=0.4, wspace=0.4)

    # add headers
    parameter_header_subplot = fig.add_subplot(gs[0, :])
    create_header(parameter_header_subplot, tested_parameter.replace('_', ' '))

    train_mean_header_subplot = fig.add_subplot(gs[2, :])
    create_header(train_mean_header_subplot, 'Train metrics')

    # add evaluation results
    df = pd.read_csv(test_result_file)
    policy_evaluation_subplot = fig.add_subplot(gs[1, :])
    create_table(policy_evaluation_subplot, 'Policy evaluation results', df)

    # add train metrics
    df = pd.read_csv(train_result_file)
    total_episodes_count = df['total_episodes_count'].iloc[0]

    ax = fig.add_subplot(gs[3, 0:3])
    plot_done_episodes_count(df, tested_parameter, ax)
    ax.set_title(f'Total episodes done out of {total_episodes_count}')

    ax = fig.add_subplot(gs[3, 3:])
    plot_mean_steps(df, tested_parameter, ax)
    ax.set_title(f'Avg episode steps per {total_episodes_count} episodes')

    plt.savefig(f'{plot_path}/results_plot.png')
    plt.close()
    plt.clf()

if __name__ == '__main__':
    tested_param = 'ep_decay'

    result_path = f'results_{tested_param}'
    train_results_path = f'../part1/{result_path}/train_result_{tested_param}.csv'
    test_results_path = f'../part1/{result_path}/test_result_{tested_param}.csv'

    create_tabular_method_report(f'../part1/results_{tested_param}', tested_param, train_results_path, test_results_path)