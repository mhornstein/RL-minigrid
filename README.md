
# Minigrid Solver using Tabular and Deep Reinforcement Learning Methods

This project presents the use of Reinforcement Learning to solve the Minigrid game, a task-oriented 2D grid-based puzzle.

## Project Structure

The project is divided into three main parts:

### Experiments

In this section, one can explore how different hyperparameter values affect the performance of the models. The script for running experiments is `experimenter.py`. To execute the experiments, use the following command:

```bash
python experimenter.py
```

To customize the configuration for the experiments, adjust the settings in the `experiment_config.py` file. It is well-documented to assist in this task. Note that for experimenting with different hyperparameter values, simply update the `tested_parameters` dictionary as needed.

#### Experiment Results

During the experiments, you can expect the following:

* Progress updates will be printed to the console.
* For each hyperparameter, a dictionary of the form "result_<hyperparameter name>" will be generated. For example, for the learning rate, you will find the "results_learning_rate" dictionary.
* Inside each hyperparameter's dictionary, you will find the following:
  1. `results_plot.png`: An illustration comparing the results of different values of the hyperparameter.
  2. `test_result_<hyperparameter name>.csv` and `train_result_<hyperparameter name>.csv`: CSV files containing the raw data used to generate the `results_plot.png`.
  3. `train_log`: A directory containing training results for each hyperparameter. For each tested value of the hyperparameter, graphs showing loss, reward, and steps convergence are plotted.

You can access a subset of the results produced during the work on the project in the "experiments" directory committed to this repository.

### Notebooks

The code presented in the Python scripts was further adapted into well-structured and ordered Colab notebooks summarizing the findings, attached to this repository.

### Report

The report is attached as a PDF and further elaborates and summarizes the results.

## Tested Environment

The `experimenter.py` script have been tested in the following environment:

-   Operating System: Windows 10
-   Python Version: 3.10.8

The notebooks were tested in the Google Colab online environment.
