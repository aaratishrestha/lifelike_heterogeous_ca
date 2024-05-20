import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt
import pandas as pd

def cell_states_count(cell_states_list: list):
    alive_count_list = []
    decay_count_list = []
    quiescent_count_list = []
    for i in range(len(cell_states_list)):
        cell_states_arr = cell_states_list[i]
        a_count = np.count_nonzero(cell_states_arr == 'a')
        d_count = np.count_nonzero(cell_states_arr == 'd')
        q_count = np.count_nonzero(cell_states_arr == 'q')

        alive_count_list.append(a_count)
        decay_count_list.append(d_count)
        quiescent_count_list.append(q_count)
    return {'alive_cell_count': alive_count_list,
            'decay_cell_count': decay_count_list, 'dead_cell_count': quiescent_count_list}


def plot_cell_states_count(alive_count_list: list, decay_count_list: list, quiescent_count_list: list, y_limit: int, plot_dir: str):
    split_plot_dir = plot_dir.split('/')
    exp_name = split_plot_dir[-4]
    plt.figure()
    plt.plot(alive_count_list, label='Alive Cells', color='green')
    plt.plot(decay_count_list, label='Decay Cells', color='orange')
    plt.plot(quiescent_count_list, label='Quiescent Cells', color='black')
    plt.ylim(0, y_limit)
    plt.title(f'Cell States count \n {exp_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Count')
    plt.legend()
    plotname = f'{plot_dir}/cell_states_count_plot.pdf'
    print('Plot fig here: ', plotname)
    plt.savefig(plotname)
    plt.close()


def unique_cumulative_rules(cell_rules_list: list):
    unique_rules_in_each_iteration = []
    unique_rules_cumm_count = []

    for rules in cell_rules_list:
        cell_rules = []
        for r in rules.tolist():
            cell_rules.extend(r)
        cell_rules = list(set(cell_rules))
        if 'nan' in cell_rules:
            cell_rules.remove('nan')
        unique_rules_in_each_iteration.append(cell_rules)

    for i in range(len(unique_rules_in_each_iteration)):
        if i > 0:
            unique_rules_in_each_iteration[i] = list(
                set(unique_rules_in_each_iteration[i - 1] + unique_rules_in_each_iteration[i]))
        unique_rules_cumm_count.append(len(unique_rules_in_each_iteration[i]))
    return {'cumulative_rule_count': unique_rules_cumm_count}


def plot_cumm_unique_rules_count(unique_rules_cumm_count: list, y_limit: int, plot_dir: str):
    split_plot_dir = plot_dir.split('/')
    exp_name = split_plot_dir[-4]
    plt.figure()
    plt.plot(unique_rules_cumm_count, label='Cumulative Unique Rules')
    plt.ylim(0, y_limit)
    plt.title(f'Cummulative Unique Rules \n {exp_name}')
    plt.xlabel('Iterations')
    plt.ylabel('Count')
    plt.legend()
    plotname = f'{plot_dir}/unique_rules_count_plot.pdf'
    plt.savefig(plotname)
    plt.close()


def count_1s_0s_dead_in_grid_state(grid_states_list: list, cell_states_list: list):
    count_0 = []
    count_1 = []
    count_dead = []

    for i in range(len(grid_states_list)):
        cell_state = cell_states_list[i]
        grid_state = grid_states_list[i].copy()
        mask = (cell_state == 'q')
        grid_state[mask] = -1
        count_0.append(np.count_nonzero(grid_state == 0))
        count_1.append(np.count_nonzero(grid_state == 1))
        count_dead.append(np.count_nonzero(grid_state == -1))
    return {'count_alive_0': count_0, 'count_alive_1': count_1}

# phenotype analysis :Count the numbers of 1s, 0s, and dead cells through the iterations and create a plot.


def plot_count_1s_0s_dead_in_grid_state(alive_count_0: list, alive_count_1: list, count_dead: list, y_limit: int, plot_dir: str):
    split_plot_dir = plot_dir.split('/')
    exp_name = split_plot_dir[-4]
    plt.figure()
    plt.plot(alive_count_0, label='alive 0', color='red')
    plt.plot(alive_count_1, label='alive 1', color='green')
    plt.plot(count_dead, label='quiescent', color='black')
    plt.ylim(0, y_limit)
    plt.title(
        f'Count Alive(0 and 1) and dead cell of Grid state \n {exp_name} ')
    plt.xlabel('iterations')
    plt.ylabel('Count')
    plt.legend()
    plotname = f'{plot_dir}/grid_state_count_alive_1_0_dead_count.pdf'
    plt.savefig(plotname)
    plt.close()


# Phenoptype Analaysis: Count the differences between iterations and generate a plot.
# Show the movement of grid states

def difference_betn_two_grid_states(grid_states_list: list):
    difference = []
    for i in range(len(grid_states_list) - 1):
        grid_state1 = grid_states_list[i].copy()
        grid_state2 = grid_states_list[i+1].copy()
        different_elements_indices = (grid_state1 != grid_state2)
        difference.append(np.count_nonzero(different_elements_indices == True))
    return {'grid_difference': difference}


def plot_difference_betn_two_grid_states(grid_difference: list, y_limit, plot_dir: str):
    split_plot_dir = plot_dir.split('/')
    exp_name = split_plot_dir[-4]
    plt.figure()
    plt.plot(grid_difference, label='Grid State Movement', color='green')
    plt.ylim(0, y_limit)
    plt.title(f'Phenotype Analysis - Grid State Movement \n{exp_name}')
    plt.xlabel('iterations')
    plt.ylabel('Count')
    plt.legend()
    plotname = f'{plot_dir}/grid_state_movement.pdf'
    plt.savefig(plotname)
    plt.close()


def process_data_needed_for_plot(ca_properties: dict, plot_dir: str):
    cell_rules_list = ca_properties['cell_rules']
    cell_states_list = ca_properties['cell_states']
    grid_states_list = ca_properties['grid_states']

    cell_states_dict = cell_states_count(cell_states_list)
    cumulative_rules_dict = unique_cumulative_rules(
        cell_rules_list)
    count_alive_dead_dict = count_1s_0s_dead_in_grid_state(
        grid_states_list, cell_states_list)
    grid_movement_dict = difference_betn_two_grid_states(
        grid_states_list)

    merged_dict = {**cell_states_dict, **
                   cumulative_rules_dict, **count_alive_dead_dict, **grid_movement_dict}
    csvname = f'{plot_dir}/plot_information.csv'
    pd.DataFrame.from_dict(merged_dict, orient='index').transpose().to_csv(
        csvname, index=False)
    return merged_dict


def visualize_data(ca_data: dict, plot_dir):
    y_limit_cs = max(ca_data['alive_cell_count'] +
                     ca_data['decay_cell_count'] + ca_data['dead_cell_count'])
    plot_cell_states_count(
        ca_data['alive_cell_count'], ca_data['decay_cell_count'], ca_data['dead_cell_count'], y_limit_cs,  plot_dir)

    plot_cumm_unique_rules_count(ca_data['cumulative_rule_count'], max(
        ca_data['cumulative_rule_count']), plot_dir)

    y_limit_grid = max(ca_data['count_alive_0'] +
                       ca_data['count_alive_1'] + ca_data['dead_cell_count'])
    plot_count_1s_0s_dead_in_grid_state(
        ca_data['count_alive_0'], ca_data['count_alive_1'], ca_data['dead_cell_count'], y_limit_grid, plot_dir)
    plot_difference_betn_two_grid_states(
        ca_data['grid_difference'], max(ca_data['grid_difference']),  plot_dir)


def get_data_for_each_exp_set(each_set_of_experiments: list):
    alive_cell_count = []
    decay_cell_count = []
    dead_cell_count = []
    cumulative_rule_count = []
    count_alive_0 = []
    count_alive_1 = []
    grid_difference = []
    for exp in each_set_of_experiments:
        alive_cell_count.append(exp['alive_cell_count'])
        decay_cell_count.append(exp['decay_cell_count'])
        dead_cell_count.append(exp['dead_cell_count'])
        cumulative_rule_count.append(exp['cumulative_rule_count'])
        count_alive_0.append(exp['count_alive_0'])
        count_alive_1.append(exp['count_alive_1'])
        grid_difference.append(exp['grid_difference'])

    return {
        'alive_cell_count': alive_cell_count,
        'decay_cell_count': decay_cell_count,
        'dead_cell_count': dead_cell_count,
        'cumulative_rule_count': cumulative_rule_count,
        'count_alive_0': count_alive_0,
        'count_alive_1': count_alive_1,
        'grid_difference': grid_difference
    }


def visualize_each_exp_set(each_set_of_experiments: list,
                           experiment_name: str,
                           iteration: int,
                           ylimit: dict,
                           plot_path: str):

    data = get_data_for_each_exp_set(each_set_of_experiments)

    each_exp_set_cell_state_count(data['alive_cell_count'],
                                    data['decay_cell_count'],
                                    data['dead_cell_count'],
                                    experiment_name,
                                    iteration,
                                    ylimit['max_cell_state_count'],
                                    plot_path)
    each_exp_set_visualize_cumulative_rule(
                                            data['cumulative_rule_count'],
                                            experiment_name, iteration,
                                            ylimit['max_cumulative_rules'],
                                            plot_path)
    each_exp_set_visualize_grid_state(
                                        data['count_alive_0'], 
                                        data['count_alive_1'], 
                                        data['dead_cell_count'], 
                                        experiment_name, 
                                        iteration, 
                                        ylimit['max_grid_state_count'], 
                                        plot_path)
    each_exp_set_visualize_grid_difference(
                                            data['grid_difference'], 
                                            experiment_name, 
                                            iteration, 
                                            ylimit['max_grid_difference'], 
                                            plot_path)

    return {experiment_name: data}


def each_exp_set_cell_state_count(alive_cell_count: list,
                                  decay_cell_count: list,
                                  dead_cell_count: list,
                                  experiment_name: str,
                                  iteration: int,
                                  ylimit: int,
                                  plot_path: str):

    x = np.arange(0, iteration)

    alive_mean = np.mean(alive_cell_count, axis=0)
    alive_std = np.std(alive_cell_count, axis=0)

    decay_mean = np.mean(decay_cell_count, axis=0)
    decay_std = np.std(decay_cell_count, axis=0)

    dead_mean = np.mean(dead_cell_count, axis=0)
    dead_std = np.std(dead_cell_count, axis=0)



    plt.figure()
    plt.plot(x, alive_mean, color='darkgreen',
              label='Alive Cell Count')
    plt.fill_between(x, alive_mean, alive_mean + alive_std, color='lightgreen')
    plt.fill_between(x, alive_mean, alive_mean - alive_std, color='lightgreen')

    plt.plot(x, decay_mean, color='orange',
              label='Decay Cell Count')
    plt.fill_between(x, decay_mean, decay_mean + decay_std, color='#FFDAB9')
    plt.fill_between(x, decay_mean, decay_mean - decay_std, color='#FFDAB9')

    plt.plot(x, dead_mean, color='black',
              label='Dead Cell Count')
    plt.fill_between(x, dead_mean, dead_mean + dead_std, color=(0, 0, 0, 0.3))
    plt.fill_between(x, dead_mean, dead_mean - dead_std, color=(0, 0, 0, 0.3))

    plt.ylim(0, ylimit)

    plt.xlabel('Iterations')
    plt.ylabel('Count')
    plt.title(f'Cell States Count \n{experiment_name}')
    plt.legend()
    plt.savefig(f'{plot_path}/cell_state_counts.pdf')

    plt.close()
    return plt


def each_exp_set_visualize_cumulative_rule(cumulative_rule_count: list,
                                           experiment_name: str,
                                           iteration: int,
                                           ylimit: int,
                                           plot_path: str):

    x = np.arange(0, iteration)
    rule_mean = np.mean(cumulative_rule_count, axis=0)
    rule_std = np.std(cumulative_rule_count, axis=0)

    plt.figure()
    plt.plot(x, rule_mean, color='blue',
             label='Cumulative Unique Rules')
    plt.fill_between(x, rule_mean, rule_mean + rule_std, color='lightblue')
    plt.fill_between(x, rule_mean, rule_mean - rule_std, color='lightblue')

    plt.ylim(0, ylimit)

    plt.xlabel('Iterations')
    plt.ylabel('Count')
    plt.title(f'Cumulative Cell Rules \n{experiment_name}')

    plt.legend()
    plt.savefig(f'{plot_path}/cummulative_unique_rules.pdf')
    plt.close()
    return plt



def each_exp_set_visualize_cumulative_rule1(cumulative_rule_count: list,
                                           experiment_name: str,
                                           iteration: int,
                                           ylimit: int,
                                           plot_path: str):

    x = np.arange(0, iteration)
    rule_mean = np.mean(cumulative_rule_count, axis=0)
    rule_std = np.std(cumulative_rule_count, axis=0)

    plt.figure()
    plt.plot(x, rule_mean, color='blue',
             label='Cumulative Unique Rules')
    plt.fill_between(x, rule_mean, rule_mean + rule_std, color='lightblue')
    plt.fill_between(x, rule_mean, rule_mean - rule_std, color='lightblue')

    plt.ylim(0, ylimit)

    plt.xlabel('Iterations')
    plt.ylabel('Count')
    plt.title(f'Cumulative Cell Rules \n{experiment_name}')
    plt.legend()
    return plt


def each_exp_set_visualize_grid_state(count_alive_0: list,
                                      count_alive_1: list,
                                      dead_cell_count: list,
                                      experiment_name: str,
                                      iteration: int,
                                      ylimit: int,
                                      plot_path: str):

    x = np.arange(0, iteration)

    alive_0_mean = np.mean(count_alive_0, axis=0)
    alive_0_std = np.std(count_alive_0, axis=0)

    alive_1_mean = np.mean(count_alive_1, axis=0)
    alive_1_std = np.std(count_alive_1, axis=0)

    dead_mean = np.mean(dead_cell_count, axis=0)
    dead_std = np.std(dead_cell_count, axis=0)

    plt.figure()
    plt.plot(x, alive_1_mean, color='darkgreen',
              label='Alive Cell Count')
    plt.fill_between(x, alive_1_mean, alive_1_mean +
                     alive_1_std, color='lightgreen')
    plt.fill_between(x, alive_1_mean, alive_1_mean -
                     alive_1_std, color='lightgreen')

    plt.plot(x, alive_0_mean, color='darkred',
              label='Decay Cell Count')
    plt.fill_between(x, alive_0_mean, alive_0_mean +
                     alive_0_std, color=(1, 0.5, 0.5, 0.5))
    plt.fill_between(x, alive_0_mean, alive_0_mean -
                     alive_0_std, color=(1, 0.5, 0.5, 0.5))

    plt.plot(x, dead_mean, color='black',
              label='Dead Cell Count')
    plt.fill_between(x, dead_mean, dead_mean + dead_std, color=(0, 0, 0, 0.3))
    plt.fill_between(x, dead_mean, dead_mean - dead_std, color=(0, 0, 0, 0.3))

    plt.ylim(0, ylimit)

    plt.xlabel('Iterations')
    plt.ylabel('Count')
    plt.title(
        f'Count Alive(0 and 1) and dead cell of Grid State \n{experiment_name}')
    plt.legend()
    plt.savefig(f'{plot_path}/grid_state_count_alive_1_0_dead_counts.pdf')
    plt.close()


def each_exp_set_visualize_grid_difference(grid_difference: list,
                                           experiment_name: str,
                                           iteration: int,
                                           ylimit: int,
                                           plot_path: str):

    

    grid_diff_mean = np.mean(grid_difference, axis=0)
    grid_diff_std = np.std(grid_difference, axis=0)
    x = np.arange(0, len(grid_diff_mean) )

    plt.figure()
    plt.plot(x, grid_diff_mean, color='darkgreen',
              label='Grid state movement')
    plt.fill_between(x, grid_diff_mean, grid_diff_mean +
                     grid_diff_std, color='lightgreen')
    plt.fill_between(x, grid_diff_mean, grid_diff_mean -
                     grid_diff_std, color='lightgreen')

    plt.ylim(0, ylimit)

    plt.xlabel('Iterations')
    plt.ylabel('Count')
    plt.title(
        f'Grid State Movement \n{experiment_name}')
    plt.legend()
    plt.savefig(f'{plot_path}/grid_state_movements.pdf')
    plt.close()
    return plt


def analyse_and_visualize_data(ca_properties: dict, plot_dir: str):
    ca_data = process_data_needed_for_plot(ca_properties, plot_dir)
    visualize_data(ca_data, plot_dir)
