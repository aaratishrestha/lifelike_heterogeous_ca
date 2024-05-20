########################### GOL WITH CELL THREE STATES: ALIVE, DECAY AND QUIESCENT      #####################

import copy
from utils import visualisation
from utils import globals
import life_like_ca as ca
import os
from datetime import datetime
import numpy as np
from os import system

RESULT_DIR = './animation'


def get_eligible_cell_indices_to_get_energy(cell_states: np.ndarray, 
                                            grid_states: np.ndarray, 
                                            prefered_living_state: int, 
                                            distribution_ratio: float):
    
    eligible_mask = (grid_states == prefered_living_state) & (
        cell_states == 'a')
    num_of_elements_for_energy = int(
        distribution_ratio * np.sum(eligible_mask))
    eligible_indices = np.argwhere(eligible_mask)
    selected_indices = eligible_indices[np.random.choice(
        len(eligible_indices), size=num_of_elements_for_energy, replace=False)]
    return selected_indices


def generate_energy(lowest_cell_energy: int, highest_energy_level: int, cell_states: np.ndarray, grid_states: np.ndarray, distribution_ratio: float, prefered_living_state: int):
    values = np.arange(lowest_cell_energy, highest_energy_level + 1)
    # get the list ofnumbers from 1 to 0.1 where the gap between two numbers are same.
    probability_distribution = np.linspace(1, distribution_ratio, len(values))
    probability_distribution /= probability_distribution.sum()

    new_energy = np.zeros(cell_states.shape,  dtype=int)
    selected_indices = get_eligible_cell_indices_to_get_energy(cell_states, grid_states, prefered_living_state, distribution_ratio)
    # Sample from energy based on probabilities
    sampled_values = np.random.choice(
        values, size=len(selected_indices), p=probability_distribution)
    # Place sampled values in the new_energy
    for idx, value in zip(selected_indices, sampled_values):
        new_energy[idx[0], idx[1]] = value
    
    return new_energy


def create_needed_folders(param: dict):

    # main directory
    result_path = param['result_dir']
    amax_val = param['amax']
    adec_val = param['adec']
    mut_val = param['mutation_rate']
    prob_to_get_neigh_genome_val = param['probability_to_get_neighbour_genome']
    grid_size = param['cell_states'].shape[0] if 'cell_states' in param else param['grid_size']
    iteration = param['iteration']

    grid = f'grid_{grid_size}*{grid_size}'
    amax = f'amax_{amax_val}'
    adec = f'adec_{adec_val}'
    mut = f'mut_{mut_val}'
    prob_to_get_neigh_genome = f'prob_to_get_neigh_genome_{prob_to_get_neigh_genome_val}'
    iter = f'iter_{iteration}'

    now = datetime.now()
    current_year = str(now.year)
    current_month = str(now.month)
    current_day = str(now.day)
    date_dir = f'{result_path}/{current_year}_{current_month}_{current_day}'
    current_hour = str(now.hour)
    current_minute = str(now.minute)
    current_second = str(now.second)
    current_milli_second = str(now.microsecond)
    cur_time = f'{current_hour}:{current_minute}:{current_second}:{current_milli_second}'

    ca_output_dir = f'{date_dir}/{grid}__{iter}__{amax}__{adec}__{mut}__{prob_to_get_neigh_genome}__{cur_time}'
    metadata_dir = f'{ca_output_dir}/metadata'
    animation_dir = f'{ca_output_dir}/animations'
    plot_dir = f'{ca_output_dir}/plots'

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(date_dir, exist_ok=True)
    os.makedirs(ca_output_dir, exist_ok=True)
    param['exp_result_dir'] = ca_output_dir
    os.makedirs(metadata_dir, exist_ok=True)
    param['metadata_dir'] = metadata_dir
    os.makedirs(animation_dir, exist_ok=True)
    param['animation_dir'] = animation_dir
    os.makedirs(plot_dir, exist_ok=True)
    param['plot_dir'] = plot_dir


def run_ca(param):
    cell_states, cell_ages, cell_rules, grid_states, cell_energy = [], [], [], [], []
    new_energies = []

    iteration = param['iteration']
    starting_living_state = param['starting_living_state']
    alternate_living_state =  starting_living_state ^ 1
    energy_interval = param['energy_interval']

    llca = ca.LLCA(param)
    prefered_living_state = starting_living_state
    new_energy = generate_energy(param['lowest_energy_level'], 
                                    param['highest_energy_level'], 
                                    llca.cell_states, 
                                    llca.grid_state,
                                    param['energy_distribution_ratio'],
                                    prefered_living_state)
    new_energies.append(new_energy)
    llca.energy += copy.deepcopy(new_energy)

    cell_states.append(copy.deepcopy(llca.cell_states))
    cell_ages.append(copy.deepcopy(llca.cell_ages))
    cell_rules.append(copy.deepcopy(llca.cell_rules))
    grid_states.append(copy.deepcopy(llca.grid_state))
    cell_energy.append(copy.deepcopy(llca.energy))
    print("Processing for ", iteration, " iteration")

    for i in range(1, iteration):
        print("Started processing of ", i, " iteration")
        llca.iterate()
        prefered_living_state = starting_living_state if (i // energy_interval % 2 == 0) else alternate_living_state
        new_energy = generate_energy(param['lowest_energy_level'], 
                                    param['highest_energy_level'], 
                                    llca.cell_states, 
                                    llca.grid_state,
                                    param['energy_distribution_ratio'],
                                    prefered_living_state)
        new_energies.append(new_energy)
        llca.energy += copy.deepcopy(new_energy)

        cell_states.append(copy.deepcopy(llca.cell_states))
        cell_ages.append(copy.deepcopy(llca.cell_ages))
        cell_rules.append(copy.deepcopy(llca.cell_rules))
        grid_states.append(copy.deepcopy(llca.grid_state))
        cell_energy.append(copy.deepcopy(llca.energy))
    
    return {'cell_states': cell_states,
            'cell_ages': cell_ages,
            'cell_rules': cell_rules,
            'grid_states': grid_states,
            'cell_energy': cell_energy}


def set_default_parameters(other_params=None):
    param = {}
    param['grid_size'] = globals.GRID_SIZE
    param['iteration'] = globals.ITERATION
    param['amax'] = globals.AMAX
    param['adec'] = globals.ADEC
    param['mutation_rate'] = globals.MUTATION_RATE

    # default_initial_cell_state is set to False if no initial cell states are passed,
    # otherwise it needs to be set True along with 'cell states'
    param['default_initial_cell_state'] = globals.DEFAULT_INITIAL_CELL_STATES
    if param['default_initial_cell_state']:

        cell_states = globals.DEFAULT_CELL_STATES
        grid_states = globals.DEFAULT_GRID_STATES
        q_check = len(np.where((cell_states == 'q') & (grid_states == 1))[0])

        if (cell_states.shape != grid_states.shape) or q_check > 0:
            print('Cell states properties do not align with Grid states properties')
            print('Exiting...')
            exit()

        param['cell_states'] = cell_states
        param['grid_states'] = grid_states
    else:
        grid_size = (globals.GRID_SIZE, globals.GRID_SIZE)
        probability = [globals.INITIAL_ALIVE_CELL_STATE_RATIO,
                       1 - globals.INITIAL_ALIVE_CELL_STATE_RATIO]
        cell_states = np.random.choice(
            ['a', 'q'], size=grid_size, p=probability)

        grid_states = np.zeros(cell_states.shape, dtype=int)
        a_positions = cell_states == 'a'
        grid_prob = [1 - globals.ALIVE_CELL_INTIAL_1_VALUE_PROB,
                     globals.ALIVE_CELL_INTIAL_1_VALUE_PROB]
        grid_states[a_positions] = np.random.choice(
            [0, 1], size=a_positions.sum(), p=grid_prob)
        # making double sure, not needed at all.
        grid_states[cell_states == 'q'] = 0

        param['cell_states'] = cell_states
        param['grid_states'] = grid_states

    param['probability_to_get_neighbour_genome'] = globals.PROB_TO_GET_ELIGIBLE_NEIGHBOURS
    param['result_dir'] = RESULT_DIR
    param['grid_size'] = globals.GRID_SIZE
    param['initial_rule'] = globals.INITIAL_RULE
    param['eligible_cell_states'] = globals.ELIGIBLE_CELL_STATES

    # Energy configurations
    param['energy_depletion'] = globals.ENERGY_DEPLETION
    param['lowest_energy_level'] = globals.LOWEST_ENERGY_LEVEL
    param['highest_energy_level'] = globals.HIGHEST_ENERGY_LEVEL
    param['energy_distribution_ratio'] = globals.ENERGY_DISTRIBUTION_RATIO
    param['starting_living_state'] = globals.ENERGY_STARTING_PREFERED_LIVING_STATE
    param['energy_interval'] = globals.ENERGY_INTERVAL


    # File names
    param['cellstate_filename'] = globals.ANIM_CELL_STATE
    param['gridstate_filename'] = globals.ANIM_GRID_STATE
    param['cellrule_filename'] = globals.RULES_ANIM_FILE_NAME

    if other_params:
        param = param | other_params

    return param

def start_process():
    start_datetime = datetime.now()
    print('STARTED APPLICATION AT: ', start_datetime)
    params = set_default_parameters() # Set default parameters
    create_needed_folders(params)
    ca_properties = run_ca(params) # Run CA
    visualisation.visual_result(ca_properties, params)
    end_datetime = datetime.now()
    print('ENDED APPLICATION AT: ', end_datetime)
    print('Total Time Taken',  end_datetime - start_datetime)
    print("COMPLETE")

if __name__ == "__main__":
    start_process()
