'''
    The files contains all the variables used in this project
'''
import numpy as np

########################### BASIC CONFIGURATION ###########################
DEFAULT_INITIAL_CELL_STATES = False  # Provide default cell state value rather than generating it randomly
GRID_SIZE = 3
ITERATION = 20
AMAX = 4
ADEC = 7
INITIAL_RULE = 'B3S23'
########################### GRID STATE CONFIGURATION ENDED ###########################


#DEFAULT VALUES ARE ASSIGNED ONLY WHEN 'DEFAULT_INITIAL_CELL_STATES' IS SET TO TRUE


DEFAULT_GRID_STATES = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)


DEFAULT_CELL_STATES = np.full(DEFAULT_GRID_STATES.shape, 'a', dtype=str)


# DEFAULT_CELL_STATES = np.full((GRID_SIZE, GRID_SIZE), 'q')
# DEFAULT_CELL_STATES[DEFAULT_CELL_STATES.shape[0] // 2, DEFAULT_CELL_STATES.shape[1] // 2] = 'a'

# DEFAULT_GRID_STATES = np.zeros((50, 50), dtype=np.int8)


########################### MUTATION CONFIGURATION ###########################
MUTATION_RATE = 0.5

########################### MUTATION CONFIGURATION ENDED ###########################

PROB_TO_GET_ELIGIBLE_NEIGHBOURS = 0.5

########################### CELL STATE CONFIGURATION ###########################
# The variable defines which cell state to consider when getting eligible neighbours for a given quiescent cell
ELIGIBLE_CELL_STATES = ['a']

# value lies between 0 to 1 where 0 refers to 0 alive cell and 1 refers to all alive cells in the first grid
INITIAL_ALIVE_CELL_STATE_RATIO = 0.5

# The quiescent cell has 0 grid value while alive cells can have 0 or 1 grid value when the grid state is initialized for the first time
# If the parameter is set to 1 all alive cells have 1 value while if it set 0 all alive cells have 0 value
ALIVE_CELL_INTIAL_1_VALUE_PROB = 0.5
########################### CELL STATE CONFIGURATION ENDED###########################



########################### ENERGY CONFIGURATION ###########################
# Lowest level of energy a cell can receive
LOWEST_ENERGY_LEVEL = 0 

# Highest level of energy a cell can receive
HIGHEST_ENERGY_LEVEL = 10

# determines in which iteration to distribute the energy. Value should be from 0 to 1. say 0.1 is given, then, energy is 
# energy is distributed to the 0.1 iterations of total iterations
ENERGY_ITERATION_RATIO = 0



# determines how much energy to distribute in the whole grid.
# value must be from 0 to 1 where 0 implies not any cell receive energy in a given grid
# where as 1 implies all cells receive energy in the given grid.
ENERGY_DISTRIBUTION_RATIO = 0.1



# Energy depletion parameter refers to how much energy can be depleted in each iteration
ENERGY_DEPLETION = 1
########################### ENERGY CONFIGURATION END ###########################

########################### ANIMATION RELATED CONFIGURATION ###########################
# Directory to save the animation
# RESULT_DIRECTORY = '/Users/a02065/masters/thesis/test_animation'
RESULT_DIRECTORY = './animation'

# Name of animations
ANIM_CELL_STATE = "animation_cell_state.gif"
ANIM_GRID_STATE = "animation_cell_grid.gif"
RULES_ANIM_FILE_NAME = "animation_cell_rule.gif"
ANIM_COLLAGE = "anim_collage.gif"

# The dictionary is used to store a unique color value to 
QUIESCENT_COLOR = [255, 255, 255]
DECAY_COLOR = [255, 165, 0]

#  Display duration of each frame in the generated animation
DURATION = 450

