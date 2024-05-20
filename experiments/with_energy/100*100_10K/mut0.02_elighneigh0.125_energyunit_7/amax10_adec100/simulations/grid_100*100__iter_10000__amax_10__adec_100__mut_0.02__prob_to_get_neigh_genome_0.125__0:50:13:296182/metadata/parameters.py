'''
    PARAMETERS: THIS IS JUST A COPY OF MAIN GLOBALS FILE AND CONFIGURED ACCORDING TO THE EXPERIMENT SETUP. 
    IF YOU WANT DESCRIPTION OF VARIABLES, LOOK INTO THE PROJECT FILE 
'''

# STANDARD PARAMETERS
DEFAULT_INITIAL_CELL_STATES = False
GRID_SIZE = 100
ITERATION = 10000
ELIGIBLE_CELL_STATES = ['a']
INITIAL_ALIVE_CELL_STATE_RATIO = 0.5
ALIVE_CELL_INTIAL_1_VALUE_PROB = 0.5
INITIAL_RULE = 'B3S23'


# VARIABLE PARAMETERS
AMAX = 10
ADEC = 50
MUTATION_RATE = 0.02
PROB_TO_GET_ELIGIBLE_NEIGHBOURS = 0.125


# ENERGY VARIABLES
ENERGY_DISTRIBUTION_RATIO = 0.02
LOWEST_ENERGY_LEVEL = 25
HIGHEST_ENERGY_LEVEL = 25
ENERGY_DEPLETION = 1
ENERGY_STARTING_PREFERED_LIVING_STATE = 0
ENERGY_INTERVAL =50

# OTHER VARIABLES
ANIM_CELL_STATE = "animation_cell_state.mp4"
ANIM_GRID_STATE = "animation_cell_grid.mp4"
RULES_ANIM_FILE_NAME = "animation_cell_rule.mp4"
ANIM_COLLAGE = "anim_collage.mp4"
QUIESCENT_COLOR = [255, 255, 255]
DECAY_COLOR = [255, 165, 0]
DURATION = 450
