'''
    PARAMETERS: THIS IS JUST A COPY OF MAIN GLOBALS FILE AND CONFIGURED ACCORDING TO THE EXPERIMENT SETUP. 
    IF YOU WANT DESCRIPTION OF VARIABLES, LOOK INTO THE PROJECT FILE 
'''

#The parameter determine whether or now to use random grid and cell state in the first generation.
# If sets true, two additional parameters(2D numpy array of same size) need to passed, DEFAULT_GRID_STATES and DEFAULT_CELL_STATES
DEFAULT_INITIAL_CELL_STATES = False

# The CA uses squared grid.
GRID_SIZE = 100

# The number of generations
ITERATION = 5000

# The parameter defines eligible neighbouring cell states for inheritance.
# Eligible cell states can be both alive(a) and decay(d)
ELIGIBLE_CELL_STATES = ['a']

# The parameter controls the number of alive and dead cell in the first generation.
# Values lie between 0 to 1 where 0 refers to 0 alive cell and 1 refers to all alive cells in the first generation of the grid
INITIAL_ALIVE_CELL_STATE_RATIO = 0.5 

# The parameter controls the number of 1s and 0s of the alive cell in the first generation.
# Values lie between 0 to 1 where 0 refers to only 0 value of alive cell and 1 refers only 1 value of alive cells throughout the grid in the first generation of the grid
ALIVE_CELL_INTIAL_1_VALUE_PROB = 0.5

# Grid starts with a homogeneous rule. The parameter defines which homogenous rule to use in the first generation.
INITIAL_RULE = 'B3S23'


AMAX = 10
ADEC = 125


########################### INHERITANCE CONFIGURATION ###########################
MUTATION_RATE = 0.02
PROB_TO_GET_ELIGIBLE_NEIGHBOURS = 0.125


########################### ENERGY CONFIGURATION ###########################

# Energy distribution to the number of cell in the grid where 0 represents no energy at all to any cells 
# while 1 represents energy to all the cells in the grid.
ENERGY_DISTRIBUTION_RATIO = 0.02
LOWEST_ENERGY_LEVEL = 25    # The lowest energy level cells can get
HIGHEST_ENERGY_LEVEL = 25   # The highest energy level cells can get
ENERGY_DEPLETION = 1        # Energy depletion per generation

# The parameter determines the interval for distributing energy. 
# It specifies the duration for allocating energy to one state before switching to another
ENERGY_INTERVAL = 50        # The 
ENERGY_STARTING_PREFERED_LIVING_STATE = 0      #If energy is given in interval on the basis of grid state, the parameter defines which state to give energy in first interval.


# VARIABLES FOR VISUALIZATION
ANIM_CELL_STATE = "animation_cell_state.mp4"
ANIM_GRID_STATE = "animation_cell_grid.mp4"
RULES_ANIM_FILE_NAME = "animation_cell_rule.mp4"
ANIM_COLLAGE = "anim_collage.mp4"
QUIESCENT_COLOR = [255, 255, 255]
DECAY_COLOR = [255, 165, 0]
DURATION = 450
FPS = 50
