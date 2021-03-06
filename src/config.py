"""
Variables set by the command line arguments dictating which parts of the program to execute.
"""

# Constants
RANDOM_SEED = 111
fontsizes = {
    'title': 15,
    'axis': 14,
    'ticks': 11
}
binary_labels = ["background", "seal"]
multi_labels = ["background", "whitecoat", "moulted pup", "dead pup", "juvenile"]

# Variables set by command line arguments/flags
section = "data_vis"            # The section of the code to run.
dataset = "binary"              # The dataset to use.
model = "mlp"                   # The classification model to train/test (default is mlp).
is_grid_search = False          # Run the grid search algorithm to determine the optimal hyper-parameters for the model.
is_randomised_search = False    # Run a randomised search  to determine the optimal hyper-parameters for the model.
verbose_mode = False            # Boolean used to print additional logs for debugging purposes.
