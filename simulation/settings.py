# Define global variables
DATA_LOCAL_FOLDER = '/mimer/NOBACKUP/groups/two_second_football'
seasons = ['2022', '2023']
competitions = ['Allsvenskan']
reload_data = True
seconds_into_the_future = 2
FPS = 25
pitch_length = 105
pitch_width = 68
normalize = True

# Define the columns that we want to predict
y_cols = ['x_future', 'y_future']             # Standard y_cols for 2 seconds model
# y_cols = ['x_future_25', 'y_future_25']     # Use this if you want 1 second model
# y_cols = ['x_future_50', 'y_future_50']     # Use this if you want 2 seconds model
# y_cols = ['x_future_75', 'y_future_75']     # Use this if you want 3 seconds model

# Train, test, val
train_size = 0.7
test_size = 0.1
val_size = 0.2
random_state = 42

# Original colors
ajax_red = '#e40615'
ajax_white = '#ffffff'
ajax_yellow = '#f7be2f'
ajax_grey = '#eeeeee'
ajax_black = '#000000'

# Lighter shades
ajax_red_light = '#ff9ea8'
ajax_white_light = '#e9e9e9'
ajax_yellow_light = '#f7de97'
ajax_black_light = '#424242'