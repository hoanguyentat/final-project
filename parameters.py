# Hyperparameter
growth_k = 48
data_path = 'data/wiki_db_32.mat'
# Number of (dense block + Transition Layer)
nb_block = 3
init_learning_rate = 0.1
# AdamOptimizer epsilon
epsilon = 1e-4

dropout_rate = 0.2
class_num = 2
image_size = 32
img_channels = 3

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

# Label & batch_size
batch_size = 100

train_fraction = 0.8
test_fraction = 0.2

test_batch_size = 100
total_epochs = 100