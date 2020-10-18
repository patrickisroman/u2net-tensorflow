import pathlib

# Model
resume = None
weight_dir = pathlib.Path('weights').absolute()
weights_file = weight_dir.joinpath('u2net.h5')
default_in_shape = (320, 320, 3)
default_out_shape = (320, 320, 1)

# Training
batch_size = 12
epochs = 10000
learning_rate = 0.001
save_interval = 1000

# Dataset 
dataset_url = 'http://saliencydetection.net/duts/download/DUTS-TR.zip'
current_location = pathlib.Path(__file__).absolute().parents[0]
root_data_dir = pathlib.Path('data')
dataset_dir = root_data_dir.joinpath('DUTS-TR')
image_dir = dataset_dir.joinpath('DUTS-TR-Image')
mask_dir = dataset_dir.joinpath('DUTS-TR-Mask')

# Evaluation
output_dir = pathlib.Path('out')