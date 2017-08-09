'''
Constants & Hyperparameters
'''
#add an experiment ID with the timestamp in order to catalog each experiment
#id=1

image_size = 28
num_labels = 10
num_channels = 1 # grayscale
batch_size = 16

#the following included to bridge RNN and LeNet Code with diff variable names
input_size = image_size**2
num_classes = num_labels
state_size = 2000
num_batches = 2000
#where 1 = target, 2 = distractor, 3 = blank
temporal_order = [1,3]
#the number of timesteps that the input data will be shown
num_steps = len(temporal_order)
#the total number of timesteps for which an image will be shown
num_timesteps = 1
image_size = 28
patch_size = 5


image_size = 28


kernelSize = 5
depth1Size = 6
depth2Size = 16
num_channels = 1

padding="SAME"
convStride = 1
poolStride = 2
poolFilterSize = 2

FC1HiddenUnit = 360
FC2HiddenUnit = 784

learningRate=1e-4

