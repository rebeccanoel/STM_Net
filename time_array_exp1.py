'''
Timestep Array Fxn: Exp. 1
'''

import numpy as np
import Constants_HyperPara as con

def digit_finder(image, data_array, label_array):
	'''
	helper function that returns the label of an input image
	'''
	bool_array = np.isin(data_list, image)
	position = np.where(bool_array)
	digit = label_array[position]
	return digit
'''
def gen_timestep_array(data, labels, num_steps, temporal_order):
    num_blank_timesteps = num_steps - 1
    expanded_tensor = np.expand_dims(data, axis = -1)
    shape = np.shape(expanded_tensor)
    blank_data = np.zeros(shape = shape, dtype = np.float32)
    repeat = np.repeat(blank_data,num_blank_timesteps, axis = -1)

    final_dataset = np.concatenate((expanded_tensor,repeat),axis = -1)
    
    return final_dataset
'''

def gen_timestep_array(data, labels, num_steps, temporal_order):
    num_blank_timesteps = num_steps - 1
    expanded_tensor = np.expand_dims(data, axis = -1)
    expanded_labels = np.expand_dims(labels, axis = -1)
    data_shape = np.shape(expanded_tensor)
    labels_shape = np.shape(expanded_labels)
    blank_data = np.zeros(shape = data_shape, dtype = np.float32)
    blank_labels = np.zeros(shape = labels_shape, dtype = np.float32)
    data_repeat = np.repeat(blank_data,num_blank_timesteps, axis = -1)
    label_repeat = np.repeat(blank_labels,num_blank_timesteps, axis = -1)
    final_dataset = np.concatenate((expanded_tensor,data_repeat),axis = -1)
    final_labels = np.concatenate((expanded_labels,label_repeat),axis = -1)
    return final_dataset, final_labels
