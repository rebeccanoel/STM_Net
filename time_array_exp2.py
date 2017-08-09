'''
Timestep Array Fxn: Exp. 2
2 images separated by blank stimuli
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

def gen_distractor_array(data_array, label_array):
	#iterates through all of the images in the dataset
	distractor = []
	d_labels = []
	for image in data_array:
		label = digit_finder(image, data_array, label_array)
		need_distractor = True
		for d_image in data_array:
			while need_distractor == True:
				dis_label = digit_finder(d_image, data_array, label_array)
				if dis_label != label:
					distractor = np.concatenate((distractor, d_image))
					d_labels = np.concatenate((d_labels,dis_label))
					need_distractor = False
	return distractor, d_labels


def gen_timestep_array(data, labels, num_steps, temporal_order):
    #num_blanks_btwn_images = num_steps - 1
    expanded_tensor = np.expand_dims(data, axis = -1)
    expanded_labels = np.expand_dims(labels, axis = -1)
    shape = np.shape(expanded_tensor)
    blank_data = np.zeros(shape = shape, dtype = np.float32)
    final_dataset = expanded_tensor
    final_labels = expanded_labels
    if 2 in temporal_order:
    	d_set, d_labels = gen_distractor_array(data, labels)
    	distractor_set = np.expand_dims(d_set, axis = -1)
    	distractor_labels = np.expand_dims(d_labels, axis = -1)
    for obj in temporal_order:
    	if obj not in [1,2,3]:
    		raise ValueError("Valid inputs for temporal_order are 1, 2, or 3")
    	'''
    	if obj == 1:
    		final_dataset = np.concatenate((final_dataset,data),axis = -1)
    		final_labels = np.concatenate((final_labels, labels),axis = -1)
    	'''
    	if obj == 2:
    		final_dataset = np.concatenate((final_dataset,distractor_set),axis = -1)
    		final_labels = np.concatenate((final_labels, distractor_labels),axis = -1)
    	if obj == 3:
    		final_dataset = np.concatenate((final_dataset,blank_data))
			#make a distractor that is a different digit from the target image
    #final_dataset = np.concatenate((expanded_tensor,repeat),axis = -1)
    
    return final_dataset


#rather than introducing a num_distractor 
#(for the possiblity of including multiple distractors)
#it might be better to generate further distractors by 
#passing the distractor array itsself as the data_array 
#argument into the gen_distractor_array function 