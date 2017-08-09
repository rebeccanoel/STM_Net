'''
Generate Timestep Array Function
'''
import numpy as np
import Constants_HyperPara as con

def gen_timestep_array(data, num_steps, temporal_order):
    expanded_tensor = np.expand_dims(data, axis = -1)
    repeat = np.repeat(expanded_tensor,con.num_steps, axis = -1)
    return repeat