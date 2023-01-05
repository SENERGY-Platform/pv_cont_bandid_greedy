import numpy as np

def update_design_matrices(design_matrix_0, design_matrix_1, new_weather_input, action):
    if action==0:
        if design_matrix_0 == None:
            design_matrix_0 = new_weather_input.reshape((1,-1))
        else:
            design_matrix_0 = np.vstack((design_matrix_0, new_weather_input))
    elif action==1:
        if design_matrix_1 == None:
            design_matrix_1 = new_weather_input.reshape((1,-1))
        else:
            design_matrix_1 = np.vstack((design_matrix_1, new_weather_input))
    return design_matrix_0, design_matrix_1

def update_reward_vector(reward_vector, new_reward):
    return reward_vector

def update_actions(actions, betas, new_weather_input):
    estimated_reward_0 = np.dot(new_weather_input, betas[0])
    estimated_reward_1 = np.dot(new_weather_input, betas[1])
    action = np.argmax([estimated_reward_0, estimated_reward_1])
    actions.append(action)
    return actions

def update_betas(actions, betas, design_matrix):
    return betas