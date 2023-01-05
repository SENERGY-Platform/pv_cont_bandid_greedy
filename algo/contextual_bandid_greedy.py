import numpy as np

def update_design_matrix(design_matrix, new_weather_input, weather_dim):
    if design_matrix == None:
        design_matrix = new_weather_input.reshape((1,weather_dim))
    else:
        design_matrix = np.vstack((design_matrix, new_weather_input))
    return design_matrix

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