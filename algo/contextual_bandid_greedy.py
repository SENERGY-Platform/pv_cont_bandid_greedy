import numpy as np
from sklearn.linear_model import LinearRegression

def update_design_matrices(design_matrix_0, design_matrix_1, new_weather_input, action):
    if action==0:
        if design_matrix_0.shape == (0,0):
            design_matrix_0 = new_weather_input.reshape((1,-1))
        else:
            design_matrix_0 = np.vstack((design_matrix_0, new_weather_input))
    elif action==1:
        if design_matrix_1.shape == (0,0):
            design_matrix_1 = new_weather_input.reshape((1,-1))
        else:
            design_matrix_1 = np.vstack((design_matrix_1, new_weather_input))
    return design_matrix_0, design_matrix_1

def update_reward_vector(reward_vector, new_reward):
    return reward_vector

def update_actions(actions, beta_0, beta_1, new_weather_input):
    estimated_reward_0 = np.dot(new_weather_input, beta_0)
    estimated_reward_1 = np.dot(new_weather_input, beta_1)
    action = np.argmax([estimated_reward_0, estimated_reward_1])
    actions.append(action)
    return actions

def update_betas(action, beta_0, beta_1, num_finished_agents_0, num_finished_agents_1, design_matrix_0, design_matrix_1, rewards_0, rewards_1):
    if action==0 and num_finished_agents_0 > 0:
        rewards_0_tr = np.array(rewards_0[:num_finished_agents_0]).reshape((-1,1))
        design_matrix_0_tr = design_matrix_0[:num_finished_agents_0]

        regressor = LinearRegression()
        regressor.fit(design_matrix_0_tr, rewards_0_tr)
        beta_0 = regressor.coef_
    elif action==1 and num_finished_agents_1 > 0:
        rewards_1_tr = np.array(rewards_1[:num_finished_agents_1]).reshape((-1,1))
        design_matrix_1_tr = design_matrix_1[:num_finished_agents_1]

        regressor = LinearRegression()
        regressor.fit(design_matrix_1_tr, rewards_1_tr)
        beta_1 = regressor.coef_
    return beta_0, beta_1