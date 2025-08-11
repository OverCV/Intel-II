import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes=1000, is_training=True, render=False):
    env =  gym.make('MountainCar-v0', render_mode='human' if render else None)
    
    # Divide position and velocity into segments
    position_segments = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    velocity_segments = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)
    
    # Initialize parameters
    learning_rate = 0.1
    discount_rate = 0.99
    epsilon = 1.0
    random_number_generator = np.random.default_rng()
    epsilon_decay = 2/episodes
    
    rewards_per_episode = np.zeros(episodes)
    
    # Initialize Q-table
    
    
    if is_training:
        q_table = np.zeros((len(position_segments), len(velocity_segments), env.action_space.n))
    else:
        f = open('mountain_car.pkl', 'rb')
        q_table = pickle.load(f)
        f.close()
    
    for episode in range(1000):
        state = env.reset()[0]
        state_position_index = np.digitize(state[0], position_segments)
        state_velocity_index = np.digitize(state[1], velocity_segments)
        
        terminated = False
        
        rewards = 0
        
        while(not terminated and rewards > -1000):
            if is_training and random_number_generator.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_position_index, state_velocity_index, :])
            
            new_state, reward,terminated,_,_ = env.step(action)
            
            new_state_position_index = np.digitize(new_state[0], position_segments)
            new_state_velocity_index = np.digitize(new_state[1], velocity_segments)
            
            if is_training:
                q_table[state_position_index, state_velocity_index, action] = q_table[state_position_index, state_velocity_index, action] + learning_rate * (
                    reward + discount_rate * np.max(q_table[new_state_position_index, new_state_velocity_index, :]) - q_table[state_position_index, state_velocity_index, action])
            
            state = new_state
            state_position_index = new_state_position_index
            state_velocity_index = new_state_velocity_index
            
            rewards += reward
            
        epsilon = max(epsilon - epsilon_decay, 0)
        rewards_per_episode[episode] = rewards

    env.close()
    
    if is_training:
        f = open('mountain_car.pkl', 'wb')
        pickle.dump(q_table, f)
        f.close()
    
    mean_rewards = np.zeros(episodes)
    for i in range(episodes):
        mean_rewards[i] = np.mean(rewards_per_episode[max(0, i-100):i+1])
    
    plt.plot(mean_rewards)
    plt.savefig('mountain_car.png')
    
if __name__ == '__main__':
    run(1000, False, True)