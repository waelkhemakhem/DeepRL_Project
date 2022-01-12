import numpy as np



"""
A class made to store agent's experiences in a replay buffer
"""

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        # Initializing memory size(mem_size) and index pointer(mem_cntr)
        self.mem_size = max_size

        self.mem_cntr = 0
        # Defining matrix containers to save the states, new states, actions, rewards and done_boolean values
        self.state_memory = np.zeros((self.mem_size, *input_shape),dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    # save transitions(s,a,r,s',done) inside each of the memory buffer
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    # A function used to sample an interval of data from the memory buffer
    def sample_buffer(self, batch_size):
        # choosing a sampling size to extract a batch from the buffer
        max_mem = min(self.mem_cntr, self.mem_size)
        # choosing random samples from the buffer memory
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
