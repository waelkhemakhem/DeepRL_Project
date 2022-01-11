import collections
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gym

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


"""
We can modify specific aspects of the environment by using subclasses of gym.
"gym.Wrapper" that override how the environment processes observations, rewards, and action.
"""
class RepeatActionAndMaxFrame(gym.Wrapper):
	"""
	Atari 2600 games have a flickering effect, which is due to the platform’s limitation of rendering frames.
	The solution is to take the maximum of every pixel in every two frames and use it as an observation.
	"""
    def __init__(self, env=None, repeat=2):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2,self.shape))

    # Overriding the step superclass method
    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            self.frame_buffer[i] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info
    
    # Overriding the reset superclass method
    def reset(self):
        obs = self.env.reset()
        self.frame_buffer = np.zeros_like((2,self.shape))
        self.frame_buffer[0] = obs
        return obs


"""
 "gym.ObservationWrapper" used to modify the observations returned by the environment. To do this, override the observation method of 
 the environment. This method accepts a single parameter (the observation to be modified) 
 and returns the modified observation.
"""
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        # defining the window dimensions
        self.shape=(shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0, high=1.0,
                                              shape=self.shape,dtype=np.float32)
    def observation(self, obs):
    	# reducing complexity by grayscaling the image (from RGB to Grayscale)
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resizing the image to (84,84,1) to reduce complexity and speeden up the learning process
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)

        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = np.swapaxes(new_obs, 2,0)
        # Normalizing pixels
        new_obs = new_obs / 255.0
        return new_obs

"""
the class StackFrames several (four) subsequent frames together
 and use them as the observation at every state. For this reason, 
 the preprocessing stacks four frames together resulting in a final state space size of 84 by 84 by 4.
"""
class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                             env.observation_space.low.repeat(n_steps, axis=0),
                             env.observation_space.high.repeat(n_steps, axis=0),
                             dtype=np.float32)

        self.stack = collections.deque(maxlen=n_steps)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)
        obs = np.array(self.stack).reshape(self.observation_space.low.shape)

        return obs

# get the env from gym => overide to RepeatActionAndMaxFrame => overide => PreprocessFrame => StackFrames
def make_env(env_name, shape=(84,84,1), skip=4):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, skip)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, skip)

    return env
