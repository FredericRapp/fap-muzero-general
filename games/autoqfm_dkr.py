import numpy as np
#dkr imports 
import dill as pickle
from squlearn.expectation_operator import SinglePauli
from squlearn.kernel.matrix import ProjectedQuantumKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.kernel_ridge import KernelRidge
# takes packages files from the specified folder
import os
# import sys
# sys.path.insert(0, os.path.join(os.getcwd(), '..','..'))
from squlearn.optimizers import SPSA,Adam
from squlearn.optimizers.approximated_gradients import StochasticPerturbationGradient
from games.generate_quantum_data import create_quantum_fmnist

gamma = 0.2
num_components = 4
category_filter_list = [2,0]
num_training, num_test = 100, 100
X_train,Y_train,X_test,Y_test = create_quantum_fmnist(
    num_training, num_test, num_components=num_components,category_filter_list=category_filter_list, num_su2_layers=0, su2_random_seed=42)

# lets try a QML approach based on a random feature map 
from squlearn.feature_map.layered_feature_map import LayeredFeatureMap
from squlearn.util import Executor

num_qubits = 8
max_num_gates = 8
max_steps = 12

actions = [         # action integer number
    "remove_first", #0
    "remove_last",  #1
    "",             #2
    "H",            #3
    f"Rx(x;=np.pi*(x),{{x}})",        #4
    f"Ry(x;=np.pi*(x),{{x}})",        #5
    f"Rz(x;=np.pi*(x),{{x}})",        #6
    f"Rx(p)",        #7
    f"Ry(p)",        #8
    f"Rz(p)",        #9
    f"crx(p)",        #10
    f"cry(p)",        #11
    f"crz(p)",        #12
    'cx',           #13
    f'Rx(p,x;=y*np.arctan(x),{{y,x}})', #14
    'X',            #15
    'Y',            #16
    'Z',            #17
    ]

num_actions = len(actions)

def model_complexity(K,y,reg_lambda=0.0):
    # just as a quick check 
    try:
        sqrt_K = sqrtm(K)
    except Exception as e:
        print("Error: Failed to find square root of matrix k:", e)
        return 1e6
    K_shift_inv = np.linalg.inv(K+np.eye(K.shape[0])*reg_lambda)
    K_shift_inv_squared = np.matmul(K_shift_inv,K_shift_inv)
    KK = np.matmul(sqrt_K,np.matmul(K_shift_inv_squared,sqrt_K))
    return np.real(np.dot(y,np.dot(KK,y)))

def gen_model_from_str(X,Y,fm:str,num_qubits,num_features,maxiter=0,file="adam_log.log",fit=False):
    lfm = LayeredFeatureMap.from_string(fm,num_qubits,num_features)
    np.random.seed(0)
    param_ini = np.random.uniform(-1,1,lfm.num_parameters)

    kernel_matrix = ProjectedQuantumKernel(lfm,Executor("statevector_simulator"),initial_parameters=param_ini,gamma=gamma)
    print("kernel_matrix.num_parameters",kernel_matrix.num_parameters)
    if maxiter > 0 and kernel_matrix.num_parameters > 0:

        def func(param):
            kernel_matrix.assign_parameters(param)
            K = kernel_matrix.evaluate(X)
            return model_complexity(K,Y,1e-3)

        param = Adam({"maxiter":maxiter,"lr":0.1,"log_file":file,"tol":1e-1}).minimize(func,kernel_matrix.parameters,StochasticPerturbationGradient(func)).x

        kernel_matrix.assign_parameters(param)

    krr = KernelRidge(alpha=1e-10,kernel=kernel_matrix.evaluate_pairwise)
    if fit:
        krr.fit(X, Y)

    return krr


# Environment for reinforcement training
import gym
from gym import spaces
from gym.utils import seeding
from scipy.linalg import eigh
import scipy
from scipy.linalg import sqrtm
from qiskit import Aer

# Environment for reinforcement training

class PlyEnv(gym.Env):
    def __init__(self):
        """
        Initialization function of the environment
        """

        super(PlyEnv, self).__init__()
        # Define action and observation space (integer arrays)

        # Defines the possible actions
        self.action_space = spaces.Discrete(len(actions))

        # Observations feature map converted to integer string
        self.observation_space = spaces.Box(low=0, high=len(actions)-1,
                                            shape=(max_num_gates,),dtype=int)

        # String for storing the feature map
        self.fm_str = ""

        self.steps_done = 0 # Counter for performed actions
        self.last_MSE = -1000  # Variable for storing the last fit error
        self.best_MSE = -1000  # Variable for storing the best fit error
        self.best_fm_str = "" # Variable for string the best feature map as string
        self.best_case_counter=0

        # Dictionary for string to integer labels
        self.action_dic={}
        i=0
        for a in actions:
            if a=="remove_first":
                continue
            if a=="remove_last":
                continue
            self.action_dic[a]=i
            i+=1
        print('muzero autoqfm akr environment')
    def reset(self):
        """
        Reset function of the environment
        Resets the feature map to inital state
        """

        self.fm_str = ""
        self.done = False
        self.steps_done = 0
        self.last_MSE=100

        self.observations = self.text_to_int(self.fm_str)

        return self.observations

    def step(self,action):
        """
        Function that performce an action and returns the reward and the resulting
        feature-map as observation space
        """

        self.steps_done += 1

        reward = 0.0

        if action == 0:
            # Remove first action
            self.fm_str = self.fm_str[self.fm_str.find('-')+1:]
        elif action == 1:
            # Remove last action
            i = self.fm_str.rfind('-',0,-1)
            self.fm_str = self.fm_str[:i] + '-'
        else:
            # Add gates to the feature map
            self.fm_str = self.fm_str + actions[action] + "-"


        if "(x)" not in self.fm_str:
            # Capture the case, that there is no X in the circuit
            reward = -100 # Strong penalty since this is not a suitable feature map
            # MSE to default
            self.last_MSE=1000
            fit_loss=1000
        else:
            # Calculate the MSE fit error of the current feature map
            krr = self.get_krr(X_train, Y_train)
            fit_loss = np.min(cross_val_score(krr, X_train, Y_train, cv=4))

            # Reward function
            if fit_loss >= self.best_MSE+1e-4:
                # New bestcase: strong reward
                print("new total best case: ",self.fm_str,"fit_loss",fit_loss)
                self.best_fm_str = self.fm_str
                self.best_case_counter = self.best_case_counter + 1
                reward = 100.0
            elif fit_loss >= self.best_MSE-1e-4:
                # Reached best case: positive reward
                print("reached best case: ",self.fm_str,"fit_loss",fit_loss)
                self.best_fm_str = self.fm_str
                reward = 25.0
            else:
                if action == 2:
                    reward = -2
                elif action == 0 or action == 1:
                    reward = 2
                else:
                    reward = -5.0

            # Update last and best MSE
            self.last_MSE = fit_loss
            self.best_MSE = max(self.best_MSE,fit_loss)


        # Create observation integer array
        self.observations = self.text_to_int(self.fm_str)

        # If too many steps are done, finish this environment
        if self.steps_done >= max_steps:
            self.done = True

        # If the feature map is to long, finish this environment
        if np.count_nonzero(self.observations) >= max_num_gates:
            self.done = True

        print("{i} Fmap: {fmap} Reward: {reward} CV Loss: {fit_loss} \n".format(i=self.steps_done,fmap=self.fm_str,reward=reward,fit_loss=fit_loss),end='')

        info = {}
        # Return featuremap as observation, current reward for action, done (=true if environment is done)
        return self.observations,reward,self.done,info

    def get_krr(self,X,Y):
        """
        Create Kernel Ridge class from feature map
        """
        return gen_model_from_str(X_train,Y_train,self.fm_str,num_qubits,X_train.shape[1],maxiter=0,file="random.log")


    def text_to_int(self,text:str):
        """
        Helper function for generating the observation integer array from the feature_map string
        """
        text_array = text.split('-')
        text_array.pop()
        observation = np.zeros(max_num_gates,dtype=int)

        i = 0
        for t in text_array:
            if t != "":
                observation[i] = self.action_dic[t]
                i=i+1
        return observation
    
    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(len(actions)))  # Return all actions as legal

import datetime
import pathlib
import torch

from games.abstract_game import AbstractGame

logdir = 'dkr_autoqfm_muzero_run/results'
# if not os.path.exists(logdir):
#     os.makedirs(logdir)


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 42  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (1, 1, max_num_gates)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        #self.observation_shape = (1, 1, 1)
        self.action_space = list(range(len(actions)))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 10  # Maximum number of moves if game is not finished before
        self.num_simulations = 20  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        # self.pb_c_base = 19652
        # self.pb_c_init = 1.25
        self.pb_c_base = 2   
        self.pb_c_init = 1.25


        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        # think about support size value
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 2  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 2  # Number of channels in policy head
        self.resnet_fc_reward_layers = [16]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [16]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [16]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 8
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network


        # if not os.path.exists(logdir):
        #     os.makedirs(logdir)
        ### Training
        self.results_path = pathlib.Path(logdir).resolve().parents[1] / "dkr_autoqfm/results" / pathlib.Path(logdir).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 100000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 16  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.01#0.02  # Initial learning rate
        self.lr_decay_rate = 1#0.8  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000



        ### Replay Buffer           
        #                       -> FAP changed from 500 to 10
        self.replay_buffer_size = 100  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        #           FAP changed to False
        self.PER = False  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 0.35


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = PlyEnv()
        # if seed is not None:
        #     self.env.seed(seed)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done, _ = self.env.step(action)
        #return np.array([[observation]]), reward, done
        #print('observation:', observation)
        return np.array([[observation]]), reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """

        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return np.array([[self.env.reset()]])

    def close(self):
        """
        Properly close the game.
        """
        pass

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return 0

    def render(self):
        """
        Display the game observation.
        """
        pass