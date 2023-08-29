import numpy as np
# takes packages files from the specified folder
import os
# import sys
# sys.path.insert(0, os.path.join(os.getcwd(), '..','..'))

from games.generate_quantum_data import create_quantum_fmnist

num_components = 4
category_filter_list = [2,0]
num_training, num_test = 100, 100
X_train,Y_train,X_test,Y_test = create_quantum_fmnist(
    num_training, num_test, num_components=num_components,category_filter_list=category_filter_list, num_su2_layers=0, su2_random_seed=42)

# lets try a QML approach based on a random feature map 
from squlearn.feature_map.layered_feature_map import LayeredFeatureMap
from squlearn.kernel.matrix import FidelityKernel
from squlearn.util import Executor

num_qubits=4
max_num_gates = 10
max_steps = 25


actions = [         # action integer number
    "",             #0
    "H",            #1
    f"Rx(x;=(x),{{x}})",        #2
    f"Ry(x;=(x),{{x}})",        #3
    f"Rz(x;=(x),{{x}})",        #4
    f"Rx(p,x;=y*(x),{{y,x}})",        #5
    f"Ry(p,x;=y*(x),{{y,x}})",        #6
    f"Rz(p,x;=y*(x),{{y,x}})",        #7
    #"cz", #8
    #"cy",  #9
    'cx',           #10    
    f'Rx(p,x;=y*np.cos(x),{{y,x}})', #11 # changed arccos to cos due to no rescaling
    'X',            #12
    'Y',            #13
    'Z',            #14
    'S',            #15
    'T'             #16
   ]
#alternative action set 
# actions = [         # action integer number
#     "",             #0
#     "H",            #1
#     f"Rx(x;=(x),{{x}})",        #2
#     f"Ry(x;=(x),{{x}})",        #3
#     f"Rz(x;=(x),{{x}})",        #4
#     f"Rx(x;=np.pi/2*(x),{{x}})",        #5
#     f"Ry(x;=np.pi/2*(x),{{x}})",        #6
#     f"Rz(x;=np.pi/2*(x),{{x}})",        #7
#     f"Rx(x;=np.pi*(x),{{x}})",        #5
#     f"Ry(x;=np.pi*(x),{{x}})",        #6
#     f"Rz(x;=np.pi*(x),{{x}})",        #7
#     f"Rx(x;=1.5*np.pi*(x),{{x}})",        #5
#     f"Ry(x;=1.5*np.pi*(x),{{x}})",        #6
#     f"Rz(x;=1.5*np.pi*(x),{{x}})",        #7
#     f"Rx(x;=2*np.pi*(x),{{x}})",        #5
#     f"Ry(x;=2*np.pi*(x),{{x}})",        #6
#     f"Rz(x;=2*np.pi*(x),{{x}})",        #7
#     #"cz", #8
#     #"cy",  #9
#     'cx',           #10    
#     f'Rx(x;=np.pi/2*np.cos(x),{{x}})', #11 # changed arccos to cos due to no rescaling
#     f'Rx(x;=np.pi*np.cos(x),{{x}})', #11 # changed arccos to cos due to no rescaling
#     f'Rx(x;=2*np.pi*np.cos(x),{{x}})', #11 # changed arccos to cos due to no rescaling
#     'X',            #12
#     'Y',            #13
#     'Z',            #14
#     'S',            #15
#     'T'             #16
#     ]
num_actions = len(actions)


# Environment for reinforcement training
import gym
from gym import spaces
from gym.utils import seeding
from scipy.linalg import eigh
import scipy
from scipy.linalg import sqrtm
from qiskit import Aer

class KernelPropertiyEnv(gym.Env):
    def __init__(self, training_data, training_labels, num_qubits, num_features):
        """
        Initialization function of the environment
        """

        super(KernelPropertiyEnv, self).__init__()
        # Define action and observation space (integer arrays)

        # Defines the possible actions
        self.action_space = spaces.Discrete(len(actions))

        # Observations feature map converted to integer string
        self.observation_space = spaces.Box(low=0, high=len(actions)-1,
                                            shape=(max_num_gates,),dtype=int)
        # spaces.Dict({
        #     'fm':spaces.Box(low=0, high=len(actions)-1,
        #                                     shape=(max_num_gates,),dtype=int),

            # the other part of the observation space are the properties of our quantum kernel 
            # kernel target alignment
            # kernel variance -> indicator for exponential concentration
            # 'var_kernel': spaces.Box(low=0, high=1000,
            #                                 shape=(1,),dtype=float),

            # # prediction error bound
            # 'pred_error': spaces.Box(low=0, high=1e10,
            #                                 shape=(1,),dtype=float),

            # 'best_overall_pred_error': spaces.Box(low=0, high=1e10, shape=(1,), dtype=float),
        #})

        self.best_overall_pred_error = 1e10
        self.best_overall_fm = ""
        self.training_data  = training_data
        self.training_labels = training_labels
        #self.seed()
        self.seed = 42
        # Dictionary for string to integer labels
        self.action_dic={}
        i=0
        for a in actions:
            self.action_dic[a]=i
            i+=1

        # try to calculate the best possible classical kernel matrix in the init -> then it just has to be
        # evaluated once as long as the training data stays the same
        self._num_qubits = num_qubits
        self._num_features = num_features

        print('Lets goooo AutoQFM!')
        #print('legal autoqfm actions:', self.legal_actions())
    def reset(self):
        """
        Reset function of the environment
        Resets the feature map to inital state
        """

        self.fm_str = ""
        self.best_fm_str = "" 
        self.last_action = ""
        self.done = False
        self.steps_done = 0
        self.best_spec_quotient = 0.0
        self.last_pred_error = 0.0 
        self.best_pred_error = 1e6
        # self.last_KTA = 0.0
        # self.best_KTA = 0.0
        self.reward = 0.0
        self.observations = self.text_to_int(self.fm_str)
        #{'fm':self.text_to_int(self.fm_str),
                        # 'var_kernel':np.array([10.0]),  
                        # 'pred_error':np.array([1e6]), 
                        # 'best_overall_pred_error':np.array([self.best_overall_pred_error]),         
                       #}

        return self.observations

    def step(self,action):
        """
        Function that performs an action and returns the reward and the resulting
        feature-map as observation space
        """
        self.steps_done += 1
        
        pred_error = 1e6
        var_kernel = 10.0
        punish_exp_conc = 0.0
        reward_pred_error = 0.0
        punish_x = 0.0 
        punish_action = 0.0
        reward_overall_best = 0.0
        #reward_improved_kta = 0.0


        # Add gates to the feature map
        self.fm_str = self.fm_str + actions[action] + "-"

        print('step in the AQGM envi with respective feature map:', self.steps_done, self.fm_str)
        if action == self.last_action and self.steps_done > 1:
            # Punish same action in a row
            punish_action = -25.0
            print('Two same actions in a row!')

        
        if "(x)" not in self.fm_str:
            # Capture the case, that there is no X in the circuit
            punish_x = -100.0
            #self.reward = -100.0 # Strong penalty since this is not a suitable feature map
            print("No X in the circuit!",self.fm_str)
        else:
            # Calculate the quantum kernel and validate its properties
            q_kernel_matrix = self.return_kernel_matrix(x_train=self.training_data)

            #geom_diff, g_tra = self.geometric_difference(self.classical_kernel_matrix, q_kernel_matrix)

            pred_error = self.prediction_error_bound(k=q_kernel_matrix, y=self.training_labels)
            pred_error = np.real_if_close(pred_error, tol=1e-7)

            # Calculate the kernel target alignment
            #kta = self.compute_kernel_target_alignment(q_kernel_matrix, labels=self.training_labels)
            #print('prediction error bound in a NORMAL step:', pred_error, self.steps_done)
            # Calculate the kernel variance -> indicator for exponential concentration
            var_kernel = self.matrix_variance_exclude_diagonal(q_kernel_matrix)

            if var_kernel < 0.001:
                print("kernel variance is very small, exponential concentration!",self.fm_str,var_kernel)
                punish_exp_conc = -100.0
            else:
                # reward improvement of the prediction error bound
                
                if pred_error <= (self.best_pred_error-0.01) and pred_error <= (self.last_pred_error-0.01) and self.last_pred_error != 0.0:
                    print("better prediction error bound!",self.fm_str,pred_error, self.best_pred_error, self.steps_done)
                    self.best_fm_str = self.fm_str
                    self.best_pred_error = pred_error
                    # change here! more reward for improvements in prediction error than kta! 
                    reward_pred_error = 100.0

                    # Reward for surpassing overall best prediction error
                    if pred_error < self.best_overall_pred_error:
                        reward_overall_best = 200.0  # You can adjust the reward value
                        self.best_overall_pred_error = pred_error
                        self.best_overall_fm = self.fm_str
                        print('best overall prediction error and FM:', self.best_overall_pred_error, self.best_overall_fm)
                # reward if the found kta is better then the current best kta to motivate the agent to find a better solution
                # if kta >= (self.best_KTA+0.1) and kta >= (self.last_KTA+0.1) and self.last_KTA != 0.0:
                #     print("better KTA!",kta, self.best_KTA)
                #     #self.best_fm_str = self.fm_str
                #     self.best_KTA = kta
                #     reward_improved_kta = 50.0
            self.last_pred_error = pred_error
            #self.last_KTA = kta

             
        # probably the split of the reward in that way makes no sense because both of them dont happen at the same time anyway
        self.reward =  punish_exp_conc + reward_pred_error + punish_x + reward_overall_best + punish_action #+ reward_improved_kta

        self.last_action = action
        # Create observation integer array
        self.observations = self.text_to_int(self.fm_str)
        #{'fm':self.text_to_int(self.fm_str),
                            # 'var_kernel':np.array([var_kernel]),
                            # 'pred_error':np.array([pred_error]),
                            # 'best_overall_pred_error':np.array([self.best_overall_pred_error]),
                            #}

        # If too many steps are done, finish this environment
        if self.steps_done >= max_steps:
            self.done = True

        # If the feature map is to long, finish this environment
        if np.count_nonzero(self.observations) >= max_num_gates:
            self.done = True
        info = {}
        # Return featuremap as observation, current reward for action, done (=true if environment is done)
        return self.observations,self.reward,self.done,info

    def return_kernel_matrix(self, x_train):
        """
        Create Kernel Ridge class from feature map
        """
        lfm = LayeredFeatureMap.from_string(self.fm_str,num_qubits=self._num_qubits,num_features=self._num_features)
        
        if lfm.num_parameters > 0:
            np.random.seed(self.seed)
            param_list = np.random.uniform(0,2*np.pi,num_qubits)
            list_=[]
            for i in range(int(lfm.num_parameters/self._num_qubits)):
                for j in range(self._num_qubits):
                    list_.append(param_list[j])
            initial_parameters = np.array(list_)
        else:
            initial_parameters = []

        #kernel_matrix = ProjectedQuantumKernel(lfm,Executor("statevector_simulator"),initial_parameters=initial_parameters,gamma=0.25)
        quantum_kernel = FidelityKernel(lfm,Executor("statevector_simulator"),initial_parameters=initial_parameters)
    
        # Do Kernel Ridge Regression
        quantum_kernel.assign_parameters(initial_parameters)
        kernel_matrix = quantum_kernel.evaluate(x_train)
        return kernel_matrix
    
    def compute_kernel_target_alignment(self,kmatrix, labels, rescale_class_labels=True):
        # write some code that normalizes the labels    
        
        if rescale_class_labels:
            nplus = np.count_nonzero(np.array(labels) == 1)
            nminus = len(labels) - nplus
            _Y = np.array([y / nplus if y == 1 else y / nminus for y in labels])
        else:
            _Y = np.array(labels)
        
        T = np.outer(_Y, _Y)
        inner_product = np.sum(kmatrix * T)
        norm = np.sqrt(np.sum(kmatrix * kmatrix) * np.sum(T * T))
        alignment = inner_product / norm
        return alignment

    def matrix_variance_exclude_diagonal(self,matrix):
        # Exclude diagonal elements
        flattened_matrix = matrix.flatten()
        diagonal_indices = np.arange(0, len(flattened_matrix), matrix.shape[1] + 1)
        flattened_matrix_without_diagonal = np.delete(flattened_matrix, diagonal_indices)
        
        variance = np.var(flattened_matrix_without_diagonal)
        return variance
    
    
    def prediction_error_bound(
        self,
        k: np.ndarray,
        y: np.ndarray,
        lambda_reg: float = 0.001,
        ) -> float:
        """
        Function for calculating the prediction error s_K(N) bound for a given kernel matrix K and labels y

        s_k(N) = sum_ij [sqrt(K)*(K+lambda_reg*I)_ij)^-2*sqrt(K)]_ij * y_i * y_j

        See supplementary material of DOI:10.1038/s41467-021-22539-9 for details

        Args:
            k (np.ndarray): Kernel matrix
            y (np.ndarray): Labels
            lambda_reg (float): Regularization parameter lambda_reg

        """
        
        try:
            sqrt_k = sqrtm(k)
        except Exception as e:
            print("Error: Failed to find square root of matrix k:", e)
            return 1e6
        
        k_shift_inv = np.linalg.inv(k+np.eye(k.shape[0])*lambda_reg)
        k_shift_inv_squared = np.matmul(k_shift_inv,k_shift_inv)
        kk = np.matmul(sqrt_k,np.matmul(k_shift_inv_squared,sqrt_k))
        return np.real_if_close(np.dot(y,np.dot(kk,y)))

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
    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

# from stable_baselines3.common.env_checker import check_env
# env = KernelPropertiyEnv(training_data=X_train, training_labels=Y_train, num_qubits=num_qubits,
#                          num_features=4)
# check_env(env)

import datetime
import pathlib
import torch

from games.abstract_game import AbstractGame

logdir = 'autoqfm_muzero_run/results'
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
        self.results_path = pathlib.Path(logdir).resolve().parents[1] / "autoqfm/results" / pathlib.Path(logdir).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
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
        self.lr_init = 0.1#0.02  # Initial learning rate
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
        self.env = KernelPropertiyEnv(training_data=X_train, training_labels=Y_train, num_qubits=num_qubits,
                         num_features=num_components)
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