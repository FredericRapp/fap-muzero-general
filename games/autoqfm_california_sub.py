import numpy as np
import json
# takes packages files from the specified folder

#from sklearn.datasets import fetch_california_housing
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge

# Load data
# housing = fetch_california_housing()
# X = housing.data
# y = housing.target
# X = MinMaxScaler().fit_transform(X)
# Generate a random classification problem with 20 samples
X, y = make_classification(n_samples=100, n_features=8, n_informative=3, n_classes=2, random_state=42)

# # Define the desired number of data points
# desired_num_samples = 400  # Adjust this as needed

# # Randomly sample data points
# # set the seed to ensure reproducibility
# np.random.seed(42)
# random_indices = np.random.choice(len(X), desired_num_samples, replace=False)
# X_subset = X[random_indices]
# y_subset = y[random_indices]
# # Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# print(X_train)

# lets try a QML approach based on a random feature map 
from squlearn.feature_map.layered_feature_map import LayeredFeatureMap
from squlearn.kernel.matrix import FidelityKernel, ProjectedQuantumKernel
from squlearn.util import Executor

num_qubits=8
max_num_gates = 10
max_steps = 12


# actions = [         # action integer number
#     "",             #0
#     "H",            #1
#     f"Rx(x;=(x),{{x}})",        #2
#     f"Ry(x;=(x),{{x}})",        #3
#     f"Rz(x;=(x),{{x}})",        #4
#     f"Rx(p,x;=y*(x),{{y,x}})",        #5
#     f"Ry(p,x;=y*(x),{{y,x}})",        #6
#     f"Rz(p,x;=y*(x),{{y,x}})",        #7
#     #"cz", #8
#     #"cy",  #9
#     'cx',           #10    
#     f'Rx(p,x;=y*np.cos(x),{{y,x}})', #11 # changed arccos to cos due to no rescaling
#     'X',            #12
#     'Y',            #13
#     'Z',            #14
#     'S',            #15
#     'T'             #16
#    ]


actions = [         # action integer number
    "",             #2
    "H",            #3
    r"Rx(x;=(x),{x})",        #2
    r"Ry(x;=(x),{x})",        #3
    r"Rz(x;=(x),{x})",        #4
    r"Rx(x;=np.pi*(x),{x})",        #4
    r"Ry(x;=np.pi*(x),{x})",        #5
    r"Rz(x;=np.pi*(x),{x})",        #6
    'cx',           #13
    r'Rx(x;=np.arctan(x),{x})', #14
    'X',            #15
    'Y',            #16
    'Z',            #17
    r"Rx(p;=0*p+5/3*np.pi,{p})",        #18
    r"Ry(p;=0*p+5/3*np.pi,{p})",       #19
    r"Rz(p;=0*p+5/3*np.pi,{p})",       #20
    r"crx(p;=0*p+5/3*np.pi,{p})",      #21
    r"cry(p;=0*p+5/3*np.pi,{p})",      #22
    r"crz(p;=0*p+5/3*np.pi,{p})",      #23
    r"Rx(p;=0*p+4/3*np.pi,{p})",        #24
    r"Ry(p;=0*p+4/3*np.pi,{p})",        #25
    r"Rz(p;=0*p+4/3*np.pi,{p})",        #26
    r"crx(p;=0*p+4/3*np.pi,{p})",       #27
    r"cry(p;=0*p+4/3*np.pi,{p})",       #28
    r"crz(p;=0*p+4/3*np.pi,{p})",       #29
    r"Rx(p;=0*p+2/3*np.pi,{p})",         #30
    r"Ry(p;=0*p+2/3*np.pi,{p})",         #31
    r"Rz(p;=0*p+2/3*np.pi,{p})",         #32
    r"crx(p;=0*p+2/3*np.pi,{p})",        #33
    r"cry(p;=0*p+2/3*np.pi,{p})",        #34
    r"crz(p;=0*p+2/3*np.pi,{p})",        #35
    r"Rx(p;=0*p+1/3*np.pi,{p})",         #36
    r"Ry(p;=0*p+1/3*np.pi,{p})",         #37
    r"Rz(p;=0*p+1/3*np.pi,{p})",         #38
    r"crx(p;=0*p+1/3*np.pi,{p})",        #39
    r"cry(p;=0*p+1/3*np.pi,{p})",        #40
    r"crz(p;=0*p+1/3*np.pi,{p})",        #41
    ]
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
    def __init__(self, training_data, training_labels, num_qubits, num_features, result_dict):
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
        self.storage_dict = result_dict
        self.storage_manager = StorageManager(result_dict)
        

        # self.best_overall_pred_error = 1e10
        # self.best_overall_fm = ""
        self.training_data  = training_data
        self.training_labels = training_labels

        self.best_pred_error = np.inf

        #self.seed()
        #self.seed = 42
        self.seed = np.random.randint(0,1000)
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

        print('Lets goooo PQK AutoQFM CALI SUBSET!')
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


        self.reward = 0.0
        self.observations = self.text_to_int(self.fm_str)

        return self.observations

    def step(self,action):
        """
        Function that performs an action and returns the reward and the resulting
        feature-map as observation space
        """
        self.steps_done += 1
        
        pred_error = np.inf
        var_kernel = np.inf
        punish_exp_conc = 0.0
        reward_pred_error = 0.0
        punish_x = 0.0 
        punish_action = 0.0
        reward_overall_best = 0.0


        # Add gates to the feature map
        self.fm_str = self.fm_str + actions[action] + "-"

        print('step in the AQGM envi with respective feature map:', self.steps_done, self.fm_str)
        if self.steps_done >= 1 and action == self.last_action:
            print("same action as before!",self.fm_str)
            punish_action = -1.0
        
        if "(x)" not in self.fm_str:
            # Capture the case, that there is no X in the circuit
            punish_x = -1.0
            print("No X in the circuit!",self.fm_str)
        else:
            # Calculate the quantum kernel and validate its properties
            q_kernel_matrix = self.return_kernel_matrix(x_train=self.training_data)

            pred_error = self.prediction_error_bound(k=q_kernel_matrix, y=self.training_labels)
            pred_error = np.real_if_close(pred_error, tol=1e-7)
            # Calculate the kernel variance -> indicator for exponential concentration
            var_kernel = self.matrix_variance_exclude_diagonal(q_kernel_matrix)

            if var_kernel < 0.001:
                print("kernel variance is very small, exponential concentration!",self.fm_str,var_kernel)
                punish_exp_conc = -10.0
            else:
                # reward improvement of the prediction error bound
                
                if pred_error <= self.best_pred_error:
                    print("better prediction error bound!",pred_error, self.best_pred_error,self.fm_str, self.steps_done)
                    self.best_fm_str = self.fm_str
                    self.best_pred_error = pred_error
                    # change here! more reward for improvements in prediction error than kta! 
                    reward_pred_error = 100.0
                    self.storage_dict["best_fm_str"].append(self.best_fm_str)
                    self.storage_dict["best_pred_error"].append(pred_error)
                    # Ensure any numpy arrays are converted to lists before serializing to JSON
                    self.storage_dict["best_pred_error"] = [entry.tolist() if isinstance(entry, np.ndarray) else entry for entry in self.storage_dict["best_pred_error"]]
                    print('append happened!:', self.storage_dict["best_pred_error"])
                    self.storage_manager.update_storage_dict(self)
                else: 
                    punish_action = -1.0
            
            self.last_pred_error = pred_error

             
        # probably the split of the reward in that way makes no sense because both of them dont happen at the same time anyway
        self.reward =  punish_exp_conc + reward_pred_error + punish_x + reward_overall_best + punish_action 

        self.last_action = action
        # Create observation integer array
        self.observations = self.text_to_int(self.fm_str)

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
        np.random.seed(42)
        param_ini = np.random.uniform(-1,1,lfm.num_parameters)

        #kernel_matrix = ProjectedQuantumKernel(lfm,Executor("statevector_simulator"),initial_parameters=initial_parameters,gamma=0.25)
        quantum_kernel = ProjectedQuantumKernel(lfm,Executor("statevector_simulator"),initial_parameters=param_ini)
        
        quantum_kernel.assign_parameters(param_ini)
        
        kernel_matrix = quantum_kernel.evaluate(x_train)
        return kernel_matrix


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
        return list(range(len(actions)))  

import datetime
import pathlib
import torch

from games.abstract_game import AbstractGame

logdir = 'autoqfm_muzero_run_cali/results'


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
        # changed from 0 to 10
        self.stacked_observations = 10  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = max_steps  # Maximum number of moves if game is not finished before
        self.num_simulations = max_steps  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        # self.pb_c_base = 19652
        # self.pb_c_init = 1.25
        self.pb_c_base = 5000
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
        self.encoding_size = 10
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network


        # if not os.path.exists(logdir):
        #     os.makedirs(logdir)
        ### Training
        self.results_path = pathlib.Path(logdir).resolve().parents[1] / "make_classification/results" / pathlib.Path(logdir).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 100000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 32 #instead of 16 # Number of parts of games to train on at each training step
        # changed from 10 to 100
        self.checkpoint_interval = 100  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.01#0.02  # Initial learning rate
        self.lr_decay_rate = 1#0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 5000



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
        temp = 0.35
        return temp


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        
        self.env = KernelPropertiyEnv(training_data=X_train, training_labels=Y_train, num_qubits=num_qubits,
                         num_features=8, result_dict={"best_fm_str": list(), "best_pred_error": list()})
        #self.storage_manager = StorageManager(self.env)


    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done, _ = self.env.step(action)
        #self.storage_manager.update_storage_dict() 
        #self.update_storage_dict()
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

    def import_storage(file_name):
        import json
        with open(file_name, "r") as json_file:
            loaded_data = json.load(json_file)
        best_pred_error_list = loaded_data["best_pred_error"]
        best_circuit = loaded_data["best_fm_str"]
        # Find the index of the minimum value in the list
        min_index = min(enumerate(best_pred_error_list), key=lambda x: x[1])[0]
        # Get the minimum value and best circuit 
        min_value = best_pred_error_list[min_index]
        best_circuit = best_circuit[min_index]
        print("best prediction error:", min_value)
        print("best circuit:", best_circuit)

        return loaded_data
    
class StorageManager:
    def __init__(self, result_dict):
        #self.env = env
        #self.storage_dict = {"best_fm_str": list(), "best_pred_error": list()}
        self.storage_dict = result_dict

    def update_storage_dict(self, env):
        # Update the storage_dict using data from the environment
        self.storage_dict["best_fm_str"].extend(env.storage_dict["best_fm_str"])
        self.storage_dict["best_pred_error"].extend(env.storage_dict["best_pred_error"])

        # Ensure any numpy arrays are converted to lists before serializing to JSON
        self.storage_dict["best_pred_error"] = [entry.tolist() if isinstance(entry, np.ndarray) else entry for entry in self.storage_dict["best_pred_error"]]
        print('update happened!:', self.storage_dict["best_pred_error"])
        # Serialize the updated dictionary to a JSON file
        with open("make_calssif_storage.json", "w") as json_file:
            storage_dict = {
                "best_fm_str": self.storage_dict["best_fm_str"],
                "best_pred_error": self.storage_dict["best_pred_error"]
            }
            json.dump(storage_dict, json_file)
            json_file.write("\n")
    
