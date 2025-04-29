"""
CompTuner: Compiler Autotuning through Multiple Phase Learning

This tool uses machine learning (Random Forest) and Particle Swarm Optimization (PSO) 
to automatically find the optimal compiler flag combinations that maximize program performance.
It compares the performance of different flag combinations against the baseline -O3 optimization.
"""
import os,sys
import random, time, copy,subprocess, argparse
import math
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.stats import norm



def write_log(ss, file):
    """ Write message to the log file """
    with open(file, 'a') as log:
        log.write(ss + '\n')


def execute_terminal_command(command):
    """ Execute shell command and handle errors
    
    Args:
        command: The shell command to execute
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            if result.stdout:
                print("Command output:")
                print(result.stdout)
        else:
            if result.stderr:
                print("Error output:")
                print(result.stderr)
    except Exception as e:
        print("Error executing command:", str(e))

def get_objective_score(independent, k_iter, SOURCE_PATH, GCC_PATH, INCLUDE_PATH, EXEC_PARAM, LOG_FILE, all_flags):
    """ Obtain the speedup ratio between current flag configuration and O3
    
    Args:
        independent: Binary vector representing which flags to use
        k_iter: Iteration counter
        SOURCE_PATH: Path to program source code
        GCC_PATH: Path to gcc compiler
        INCLUDE_PATH: Path to include files
        EXEC_PARAM: Parameters for program execution
        LOG_FILE: Path to log file
        all_flags: List of all compiler flags
        
    Returns:
        Speedup ratio (time_O3/time_current)
    """
    # Construct optimization flags string
    opt = ''
    for i in range(len(independent)):
        if independent[i]:
            opt = opt + all_flags[i] + ' '
        else:
            negated_flag_name = all_flags[i].replace("-f", "-fno-", 1)
            opt = opt + negated_flag_name + ' '
    
    # Compile with the custom optimization flags
    command = f"{GCC_PATH} -O2 {opt} -c {INCLUDE_PATH} {SOURCE_PATH}/*.c"
    execute_terminal_command(command)
    command2 = f"{GCC_PATH} -o a.out -O2 {opt} -lm *.o"
    execute_terminal_command(command2)
    
    # Execute and measure time with custom optimization
    time_start = time.time()
    command3 = f"./a.out {EXEC_PARAM}"
    execute_terminal_command(command3)
    time_end = time.time()  
    cmd4 = 'rm -rf *.o *.I *.s a.out'
    execute_terminal_command(cmd4)
    time_c = time_end - time_start   # Time with custom optimization
    
    # Compile with O3 optimization
    time_o3 = time.time()
    command = f"{GCC_PATH} -O3 -c {INCLUDE_PATH} {SOURCE_PATH}/*.c"
    execute_terminal_command(command)
    command2 = f"{GCC_PATH} -o a.out -O3 -lm *.o"
    execute_terminal_command(command2)
    time_o3 = time.time()
    
    # Execute and measure time with O3 optimization
    command3 = "./a.out {EXEC_PARAM}"
    execute_terminal_command(command3)
    time_o3_end = time.time()  
    cmd4 = 'rm -rf *.o *.I *.s a.out'
    execute_terminal_command(cmd4)
    time_o3_c = time_o3_end - time_o3   # Time with O3
    
    # Log the speedup
    op_str = "iteration:{} speedup:{}".format(str(k_iter), str(time_o3_c /time_c))
    write_log(op_str, LOG_FILE)
    return (time_o3_c /time_c)

# Global variable to track time consumption
ts_tem = []  # time consumption
    
class compTuner:
    def __init__(self, dim, c1, c2, w, get_objective_score, random, source_path, gcc_path, include_path, exec_param, log_file, flags):
        """Initialize the CompTuner optimizer
        
        Args:
            dim: Number of compiler flags
            c1: PSO parameter - cognitive coefficient (personal best influence)
            c2: PSO parameter - social coefficient (global best influence)
            w: PSO parameter - inertia weight
            get_objective_score: Function to obtain true speedup
            random: Random seed
            source_path: Program's path
            gcc_path: GCC compiler path
            include_path: Header file path for program
            exec_param: Execution parameters
            log_file: File to record results
            flags: All compiler flags to consider
        """
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.dim = dim
        self.V = []          # Velocity vector for PSO
        self.pbest = []      # Best vector of each particle
        self.gbest = []      # Best vector across all particles
        self.p_fit = []      # Best performance of each particle
        self.fit = 0         # Best performance across all particles
        self.get_objective_score = get_objective_score 
        self.random = random
        self.SOURCE_PATH = source_path
        self.GCC_PATH = gcc_path
        self.INCLUDE_PATH = include_path
        self.EXEC_PARAM = exec_param
        self.LOG_FILE = log_file
        self.all_flags = flags

    def generate_random_conf(self, x):
        """Generate a binary configuration vector from an integer
        
        Args:
            x: Random integer seed
            
        Returns:
            Binary vector representing compiler flag configuration
        """
        # Convert integer to binary string and remove '0b' prefix
        comb = bin(x).replace('0b', '')
        # Pad with leading zeros to match dimension
        comb = '0' * (self.dim - len(comb)) + comb
        conf = []
        # Convert binary string to list of 0s and 1s
        for k, s in enumerate(comb):
            if s == '1':
                conf.append(1)
            else:
                conf.append(0)
        return conf

    def get_ei(self, preds, eta):
        """Calculate the Expected Improvement (EI) acquisition function
        
        Args:
            preds: Model predictions for each configuration
            eta: Current best performance (global best)
            
        Returns:
            EI values for each configuration
        """
        preds = np.array(preds).transpose(1, 0)
        m = np.mean(preds, axis=1)
        s = np.std(preds, axis=1)

        def calculate_f(eta, m, s):
            """Helper function for EI calculation"""
            z = (eta - m) / s
            return (eta - m) * norm.cdf(z) + s * norm.pdf(z)

        # Handle case where standard deviation is zero
        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f(eta, m, s)
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f(eta, m, s)
        return f
    
    def get_ei_predict(self, model, now_best, wait_for_train):
        """Calculate EI values for a set of configurations
        
        Args:
            model: RandomForest Model
            now_best: Current global best performance
            wait_for_train: Set of configurations to evaluate
            
        Returns:
            List of [configuration, EI value] pairs
        """
        preds = []
        estimators = model.estimators_
        # Get predictions from each tree in the RandomForest
        for e in estimators:
            preds.append(e.predict(np.array(wait_for_train)))
        acq_val_incumbent = self.get_ei(preds, now_best)
        return [[i, a] for a, i in zip(acq_val_incumbent, wait_for_train)]

    def runtime_predict(self, model, wait_for_train):
        """Predict the performance of a set of configurations
        
        Args:
            model: RandomForest Model
            wait_for_train: Set of configurations to evaluate
            
        Returns:
            List of [configuration, predicted_performance] pairs
        """
        estimators = model.estimators_
        sum_of_predictions = np.zeros(len(wait_for_train))
        # Average predictions from all trees
        for tree in estimators:
            predictions = tree.predict(wait_for_train)
            sum_of_predictions += predictions
        a = []
        average_prediction = sum_of_predictions / len(estimators)
        for i in range(len(wait_for_train)):
            x = [wait_for_train[i], average_prediction[i]]
            a.append(x)
        return a
    
    def getPrecision(self, model, seq):
        """Calculate the precision of model prediction compared to actual performance
        
        Args:
            model: RandomForest Model
            seq: Configuration sequence to evaluate
            
        Returns:
            Relative error and true performance
        """
        # Get actual performance
        true_running = self.get_objective_score(seq, k_iter=100086, SOURCE_PATH=self.SOURCE_PATH, GCC_PATH=self.GCC_PATH, INCLUDE_PATH=self.INCLUDE_PATH, EXEC_PARAM=self.EXEC_PARAM, LOG_FILE=self.LOG_FILE, all_flags = self.all_flags)
        estimators = model.estimators_
        res = []
        # Get predictions from each tree
        for e in estimators:
            tmp = e.predict(np.array(seq).reshape(1, -1))
            res.append(tmp)
        acc_predict = np.mean(res)
        # Return relative error and true performance
        return abs(true_running - acc_predict) / true_running, true_running
    
    def selectByDistribution(self, merged_predicted_objectives):
        """Select a configuration based on performance distribution
        
        Args:
            merged_predicted_objectives: List of [config, performance] pairs
            
        Returns:
            Index of selected configuration
        """
        # Calculate differences from the best performance
        diffs = [abs(perf - merged_predicted_objectives[0][1]) for seq, perf in merged_predicted_objectives]
        diffs_sum = sum(diffs)
        # Convert to probability distribution
        probabilities = [diff / diffs_sum for diff in diffs]
        index = list(range(len(diffs)))
        # Sample from distribution
        idx = np.random.choice(index, p=probabilities)
        return idx
    
    def build_RF_by_CompTuner(self):
        """Build the initial Random Forest model using exploration phase
        
        Returns:
            model: Trained RandomForest model
            initial_indep: List of explored configurations
            initial_dep: List of performance values for explored configurations
        """
        inital_indep = []
        # Randomly sample initial training instances
        time_begin = time.time()
        while len(inital_indep) < 2:
            x = random.randint(0, 2 ** self.dim - 1)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in inital_indep:
                inital_indep.append(initial_training_instance)
                
        # Evaluate initial configurations
        initial_dep = [self.get_objective_score(indep, k_iter=0, SOURCE_PATH=self.SOURCE_PATH, GCC_PATH=self.GCC_PATH, INCLUDE_PATH=self.INCLUDE_PATH, EXEC_PARAM=self.EXEC_PARAM, LOG_FILE=self.LOG_FILE, all_flags = self.all_flags) for indep in inital_indep]
        
        ts_tem.append(time.time() - time_begin)
        ss = '{}: best_seq {}, best_per {}'.format(str(round(ts_tem[-1])), str(max(initial_dep)), str(inital_indep[initial_dep.index(max(initial_dep))]))
        write_log(ss, self.LOG_FILE)
        
        all_acc = []
        # Initialize and train the Random Forest model
        model = RandomForestRegressor(random_state=self.random)
        model.fit(np.array(inital_indep), np.array(initial_dep))
        
        rec_size = 2
        # Active learning phase - expand training set until sufficient accuracy or size
        while rec_size < 50:
            global_best = max(initial_dep)
            estimators = model.estimators_
            
            # Generate random neighbors to evaluate
            neighbors = []
            while len(neighbors) < 30000:
                x = random.randint(0, 2 ** self.dim - 1)
                x = self.generate_random_conf(x)
                if x not in neighbors:
                    neighbors.append(x)
                    
            # Calculate Expected Improvement for all neighbors
            pred = []
            for e in estimators:
                pred.append(e.predict(np.array(neighbors)))
            acq_val_incumbent = self.get_ei(pred, global_best)
            ei_for_current = [[i, a] for a, i in zip(acq_val_incumbent, neighbors)]
            merged_predicted_objectives = sorted(ei_for_current, key=lambda x: x[1], reverse=True)
            
            # Select the best configuration and evaluate it
            acc = 0
            flag = False
            for x in merged_predicted_objectives:
                if flag:
                    break
                if x[0] not in inital_indep:
                    inital_indep.append(x[0])
                    acc, lable = self.getPrecision(model, x[0])
                    initial_dep.append(lable)
                    all_acc.append(acc)
                    flag = True
            rec_size += 1
            
            # If model accuracy is low, add another sample with probabilistic selection
            if acc > 0.05:
                indx = self.selectByDistribution(merged_predicted_objectives)
                while merged_predicted_objectives[indx][0] in inital_indep:
                    indx = self.selectByDistribution(merged_predicted_objectives)
                inital_indep.append(merged_predicted_objectives[indx][0])
                acc, label = self.getPrecision(model, merged_predicted_objectives[int(indx)][0])
                initial_dep.append(label)
                all_acc.append(acc)
                rec_size += 1
                
            # Update time tracking and logging
            ts_tem.append(time.time() - time_begin)
            ss = '{}: best_seq {}, best_per {}'.format(str(round(ts_tem[-1])), str(max(initial_dep)), str(inital_indep[initial_dep.index(max(initial_dep))]))
            write_log(ss, self.LOG_FILE)
            
            # Retrain the model with new data
            model = RandomForestRegressor(random_state=self.random)
            model.fit(np.array(inital_indep), np.array(initial_dep))
            
            # Exit if model accuracy is good enough
            if rec_size < 50 and np.mean(all_acc) < 0.04:
                break
                
        return model, inital_indep, initial_dep
    
    def getDistance(self, seq1, seq2):
        """Calculate cosine similarity between two configuration vectors
        
        Args:
            seq1: First configuration vector
            seq2: Second configuration vector
            
        Returns:
            Cosine similarity (measure of vector similarity)
        """
        t1 = np.array(seq1)
        t2 = np.array(seq2)
        s1_norm = np.linalg.norm(t1)
        s2_norm = np.linalg.norm(t2)
        cos = np.dot(t1, t2) / (s1_norm * s2_norm)
        return cos
    
    def init_v(self, n, d, V_max, V_min):
        """Initialize velocity vectors for PSO
        
        Args:
            n: Number of particles
            d: Dimension of each particle (number of flags)
            V_max: Maximum velocity
            V_min: Minimum velocity
            
        Returns:
            Initialized velocity vectors
        """
        v = []
        for i in range(n):
            vi = []
            for j in range(d):
                a = random.random() * (V_max - V_min) + V_min
                vi.append(a)
            v.append(vi)
        return v
    
    def update_v(self, v, x, m, n, pbest, g, w, c1, c2, vmax, vmin):
        """Update velocity vectors for PSO algorithm
        
        Args:
            v: Current velocity vectors
            x: Current position vectors
            m: Number of particles
            n: Dimension of each particle
            pbest: Personal best positions
            g: Global best position
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
            vmax: Maximum velocity
            vmin: Minimum velocity
            
        Returns:
            Updated velocity vectors
        """
        for i in range(m):
            a = random.random()
            b = random.random()
            for j in range(n):
                # PSO velocity update equation
                v[i][j] = w * v[i][j] + c1 * a * (pbest[i][j] - x[i][j]) + c2 * b * (g[j] - x[i][j])
                # Constrain velocity within bounds
                if v[i][j] < vmin:
                    v[i][j] = vmin
                if v[i][j] > vmax:
                    v[i][j] = vmax
        return v
    
    def run(self):
        """Main optimization loop
        
        Combines the ML model with PSO to find optimal compiler flags
        """
        ts = []
        # Build the initial Random Forest model
        model, inital_indep, inital_dep = self.build_RF_by_CompTuner()
        begin = time.time()
        
        # Initialize PSO parameters
        self.V = self.init_v(len(inital_indep), len(inital_indep[0]), 10, -10)
        self.fit = 0
        self.pbest = list(inital_indep)
        self.p_fit = list(inital_dep)
        
        # Set initial global best
        for i in range(len(inital_dep)):
            tmp = inital_dep[i]
            if tmp > self.fit:
                self.fit = tmp
                self.gbest = inital_indep[i]
                
        end = time.time() + ts_tem[-1]
        ts.append(end - begin)
        ss = '{}: best {}, cur-best-seq {}'.format(str(round(end - begin)), str(self.fit), str(self.gbest))
        write_log(ss, self.LOG_FILE)
        
        t = 0
        # Main optimization loop - run until time limit
        while ts[-1] < 5000:
            if t == 0:
                # First iteration - update all particles
                self.V = self.update_v(self.V, inital_indep, len(inital_indep), len(inital_indep[0]), self.pbest, self.gbest, self.w, self.c1, self.c2, 10, -10)
                for i in range(len(inital_indep)):
                    for j in range(len(inital_indep[0])):
                        a = random.random()
                        # Sigmoid function to convert continuous velocity to binary position
                        if 1.0 / (1 + math.exp(-self.V[i][j])) > a:
                            inital_indep[i][j] = 1
                        else:
                            inital_indep[i][j] = 0
                t = t + 1
            else:
                # Use model to predict performance of current particles
                merged_predicted_objectives = self.runtime_predict(model, inital_indep)
                
                # Update personal best for each particle
                for i in range(len(merged_predicted_objectives)):
                    if merged_predicted_objectives[i][1] > self.p_fit[i]:
                        self.p_fit[i] = merged_predicted_objectives[i][1]
                        self.pbest[i] = merged_predicted_objectives[i][0]
                        
                # Sort particles by predicted performance
                sort_merged_predicted_objectives = sorted(merged_predicted_objectives, key=lambda x: x[1], reverse=True)
                current_best_seq = sort_merged_predicted_objectives[0][0]
                
                # Evaluate the best predicted configuration
                temp = self.get_objective_score(current_best_seq, 1000086, SOURCE_PATH=self.SOURCE_PATH, GCC_PATH=self.GCC_PATH, INCLUDE_PATH=self.INCLUDE_PATH, EXEC_PARAM=self.EXEC_PARAM, LOG_FILE=self.LOG_FILE, all_flags=self.all_flags)
                
                if temp > self.fit:
                    # If we found a new best, update global best and move all particles
                    self.gbest = current_best_seq
                    self.fit = temp
                    self.V = self.update_v(self.V, inital_indep, len(inital_indep), len(inital_indep[0]), self.pbest,
                                           self.gbest, self.w, self.c1, self.c2, 10, -10)
                    for i in range(len(inital_indep)):
                        for j in range(len(inital_indep[0])):
                            a = random.random()
                            if 1.0 / (1 + math.exp(-self.V[i][j])) > a:
                                inital_indep[i][j] = 1
                            else:
                                inital_indep[i][j] = 0
                else:
                    """
                    Different update strategy for exploration
                    - Divide particles into two groups based on distance from current best
                    - Update them differently to balance exploration and exploitation
                    """
                    # Calculate average distance to current best
                    avg_dis = 0.0
                    for i in range(1, len(merged_predicted_objectives)):
                        avg_dis = avg_dis + self.getDistance(merged_predicted_objectives[i][0], current_best_seq)
                    
                    avg_dis = avg_dis / (len(inital_indep) - 1)
                    
                    # Divide particles into two groups: better (closer) and worse (further)
                    better_seed_indep = []
                    worse_seed_indep = []
                    better_seed_seq = []
                    worse_seed_seq = []
                    better_seed_pbest = []
                    worse_seed_pbest = []
                    better_seed_V = []
                    worse_seed_V = []
        
                    for i in range(0, len(merged_predicted_objectives)):
                        if self.getDistance(merged_predicted_objectives[i][0], current_best_seq) > avg_dis:
                            worse_seed_indep.append(i)
                            worse_seed_seq.append(merged_predicted_objectives[i][0])
                            worse_seed_pbest.append(self.pbest[i])
                            worse_seed_V.append(self.V[i])
                        else:
                            better_seed_indep.append(i)
                            better_seed_seq.append(merged_predicted_objectives[i][0])
                            better_seed_pbest.append(self.pbest[i])
                            better_seed_V.append(self.V[i])
                            
                    """
                    Update better particles - focus more on personal best (exploitation)
                    """
                    V_for_better = self.update_v(better_seed_V, better_seed_seq, len(better_seed_seq),
                                                 len(better_seed_seq[0]), better_seed_pbest, self.gbest
                                                 , self.w, 2 * self.c1, self.c2, 10, -10)
                    for i in range(len(better_seed_seq)):
                        for j in range(len(better_seed_seq[0])):
                            a = random.random()
                            if 1.0 / (1 + math.exp(-V_for_better[i][j])) > a:
                                better_seed_seq[i][j] = 1
                            else:
                                better_seed_seq[i][j] = 0
                                
                    """
                    Update worse particles - focus more on global best (exploration)
                    """
                    V_for_worse = self.update_v(worse_seed_V, worse_seed_seq, len(worse_seed_seq),
                                                len(worse_seed_seq[0]), worse_seed_pbest, self.gbest
                                                , self.w, self.c1, 2 * self.c2, 10, -10)
                    for i in range(len(worse_seed_seq)):
                        for j in range(len(worse_seed_seq[0])):
                            a = random.random()
                            if 1.0 / (1 + math.exp(-V_for_worse[i][j])) > a:
                                worse_seed_seq[i][j] = 1
                            else:
                                worse_seed_seq[i][j] = 0
                                
                    # Combine updated particles back into the population
                    for i in range(len(better_seed_seq)):
                        inital_indep[better_seed_indep[i]] = better_seed_seq[i]
                    for i in range(len(worse_seed_seq)):
                        inital_indep[worse_seed_indep[i]] = worse_seed_seq[i]
                t = t + 1

            # Update time tracking and log progress
            ts.append(time.time() - begin + ts_tem[-1])
            ss = '{}: cur-best {}, cur-best-seq {}'.format(str(round(ts[-1])), str(self.fit), str(self.gbest))
            write_log(ss, self.LOG_FILE)
            
            # Exit if time limit reached
            if (time.time() + ts_tem[-1] - begin) > 6000:
                break


def read_flags_from_file(file_path):
    """Read compiler flags from a file
    
    Args:
        file_path: Path to file containing compiler flags
        
    Returns:
        List of compiler flags
    """
    with open(file_path, 'r') as file:
        flags = file.read().strip()
    return [flag.strip() for flag in flags.split(',') if flag.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compiler Autotuning through Multiple Phase Learning")
    
    parser.add_argument("--log_file", type=str, required=True,
                        help="File to save log")
    
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to the source program for tuning")
    
    parser.add_argument("--gcc_path", type=str, required=True,
                        help="Path of compiler")
    
    parser.add_argument("--exec_param", type=str, default=None,
                        help="Execution parameter for the output executable (can be empty)")
    
    parser.add_argument("--flag_path", type=str, required=True,
                        help="Tuning flags file")
    
    
    args = parser.parse_args()
    if args.exec_param:
        EXEC_PARAM = args.exec_param
    else:
        EXEC_PARAM = '' 
    
    # Read compiler flags from file
    if args.flag_path:
        all_flags = read_flags_from_file(args.flag_path)
    else:
        all_flags = ['-O2']
        print('No flags')

    # Configure parameters for CompTuner
    com_params = {}
    com_params['dim'] = len(all_flags)
    com_params['get_objective_score'] = get_objective_score
    com_params['c1'] = 2              # PSO cognitive parameter
    com_params['c2'] = 2              # PSO social parameter
    com_params['w'] = 0.6             # PSO inertia weight
    com_params['random'] = 456        # Random seed
    com_params['source_path'] = args.source_path
    com_params['gcc_path'] = args.gcc_path
    com_params['include_path'] = '-I /home/user/polybench-code/utilities /home/user/polybench-code/utilities/polybench.c'
    com_params['exec_param'] = args.exec_param
    LOG_DIR = 'log' + os.sep
    LOG_FILE = LOG_DIR +  args.log_file
    com_params['log_file'] = LOG_FILE
    com_params['flags'] = all_flags
    
    # Create and run the CompTuner
    com = compTuner(**com_params)
    com.run()
