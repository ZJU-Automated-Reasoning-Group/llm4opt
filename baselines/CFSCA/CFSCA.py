import os, sys, random, time, copy, subprocess, itertools, math, argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm

# Global variable to track execution time at different steps
time_tem = []

def write_log(ss, file):
    """
    Write log messages to the specified file
    
    Args:
        ss: String to write to log
        file: Path to log file
    """
    with open(file, 'a') as log:
        log.write(ss + '\n')
    
def execute_terminal_command(command):
    """
    Execute a shell command and handle its output
    
    Args:
        command: Shell command to execute
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            if result.stdout:
                print("命令输出：")  # Command output
                print(result.stdout)
        else:
            if result.stderr:
                print("错误输出：")  # Error output
                print(result.stderr)
    except Exception as e:
        print("执行命令时出现错误：", str(e))  # Error occurred while executing command

def get_objective_score(independent, k_iter, SOURCE_PATH, GCC_PATH, INCLUDE_PATH, EXEC_PARAM, LOG_FILE, all_flags):
    """
    Compile and run program with specified flags, then calculate speedup relative to -O3
    
    Args:
        independent: Binary array representing which flags to enable
        k_iter: Iteration number for logging
        SOURCE_PATH: Path to source code
        GCC_PATH: Path to GCC compiler
        INCLUDE_PATH: Include path for compilation
        EXEC_PARAM: Parameters for program execution
        LOG_FILE: Path to log file
        all_flags: List of compiler flags to use
        
    Returns:
        Speedup ratio compared to -O3 (time_o3_c / time_c)
    """
    # Construct optimization flags string
    opt = ''
    for i in range(len(independent)):
        if independent[i]:
            opt = opt + all_flags[i] + ' '  # Enable flag
        else:
            negated_flag_name = all_flags[i].replace("-f", "-fno-", 1)  # Convert flag to negative form
            opt = opt + negated_flag_name + ' '  # Disable flag
    
    # Compile and run with custom optimization flags
    command = f"{GCC_PATH} -O2 {opt} -c {INCLUDE_PATH} {SOURCE_PATH}/*.c"
    execute_terminal_command(command)
    command2 = f"{GCC_PATH} -o a.out -O2 {opt} -lm *.o"
    execute_terminal_command(command2)
    time_start = time.time()
    command3 = f"./a.out {EXEC_PARAM}"
    execute_terminal_command(command3)
    time_end = time.time()  
    cmd4 = 'rm -rf *.o *.I *.s a.out'
    execute_terminal_command(cmd4)
    time_c = time_end - time_start   # Execution time with custom optimization

    # Compile and run with standard -O3 optimization for comparison
    time_o3 = time.time()
    command = f"{GCC_PATH} -O3 -c {INCLUDE_PATH} {SOURCE_PATH}/*.c"
    execute_terminal_command(command)
    command2 = f"{GCC_PATH} -o a.out -O3 -lm *.o"
    execute_terminal_command(command2)
    time_o3 = time.time()
    command3 = "./a.out {EXEC_PARAM}"
    execute_terminal_command(command3)
    time_o3_end = time.time()  
    cmd4 = 'rm -rf *.o *.I *.s a.out'
    execute_terminal_command(cmd4)
    time_o3_c = time_o3_end - time_o3   # Execution time with -O3

    # Log the speedup and return it
    op_str = "iteration:{} speedup:{}".format(str(k_iter), str(time_o3_c /time_c))
    write_log(op_str, LOG_FILE)
    return (time_o3_c /time_c)  # Return speedup ratio

class CFSCA:
    """
    Critical Flag Selection and Combination Algorithm for compiler flag optimization
    Uses machine learning to identify critical compiler flags and their combinations
    that provide better performance than standard optimization levels
    """
    def __init__(self, dim, get_objective_score, seed, related_flags, source_path, gcc_path, include_path, exec_param, log_file, flags):
        """
        Initialize CFSCA algorithm
        
        Args:
            dim: Number of compiler flags to consider
            get_objective_score: Function to evaluate speedup
            seed: Random seed for reproducibility
            related_flags: Known related flags for the target program
            source_path: Path to program source code
            gcc_path: Path to GCC compiler
            include_path: Include path for compilation
            exec_param: Execution parameters for the program
            log_file: File to record results
            flags: List of all compiler flags
        """
        self.dim = dim
        self.get_objective_score = get_objective_score
        self.seed = seed
        self.related = related_flags
        self.critical = []  # Will store identified critical flags
        self.global_best_per = 0.0  # Best performance found
        self.global_best_seq = []  # Best flag combination found
        self.random = random
        self.SOURCE_PATH = source_path
        self.GCC_PATH = gcc_path
        self.INCLUDE_PATH = include_path
        self.EXEC_PARAM = exec_param
        self.LOG_FILE = log_file
        self.all_flags = flags
        
    def generate_random_conf(self, x):
        """
        Generate a binary configuration sequence from a random number
        
        Args:
            x: Random integer value
            
        Returns:
            Binary array representing flag configuration (1=enable, 0=disable)
        """
        comb = bin(x).replace('0b', '')  # Convert to binary string
        comb = '0' * (self.dim - len(comb)) + comb  # Pad with zeros to match dimension
        conf = []
        for k, s in enumerate(comb):
            if s == '1':
                conf.append(1)
            else:
                conf.append(0)
        return conf

    def get_ei(self, preds, eta):
        """
        Calculate Expected Improvement (EI) acquisition function
        Used for Bayesian optimization to guide exploration vs. exploitation
        
        Args:
            preds: Model predictions for candidate points
            eta: Current best observed value
            
        Returns:
            Expected improvement values for each candidate
        """
        preds = np.array(preds).transpose(1, 0)
        m = np.mean(preds, axis=1)  # Mean predictions
        s = np.std(preds, axis=1)   # Standard deviation of predictions

        def calculate_f(eta, m, s):
            """Calculate EI formula components"""
            z = (eta - m) / s
            return (eta - m) * norm.cdf(z) + s * norm.pdf(z)

        # Handle special case where standard deviation is zero
        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f(eta, m, s)
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f(eta, m, s)
        return f

    def get_ei_predict(self, model, now_best, wait_for_train):
        """
        Get Expected Improvement values for candidate configurations
        
        Args:
            model: Random Forest model
            now_best: Current best speedup
            wait_for_train: Candidate flag configurations
            
        Returns:
            List of [configuration, EI value] pairs
        """
        preds = []
        estimators = model.estimators_  # Get individual trees from Random Forest
        for e in estimators:
            preds.append(e.predict(np.array(wait_for_train)))
        acq_val_incumbent = self.get_ei(preds, now_best)
        return [[i, a] for a, i in zip(acq_val_incumbent, wait_for_train)]
    
    def runtime_predict(self, model, wait_for_train):
        """
        Predict performance of flag configurations using the model
        
        Args:
            model: Random Forest model
            wait_for_train: Candidate flag configurations
            
        Returns:
            List of [configuration, predicted speedup] pairs
        """
        estimators = model.estimators_
        sum_of_predictions = np.zeros(len(wait_for_train))
        # Average predictions from all trees in the forest
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
        """
        Evaluate model precision by comparing predicted vs actual speedup
        
        Args:
            model: Random Forest model
            seq: Flag configuration to evaluate
            
        Returns:
            Relative error, actual speedup
        """
        # Get actual speedup by compiling and running
        true_running = self.get_objective_score(seq, k_iter=100086, SOURCE_PATH=self.SOURCE_PATH, GCC_PATH=self.GCC_PATH, INCLUDE_PATH=self.INCLUDE_PATH, EXEC_PARAM=self.EXEC_PARAM, LOG_FILE=self.LOG_FILE, all_flags=self.all_flags)
        # Get model prediction
        estimators = model.estimators_
        res = []
        for e in estimators:
            tmp = e.predict(np.array(seq).reshape(1, -1))
            res.append(tmp)
        acc_predict = np.mean(res)
        # Return relative error and actual speedup
        return abs(true_running - acc_predict) / true_running, true_running
    
    def selectByDistribution(self, merged_predicted_objectives):
        """
        Select configuration probabilistically based on performance differences
        Helps maintain exploration in the search process
        
        Args:
            merged_predicted_objectives: List of [config, performance] pairs
            
        Returns:
            Index of the selected configuration
        """
        # Calculate differences from the best configuration
        diffs = [abs(perf - merged_predicted_objectives[0][1]) for seq, perf in merged_predicted_objectives]
        diffs_sum = sum(diffs)
        # Convert to probability distribution
        probabilities = [diff / diffs_sum for diff in diffs]
        index = list(range(len(diffs)))
        # Sample from distribution
        idx = np.random.choice(index, p=probabilities)
        return idx
    
    def build_RF_by_CompTuner(self):
        """
        Build initial Random Forest model with a small training set
        
        Returns:
            model: Trained Random Forest model
            inital_indep: Initial training configurations
            inital_dep: Corresponding speedups
        """
        inital_indep = []
        time_begin = time.time()
        
        # Randomly sample initial training instances
        while len(inital_indep) < 2:
            x = random.randint(0, 2 ** self.dim - 1)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in inital_indep:
                inital_indep.append(initial_training_instance)
        
        # Evaluate speedup for initial configurations
        inital_dep = [self.get_objective_score(indep, k_iter=0, SOURCE_PATH=self.SOURCE_PATH, GCC_PATH=self.GCC_PATH, INCLUDE_PATH=self.INCLUDE_PATH, EXEC_PARAM=self.EXEC_PARAM, LOG_FILE=self.LOG_FILE, all_flags=self.all_flags) for indep in inital_indep]
        
        all_acc = []  # Track model accuracy
        time_tem.append(time.time() - time_begin)
        
        # Initialize and train model
        model = RandomForestRegressor(random_state=self.seed)
        model.fit(np.array(inital_indep), np.array(inital_dep))
        rec_size = 2  # Number of configurations evaluated so far
        
        # Incrementally build training set until we have 11 samples
        while rec_size < 11:
            # Retrain model with current data
            model = RandomForestRegressor(random_state=self.seed)
            model.fit(np.array(inital_indep), np.array(inital_dep))
            global_best = max(inital_dep)
            estimators = model.estimators_
            
            if all_acc:
                all_acc = sorted(all_acc)
                
            # Generate random neighbors to evaluate
            neighbors = []
            for i in range(30000):
                x = random.randint(0, 2 ** self.dim - 1)
                x = self.generate_random_conf(x)
                if x not in neighbors:
                    neighbors.append(x)
                    
            # Get expected improvement for neighbors
            pred = []
            for e in estimators:
                pred.append(e.predict(np.array(neighbors)))
            acq_val_incumbent = self.get_ei(pred, global_best)
            ei_for_current = [[i, a] for a, i in zip(acq_val_incumbent, neighbors)]
            merged_predicted_objectives = sorted(ei_for_current, key=lambda x: x[1], reverse=True)
            
            # Select and evaluate new configuration with highest EI
            acc = 0
            flag = False
            for x in merged_predicted_objectives:
                if flag:
                    break
                if x[0] not in inital_indep:
                    inital_indep.append(x[0])
                    acc, lable = self.getPrecision(model, x[0])
                    inital_dep.append(lable)
                    all_acc.append(acc)
                    flag = True
            rec_size += 1

            # If model accuracy is low, add another configuration probabilistically
            if acc > 0.05:
                indx = self.selectByDistribution(merged_predicted_objectives)
                while merged_predicted_objectives[int(indx)][0] in inital_indep:
                    indx = self.selectByDistribution(merged_predicted_objectives)
                inital_indep.append(merged_predicted_objectives[int(indx)][0])
                acc, label = self.getPrecision(model, merged_predicted_objectives[int(indx)][0])
                inital_dep.append(label)
                all_acc.append(acc)
                rec_size += 1
                
            # Log progress
            time_tem.append(time.time() - time_begin)
            ss = '{}: best_seq {}, best_per {}'.format(str(round(time_tem[-1])), str(max(inital_dep)), str(inital_indep[inital_dep.index(max(inital_dep))]))
            write_log(ss, self.LOG_FILE)
            
        # Update global best
        self.global_best_per = max(inital_dep)
        self.global_best_seq = inital_indep[inital_dep.index(max(inital_dep))]
        return model, inital_indep, inital_dep
    
    def get_critical_flags(self, model, inital_indep, inital_dep):
        """
        Identify the most critical flags that influence performance
        
        Args:
            model: Trained Random Forest model
            inital_indep: Training configurations
            inital_dep: Corresponding speedups
            
        Returns:
            critical_flag_idx: Indices of the most important flags
            model_new: Updated model with new training data
        """
        candidate_seq = []
        candidate_per = []
        inital_indep_temp = copy.deepcopy(inital_indep)
        inital_dep_temp = copy.deepcopy(inital_dep)
        
        # Generate random candidate sequences
        while len(candidate_seq) < 30000:
            x = random.randint(0, 2 ** self.dim - 1)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in candidate_seq:
                candidate_seq.append(initial_training_instance)
                
        begin = time.time()
        
        # Predict performance for candidates
        all_per = self.runtime_predict(model, candidate_seq)
        candidate_per = [all[1] for all in all_per]
        
        # Track influence of each flag
        pos_seq = [0] * len(self.related)    
        
        # Find and evaluate best configuration
        now_best = max(candidate_per)
        now_best_seq = candidate_seq[candidate_per.index(now_best)]
        now_best = self.get_objective_score(now_best_seq, k_iter=100086, SOURCE_PATH=self.SOURCE_PATH, GCC_PATH=self.GCC_PATH, INCLUDE_PATH=self.INCLUDE_PATH, EXEC_PARAM=self.EXEC_PARAM, LOG_FILE=self.LOG_FILE, all_flags=self.all_flags)
        
        # Update training data and model
        inital_indep_temp.append(now_best_seq)
        inital_dep_temp.append(now_best)
        model_new = RandomForestRegressor(random_state=self.seed)
        model_new.fit(np.array(inital_indep_temp), np.array(inital_dep_temp))
        
        before_time = time_tem[-1]
        time_tem.append(time.time() - begin + before_time)
        
        # Update global best if needed
        if self.global_best_per < now_best:
            self.global_best_per = now_best
            self.global_best_seq = now_best_seq
            
        ss = '{}: best_seq {}, best_per {}'.format(str(round(time_tem[-1])), str(self.global_best_per), str(self.global_best_seq))
        write_log(ss, self.LOG_FILE)

        # Analyze each related flag's influence by flipping its value
        for idx in range(len(self.related)):
            new_candidate = []
            # Create new candidates by flipping current flag
            for j in range(len(candidate_seq)):
                seq = copy.deepcopy(candidate_seq[j])
                seq[self.related[idx]] = 1 - seq[self.related[idx]]  # Flip the flag
                new_candidate.append(seq)
                
            # Predict performance with flipped flag
            new_per = [all[1] for all in self.runtime_predict(model_new, new_candidate)]
            new_seq = [all[0] for all in self.runtime_predict(model_new, new_candidate)]
            
            # Find and evaluate best flipped configuration
            new_best_seq = new_seq[new_per.index(max(new_per))]
            new_best = self.get_objective_score(new_best_seq, k_iter=100086, SOURCE_PATH=self.SOURCE_PATH, GCC_PATH=self.GCC_PATH, INCLUDE_PATH=self.INCLUDE_PATH, EXEC_PARAM=self.EXEC_PARAM, LOG_FILE=self.LOG_FILE, all_flags=self.all_flags)
            
            if new_best > self.global_best_per:
                self.global_best_per = new_best
                self.global_best_seq = new_best_seq

            # Score flag influence based on performance changes when flipped
            for l in range(len(new_candidate)):
                if (candidate_per[l] > new_per[l] and new_candidate[l][self.related[idx]] == 1) or (candidate_per[l] < new_per[l] and new_candidate[l][self.related[idx]] == 0):
                    pos_seq[idx] -= 1  # Negative influence
                else:
                    pos_seq[idx] += 1  # Positive influence
                    
            # Update training data and model
            inital_indep_temp.append(new_best_seq)
            inital_dep_temp.append(new_best)
            model_new = RandomForestRegressor(random_state=self.seed)
            model_new.fit(np.array(inital_indep_temp), np.array(inital_dep_temp))
            
            time_tem.append(time.time() - begin + before_time)
            ss = '{}: best_seq {}, best_per {}'.format(str(round(time_tem[-1])), str(self.global_best_per), str(self.global_best_seq))
            write_log(ss, self.LOG_FILE)

        # Sort flags by influence and select top 10 as critical
        sort_pos = sorted(enumerate(pos_seq), key=lambda x: x[1], reverse=True)
        critical_flag_idx = []
        for i in range(10):
            critical_flag_idx.append(self.related[sort_pos[i][0]])
        return critical_flag_idx, model_new
    
    def searchBycritical(self, critical_flag):
        """
        Generate candidate configurations with focus on critical flags
        
        Args:
            critical_flag: List of critical flag indices
            
        Returns:
            List of candidate configurations focusing on critical flags
        """
        # Generate all combinations of 0/1 for critical flags (2^10 = 1024 combinations)
        permutations = list(itertools.product([0, 1], repeat=10))
        
        # Generate diverse random configurations
        seqs = []
        while len(seqs) < 1024 * 40:  # Generate 40K candidates
            x = random.randint(0, 2 ** self.dim - 1)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in seqs:
                seqs.append(initial_training_instance)
                
        # Set critical flags according to permutations to explore all combinations
        for i in range(len(permutations)):
            for idx in range(len(critical_flag)):
                for offset in range(0, 1024 * 40, 1024):
                    seqs[i + offset][critical_flag[idx]] = permutations[i][idx]
                    
        return seqs
    
    def run(self):
        """
        Main algorithm execution
        1. Build initial model
        2. Identify critical flags
        3. Search for optimal configuration using critical flags
        """
        begin_all = time.time()
        
        # Phase 1: Build initial model with small training set
        model, inital_indep, inital_dep = self.build_RF_by_CompTuner()
        
        # Phase 2: Identify critical flags
        critical_flag, model_new = self.get_critical_flags(model, inital_indep, inital_dep)
        
        all_before = time_tem[-1]
        begin_all = time.time()
        
        # Phase 3: Search using critical flags until time limit
        while (time_tem[-1] < 5000):  # 5000 seconds time budget
            # Generate candidates focusing on critical flags
            seq = self.searchBycritical(critical_flag)
            
            # Predict performance and sort by predicted speedup
            result = self.runtime_predict(model_new, seq)
            sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
            
            # Evaluate best predicted configuration
            true_reslut = self.get_objective_score(sorted_result[0][0], k_iter=0, SOURCE_PATH=self.SOURCE_PATH, GCC_PATH=self.GCC_PATH, INCLUDE_PATH=self.INCLUDE_PATH, EXEC_PARAM=self.EXEC_PARAM, LOG_FILE=self.LOG_FILE, all_flags=self.all_flags)
            
            # Update global best if needed
            if true_reslut > self.global_best_per:
                self.global_best_per = true_reslut
                self.global_best_seq = sorted_result[0][0]
                
            # Update time and log progress
            time_tem.append(time.time() - begin_all + all_before)
            ss = '{}: cur-best {}, cur-best-seq {}'.format(str(round(time_tem[-1])), str(self.global_best_per), str(self.global_best_seq))
            write_log(ss, self.LOG_FILE)
            
        # Final evaluation and logging
        best_result = self.get_objective_score(self.global_best_seq, k_iter=0, SOURCE_PATH=self.SOURCE_PATH, GCC_PATH=self.GCC_PATH, INCLUDE_PATH=self.INCLUDE_PATH, EXEC_PARAM=self.EXEC_PARAM, LOG_FILE=self.LOG_FILE, all_flags=self.all_flags)
        ss = '{}: cur-best {}, cur-best-seq {}'.format(str(round(time_tem[-1])), str(best_result), str(self.global_best_seq))
    
def read_flags_from_file(file_path):
    """
    Read compiler flags from a file
    
    Args:
        file_path: Path to the file containing comma-separated flags
        
    Returns:
        List of compiler flags
    """
    with open(file_path, 'r') as file:
        flags = file.read().strip()
    return [flag.strip() for flag in flags.split(',') if flag.strip()]

if __name__ == '__main__':
    # Create log directory if it doesn't exist
    LOG_DIR = 'log' + os.sep
    if not os.path.exists(LOG_DIR):
        os.system('mkdir '+LOG_DIR)
        
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CompTuner")
    
    parser.add_argument("--log_file", type=str, required=True,
                        help="File to save log")
    
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to the source program for tuning")
    
    parser.add_argument("--gcc_path", type=str, required=True,
                        help="Path of compiler")
    
    parser.add_argument("--exec_param", type=str, default=None,
                        help="Execution parameter for the output executable (can be empty)")
    
    parser.add_argument("--related_flags", type=str, default=None,
                        help="Related flags for tuning (can be a comma-separated string).")
    
    parser.add_argument("--flag_path", type=str, required=True,
                        help="Tuning flags file")
    
    args = parser.parse_args()
    
    # Set execution parameters
    if args.exec_param:
        EXEC_PARAM = args.exec_param
    else:
        EXEC_PARAM = '' 

    LOG_FILE = LOG_DIR + args.log_file

    # Parse related flags if provided
    if args.related_flags is not None:
        related_flags_list = [int(x) for x in args.related_flags.split(',')]
    else:
        related_flags_list = []

    # Read compiler flags from file
    if args.flag_path:
        all_flags = read_flags_from_file(args.flag_path)
    else:
        all_flags = ['-O2']
        print('No flags')

    # Initialize CFSCA parameters
    cfsca_params = {}
    cfsca_params['dim'] = len(all_flags)
    cfsca_params['get_objective_score'] = get_objective_score
    cfsca_params['seed'] = 456
    cfsca_params['related_flags'] = related_flags_list
    cfsca_params['source_path'] = args.source_path
    cfsca_params['gcc_path'] = args.gcc_path
    cfsca_params['include_path'] = '-I /home/user/polybench-code/utilities /home/user/polybench-code/utilities/polybench.c'
    cfsca_params['exec_param'] = args.exec_param
    cfsca_params['log_file'] = LOG_FILE
    cfsca_params['flags'] = all_flags
    
    # Create and run CFSCA instance
    cfsca = CFSCA(**cfsca_params)
    cfsca.run()