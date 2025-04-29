# CFSCA


The folder CFCSA is the code for Compiler Auto-tuning via Critical Flag Selection, which contains getrelated.py and CFSCA.py. For example, if you want to use it to tune program correlation, you firstly need run getrelated.py as command python getrelated.py --source_path=/home/user/polybench-code/datamining/correlation --flag_path=/home/user/flag.txt, to obtain the related flags of the target program. Then you can input command python CFSCA.py --log_file=correlation_cfsca.log --source_path=/home/user/polybench-code/datamining/correlation --gcc_path=gcc --flag_path=/home/user/flag.txt --related_flags=1,2,3,4,5,6,7,8,9,10.

In this command, --log_file is your log file name, --source_path is your program path, --gcc_path is your compiler path, --related_flags are related flags' index, and --flag_path is for your tuning optimization flags. Moreover, if your program has parameters, you need to input --exec_param. 