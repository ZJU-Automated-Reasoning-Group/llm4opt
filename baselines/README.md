# Baselines

## Auto-tuning

### CFSCA

采用两阶段方法，先识别关键优化标志，再进行针对性调优

~~~~
# 第一步：获取相关标志
python getrelated.py --source_path=/path/to/program --flag_path=/path/to/flags.txt

# 第二步：执行自动调优
python CFSCA.py --log_file=correlation_cfsca.log \
                --source_path=/path/to/program \
                --gcc_path=gcc \
                --flag_path=/path/to/flags.txt \
                --related_flags=1,2,3,4,5,6,7,8,9,10
~~~~

### CompTuner
通过多阶段学习进行编译器自动调优

~~~~
python CompTuner.py --log_file=correlation_comptuner.log \
                    --source_path=/path/to/program \
                    --gcc_path=gcc \
                    --flag_path=/path/to/flags.txt
~~~~


### GRACE

用于评估和测试各种优化方法的性能

支持的测试数据集：
- cbench-v1
- mibench-v1
- chstone-v0
- tensorflow-v0
- npb-v0
- opencv-v0
- blas-v0

支持的优化方法：
- none（无优化）
- ga_seq（序列遗传算法）
- prefix（前缀优化）
- oz（Oz优化器）

### mlopt

基于遗传算法的超参数优化



## Prediction