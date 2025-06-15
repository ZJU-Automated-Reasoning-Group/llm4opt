#!/bin/bash

PYTHON_SCRIPT_NAME="test_testfiles.py" 
BASE_TEST_DIR="./dataset/test_final/"
CLUSTER_RESULTS_DIR="./results_all_clusters/"
LLVM_TOOLS_PATH="./llvm_tools"
NUM_CLUSTERS=100
OUTPUT_CSV_PREFIX="test_eval" 
MAX_WORKERS=16 
MAX_WORKERS_GA=8 
GA_CANDIDATES_FACTOR=20
GA_ALLOW_ADD_DUPLICATES="" 

DATASETS=("cbench-v1" "mibench-v1" "chstone-v0" "tensorflow-v0" "npb-v0" "opencv-v0" "blas-v0")
REFINEMENT_METHODS=(none ga_seq prefix oz)

RESULTS_TABLE_FILE="final_summary_table.txt"
TEMP_STATS_DIR="temp_stats_output"
rm -rf "${TEMP_STATS_DIR}"
mkdir -p "${TEMP_STATS_DIR}"


echo "Generating summary table in ${RESULTS_TABLE_FILE}..."
printf "%-15s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
    "Dataset" "Method" "ArithMean" "GeoMean" "Median" "Success%" "InitialWorse%" "FinalWorse%" \
    > "${RESULTS_TABLE_FILE}"
echo "-----------------------------------------------------------------------------------------------------------------------------" >> "${RESULTS_TABLE_FILE}"


for dataset_subdir in "${DATASETS[@]}"; do
    for refinement_method in "${REFINEMENT_METHODS[@]}"; do
        echo ""
        echo "======================================================================"
        echo "Running: Dataset = ${dataset_subdir}, Method = ${refinement_method}"
        echo "======================================================================"

        PYTHON_CMD="python ${PYTHON_SCRIPT_NAME} \
            --base_test_dir ${BASE_TEST_DIR} \
            --specific_test_subdir ${dataset_subdir} \
            --results_dir ${CLUSTER_RESULTS_DIR} \
            --llvm_tools ${LLVM_TOOLS_PATH} \
            --num_clusters ${NUM_CLUSTERS} \
            --output_csv_prefix ${OUTPUT_CSV_PREFIX} \
            --max_workers ${MAX_WORKERS} \
            --max_workers_ga ${MAX_WORKERS_GA} \
            --refinement_method ${refinement_method} \
            --ga_candidates_factor ${GA_CANDIDATES_FACTOR} \
            ${GA_ALLOW_ADD_DUPLICATES}"

        echo "Executing: ${PYTHON_CMD}"

        TEMP_PYTHON_OUTPUT="${TEMP_STATS_DIR}/py_out_${dataset_subdir}_${refinement_method}.log"
        STATS_OUTPUT=$(${PYTHON_CMD} 2>&1 | tee "${TEMP_PYTHON_OUTPUT}" | awk '/--- SCRIPT_SUMMARY_STATS_START ---/{flag=1; next} /--- SCRIPT_SUMMARY_STATS_END ---/{flag=0} flag')

        if [ -z "${STATS_OUTPUT}" ]; then
            echo "WARNING: No stats output captured for ${dataset_subdir} with method ${refinement_method}. Check ${TEMP_PYTHON_OUTPUT}."
            printf "%-15s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
                "${dataset_subdir}" "${refinement_method}" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A" \
                >> "${RESULTS_TABLE_FILE}"
            continue
        fi

        arith_mean=$(echo "${STATS_OUTPUT}" | grep "Arithmetic Mean Improvement (%):" | cut -d':' -f2 | xargs)
        geo_mean=$(echo "${STATS_OUTPUT}" | grep "Geometric Mean Improvement (%):" | cut -d':' -f2 | xargs)
        median=$(echo "${STATS_OUTPUT}" | grep "Median Improvement (%):" | cut -d':' -f2 | xargs)
        success_rate=$(echo "${STATS_OUTPUT}" | grep "Refinement Success Rate (%):" | cut -d':' -f2 | xargs)
        initial_worse_rate=$(echo "${STATS_OUTPUT}" | grep "Initial Worse Than Oz Rate (%):" | cut -d':' -f2 | xargs) # 你需要在Python中实现这个
        final_worse_rate=$(echo "${STATS_OUTPUT}" | grep "Final Worse Than Oz Rate (%):" | cut -d':' -f2 | xargs)

        arith_mean=${arith_mean:-N/A}
        geo_mean=${geo_mean:-N/A}
        median=${median:-N/A}
        success_rate=${success_rate:-N/A}
        initial_worse_rate=${initial_worse_rate:-N/A}
        final_worse_rate=${final_worse_rate:-N/A}

        printf "%-15s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
            "${dataset_subdir}" \
            "${refinement_method}" \
            "${arith_mean}" \
            "${geo_mean}" \
            "${median}" \
            "${success_rate}" \
            "${initial_worse_rate}" \
            "${final_worse_rate}" \
            >> "${RESULTS_TABLE_FILE}"

        echo "Finished: Dataset = ${dataset_subdir}, Method = ${refinement_method}"
       
    done
done

echo "-----------------------------------------------------------------------------------------------------------------------------" >> "${RESULTS_TABLE_FILE}"
echo "All tests finished. Summary table saved to ${RESULTS_TABLE_FILE}"
echo "Individual run logs and detailed CSVs are in the current directory and '${TEMP_STATS_DIR}'."