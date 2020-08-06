#!/usr/bin/env bash

echo "Model type: ${1}"
echo "Model path: ${2}"
echo "No ext sample name: ${3}"

# Modify params.
export DATA_DIR=$(dirname ${3})
export TEST_BASE=$(basename ${3})

# Fixed params.
export MAX_LENGTH=128

export REGARD1_OUTPUT_DIR=${2}
export SENTIMENT1_OUTPUT_DIR=${2}
export TEST_FILE=${TEST_BASE}.tsv.XYZ

if [[ ${1} == "regard1" ]]
then
    export OUTPUT_DIR=${REGARD1_OUTPUT_DIR}
    export BERT_MODEL3=${2}
    export MODEL_VERSION=1
elif [[ ${1} == "sentiment1" ]]
then
    export OUTPUT_DIR=${SENTIMENT1_OUTPUT_DIR}
    export BERT_MODEL3=${2}
    export MODEL_VERSION=1
fi
export ENSEMBLE_DIR=${OUTPUT_DIR}/generated_data_ensemble
export OUTPUT_PREFIX=${DATA_DIR}/${1}_${TEST_BASE}.tsv


echo "Labeling with BERT classifier..."
python scripts/run_classifier.py --data_dir ${DATA_DIR} \
--model_type bert \
--model_name_or_path ${BERT_MODEL3} \
--output_dir ${BERT_MODEL3}  \
--max_seq_length  ${MAX_LENGTH} \
--do_predict \
--ens \
--test_file ${TEST_FILE} \
--do_lower_case \
--overwrite_cache \
--per_gpu_eval_batch_size 32 \
--model_version ${MODEL_VERSION}

echo "Collecting majority labels..."
mkdir -p ${ENSEMBLE_DIR}
cp ${OUTPUT_DIR}/${TEST_BASE}_predictions.txt ${ENSEMBLE_DIR}/1.txt
python scripts/ensemble.py --data_dir ${ENSEMBLE_DIR} --output_prefix ${OUTPUT_PREFIX} --file_with_demographics ${DATA_DIR}/${TEST_BASE}.tsv

echo "Done!"
