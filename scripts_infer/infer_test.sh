export PYTHONWARNINGS="ignore"

NUM_GPUS=2
GPUS=("0" "1")

SPLIT_FILE='./data/splits/test.json'
echo "${SPLIT_FILE}"

MODEL_NAME='holocap-qwen2.5-0.5b-lora'
echo "${MODEL_NAME}"

echo 'Predicting entity instances and attributes...'
for INDEX in "${!GPUS[@]}"; do
    GPU_IDX=${GPUS[INDEX]}
    echo "${INDEX}"
    echo "${GPU_IDX}"
    CUDA_VISIBLE_DEVICES=${GPU_IDX} python -m scripts_infer.infer_entities \
        --list-file "${SPLIT_FILE}" \
        --num-chunk "${NUM_GPUS}" --chunk-idx "${INDEX}" \
        --model-path "ckpts/${MODEL_NAME}" \
        --output "results/${MODEL_NAME}/entities/" \
        --point-cloud 'data/holocap' \
        --temperature 1.0 --num-beams 1 &
done

wait

echo 'Filtering duplicated entity instances...'
python scripts_infer/filter_entities.py \
    --model-name "${MODEL_NAME}" \
    --old-attribute-path 'entities' \
    --new-attribute-path 'filtered_entities'

echo 'Generating instance neighbors for relation prediction...'
python scripts_infer/build_relation_candidates.py \
    --model-name "${MODEL_NAME}" \
    --attribute-path 'filtered_entities' \
    --relation-path 'relation_candidates'

echo 'Predicting instance pairs and descriptions...'
for INDEX in "${!GPUS[@]}"; do
    GPU_IDX=${GPUS[INDEX]}
    echo "${INDEX}"
    echo "${GPU_IDX}"
    CUDA_VISIBLE_DEVICES=${GPU_IDX} python -m scripts_infer.infer_relations \
        --list-file "${SPLIT_FILE}" \
        --num-chunk "${NUM_GPUS}" --chunk-idx "${INDEX}" \
        --model-path "ckpts/${MODEL_NAME}" \
        --pre-prompt "results/${MODEL_NAME}/relation_candidates/" \
        --output "results/${MODEL_NAME}/relations/" \
        --point-cloud 'data/holocap' \
        --temperature 1.0 --num-beams 1 &
done

wait
