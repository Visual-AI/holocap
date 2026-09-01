SPLIT_FILE='./data/splits/test.json'
ANNOTATION_FILE='./data/annotations/test.json'
MODEL_NAME='holocap-qwen2.5-0.5b-lora'

REFERENCE_DIR="./results_eval/test/reference_atomic_attributes"
MODEL_RESULTS_DIR="./results_eval/test/${MODEL_NAME}"
ENTITY_PREDICTIONS_DIR="./results/${MODEL_NAME}/filtered_entities"
RELATION_PREDICTIONS_DIR="./results/${MODEL_NAME}/relations"
ENTITY_MAPPING_DIR="${MODEL_RESULTS_DIR}/entity_mapping"
PREDICTED_ATOMIC_ATTRIBUTES_DIR="${MODEL_RESULTS_DIR}/predicted_atomic_attributes"
ATTRIBUTE_METRICS_DIR="${MODEL_RESULTS_DIR}/attribute_metrics"
RELATION_METRICS_DIR="${MODEL_RESULTS_DIR}/relation_metrics"

echo "Evaluating split: ${SPLIT_FILE}"
echo "Model: ${MODEL_NAME}"

echo 'Preparing reference atomic attributes...'
CUDA_VISIBLE_DEVICES=0,1 python scripts_holoscore/atomize_test_references.py \
    --num-idx 83 || exit 1

echo 'Evaluating entity tags and locations...'
CUDA_VISIBLE_DEVICES=0 python scripts_holoscore/match_entities.py \
    --list "${SPLIT_FILE}" --anno "${ANNOTATION_FILE}" \
    --prediction-dir "${ENTITY_PREDICTIONS_DIR}" \
    --map-dir "${ENTITY_MAPPING_DIR}" \
    --entity-thr 0.5 --location-thr 0.3

echo 'Atomizing predicted attributes...'
CUDA_VISIBLE_DEVICES=0,1 python scripts_holoscore/atomize_predictions.py \
    --list "${SPLIT_FILE}" --anno "${ANNOTATION_FILE}" \
    --prediction-dir "${ENTITY_PREDICTIONS_DIR}" \
    --prediction-atom-dir "${PREDICTED_ATOMIC_ATTRIBUTES_DIR}" \
    --gt-dir "${REFERENCE_DIR}" \
    --map-dir "${ENTITY_MAPPING_DIR}" \
    --num-idx 83

echo 'Evaluating attributes...'
CUDA_VISIBLE_DEVICES=0 python scripts_holoscore/evaluate_test_attributes.py \
    --list "${SPLIT_FILE}" --anno "${ANNOTATION_FILE}" \
    --prediction-dir "${PREDICTED_ATOMIC_ATTRIBUTES_DIR}" \
    --gt-dir "${REFERENCE_DIR}" \
    --map-dir "${ENTITY_MAPPING_DIR}" \
    --save-dir "${ATTRIBUTE_METRICS_DIR}"

echo 'Evaluating relations...'
CUDA_VISIBLE_DEVICES=1 python scripts_holoscore/evaluate_test_relations.py \
    --list "${SPLIT_FILE}" --anno "${ANNOTATION_FILE}" \
    --prediction-dir "${RELATION_PREDICTIONS_DIR}" \
    --map-dir "${ENTITY_MAPPING_DIR}" \
    --save-dir "${RELATION_METRICS_DIR}"

echo 'Aggregating HoloScore metrics...'
python scripts_holoscore/aggregate_test_metrics.py \
    --list "${SPLIT_FILE}" --anno "${ANNOTATION_FILE}" \
    --attr-pred-dir "${PREDICTED_ATOMIC_ATTRIBUTES_DIR}" \
    --rel-pred-dir "${RELATION_PREDICTIONS_DIR}" \
    --map-dir "${ENTITY_MAPPING_DIR}" \
    --attr-save-dir "${ATTRIBUTE_METRICS_DIR}" \
    --rel-save-dir "${RELATION_METRICS_DIR}" \
    --attribute-thr 0.6 --relation-thr 0.6
