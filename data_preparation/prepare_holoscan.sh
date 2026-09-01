#!/usr/bin/env bash
set -euo pipefail

# Configure the source datasets and processed-data destination before running.
SCANNET_ROOT="/path/to/ScanNet/scannetv2"
RSCAN_ROOT="/path/to/3RScan"
MATTERPORT3D_ROOT="/path/to/Matterport3D/v1"
ARKITSCENES_ROOT="/path/to/ARKitScenes"
STRUCTURED3D_ROOT="/path/to/Structured3D/Panorama/Structured3D"
OUTPUT_ROOT="/path/to/processed/HoloScan"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SOURCE_ROOTS=(
    "${SCANNET_ROOT}"
    "${RSCAN_ROOT}"
    "${MATTERPORT3D_ROOT}"
    "${ARKITSCENES_ROOT}"
    "${STRUCTURED3D_ROOT}"
)

for source_root in "${SOURCE_ROOTS[@]}"; do
    if [[ "${source_root}" == /path/to/* ]]; then
        echo "Configure all dataset roots in ${BASH_SOURCE[0]} before running." >&2
        exit 1
    fi
    if [[ ! -d "${source_root}" ]]; then
        echo "Dataset root does not exist: ${source_root}" >&2
        exit 1
    fi
done

if [[ "${OUTPUT_ROOT}" == /path/to/* ]]; then
    echo "Configure OUTPUT_ROOT in ${BASH_SOURCE[0]} before running." >&2
    exit 1
fi

SCANNET_OUTPUT_DIR="${OUTPUT_ROOT}/scannet/holoscan_axisaligned_plys"
RSCAN_OUTPUT_DIR="${OUTPUT_ROOT}/3rscan/holoscan_axisaligned_plys"
MATTERPORT3D_OUTPUT_DIR="${OUTPUT_ROOT}/matterport3d/holoscan_axisaligned_plys"
ARKITSCENES_OUTPUT_DIR="${OUTPUT_ROOT}/arkitscenes/holoscan_axisaligned_plys"
STRUCTURED3D_OUTPUT_DIR="${OUTPUT_ROOT}/structured3d/holoscan_axisaligned_plys"

OUTPUT_DIRS=(
    "${SCANNET_OUTPUT_DIR}"
    "${RSCAN_OUTPUT_DIR}"
    "${MATTERPORT3D_OUTPUT_DIR}"
    "${ARKITSCENES_OUTPUT_DIR}"
    "${STRUCTURED3D_OUTPUT_DIR}"
)
mkdir -p "${OUTPUT_DIRS[@]}"

create_point_cloud_link() {
    local target="$1"
    local link_path="$2"
    if [[ -e "${link_path}" && ! -L "${link_path}" ]]; then
        echo "Refusing to replace non-symlink path: ${link_path}" >&2
        exit 1
    fi
    ln -sfn "${target}" "${link_path}"
}

create_point_cloud_link "${SCANNET_OUTPUT_DIR}" "${REPO_ROOT}/data/holocap/scannet_pcds"
create_point_cloud_link "${RSCAN_OUTPUT_DIR}" "${REPO_ROOT}/data/holocap/3rscan_pcds"
create_point_cloud_link "${MATTERPORT3D_OUTPUT_DIR}" "${REPO_ROOT}/data/holocap/matterport3d_pcds"
create_point_cloud_link "${ARKITSCENES_OUTPUT_DIR}" "${REPO_ROOT}/data/holocap/arkitscenes_pcds"
create_point_cloud_link "${STRUCTURED3D_OUTPUT_DIR}" "${REPO_ROOT}/data/holocap/structured3d_pcds"

for run_idx in 0 1; do
    "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/get_axisalign_data_scannet.py" \
        --data-root "${SCANNET_ROOT}" --output-dir "${SCANNET_OUTPUT_DIR}" --run-idx "${run_idx}"
done
"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/get_axisalign_data_scannet.py" \
    --data-root "${SCANNET_ROOT}" --output-dir "${SCANNET_OUTPUT_DIR}" --val

"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/get_axisalign_data_3rscan.py" \
    --data-root "${RSCAN_ROOT}" --output-dir "${RSCAN_OUTPUT_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/get_axisalign_data_3rscan.py" \
    --data-root "${RSCAN_ROOT}" --output-dir "${RSCAN_OUTPUT_DIR}" --val

"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/get_axisalign_data_matterport3d.py" \
    --data-root "${MATTERPORT3D_ROOT}" --output-dir "${MATTERPORT3D_OUTPUT_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/get_axisalign_data_matterport3d.py" \
    --data-root "${MATTERPORT3D_ROOT}" --output-dir "${MATTERPORT3D_OUTPUT_DIR}" --val

"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/get_axisalign_data_arkitscenes.py" \
    --data-root "${ARKITSCENES_ROOT}" --output-dir "${ARKITSCENES_OUTPUT_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/get_axisalign_data_arkitscenes.py" \
    --data-root "${ARKITSCENES_ROOT}" --output-dir "${ARKITSCENES_OUTPUT_DIR}" --val

for ((run_idx = 0; run_idx <= 22; run_idx++)); do
    "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/get_axisalign_data_structured3d.py" \
        --data-root "${STRUCTURED3D_ROOT}" --output-dir "${STRUCTURED3D_OUTPUT_DIR}" --run-idx "${run_idx}"
done

echo "HoloScan axis-aligned point clouds are ready under ${OUTPUT_ROOT}."
