## README for data preparation

HoloScan uses indoor scenes from [ScanNet](https://github.com/scannet/scannet), [3RScan](https://github.com/WaldJohannaU/3RScan), [Matterport3D](https://github.com/niessner/Matterport), [ARKitScenes](https://github.com/apple/ARKitScenes), and Structured3D. The raw datasets are distributed under their respective licenses. Follow each official repository to complete any required Terms of Use or access request before downloading.

### Download HoloScan annotations and metadata

The HoloScan training annotations, validation and test annotations, and axis-alignment metadata are hosted in the [HoloScan dataset repository](https://huggingface.co/datasets/LasNack/HoloScan). After configuring the `holocap` environment, run the following command from the HoloCap repository root:

```bash
huggingface-cli download LasNack/HoloScan \
  data/holocap/holocap_train.json \
  data/annotations/val.json \
  data/annotations/test.json \
  data_preparation/metadata/holocap_train.pkl \
  data_preparation/metadata/holocap_val.pkl \
  data_preparation/metadata/structured3d.pkl \
  --repo-type dataset \
  --local-dir .
```

The six files are downloaded directly into their expected paths in the HoloCap repository.

### Download raw datasets

#### ScanNet

Follow the official [ScanNet download instructions](https://github.com/scannet/scannet) to request ScanNet v2 access and obtain the official `download-scannet.py` script. Download the release with:

```bash
python download-scannet.py -o /path/to/ScanNet/scannetv2
```

HoloCap requires the reconstructed meshes in the following layout:

```text
SCANNET_ROOT/
└── scans/
    └── <scene_id>/
        └── <scene_id>_vh_clean.ply
```

#### 3RScan

Follow the official [3RScan download instructions](https://github.com/WaldJohannaU/3RScan), including the Terms of Use form. Download the textured 3D models and organize the extracted scan folders under `scans/`:

```text
RSCAN_ROOT/
└── scans/
    └── <scan_id>/
        ├── mesh.refined.v2.obj
        ├── mesh.refined.mtl
        └── mesh.refined_*.png
```

The material and texture files are required because HoloCap transfers the mesh colors to the aligned point cloud.

#### Matterport3D

Follow the official [Matterport3D download instructions](https://github.com/niessner/Matterport), sign the Terms of Use, and download the `region_segmentations` data. The official data layout is:

```text
MATTERPORT3D_ROOT/
└── scans/
    └── <building_id>/
        └── region_segmentations/
            └── <region_id>.ply
```

The current conversion script reads the same data through `scans_embodiedscan`. Create a link once after downloading:

```bash
cd /path/to/Matterport3D/v1
ln -s scans scans_embodiedscan
```

Set `MATTERPORT3D_ROOT` to `/path/to/Matterport3D/v1` in the HoloCap preparation script.

#### ARKitScenes

Clone the official [ARKitScenes repository](https://github.com/apple/ARKitScenes), accept its license terms, and download the 3DOD data with the official script:

```bash
git clone https://github.com/apple/ARKitScenes.git
cd ARKitScenes
python download_data.py 3dod \
  --video_id_csv threedod/3dod_train_val_splits.csv \
  --download_dir /path/to/ARKitScenes/3dod
```

HoloCap reads the ARKit mesh files from:

```text
ARKITSCENES_ROOT/
└── 3dod/
    └── Training/
        └── <scene_id>/
            └── <scene_id>_3dod_mesh.ply
```

#### Structured3D

TODO: add the Structured3D download and raw-data preparation instructions.

### Configure HoloCap paths

The axis-alignment metadata is provided in `data_preparation/metadata/`. The retained train and validation metadata together cover all scenes in the released HoloCap validation and test splits.

Edit the following absolute paths at the top of `data_preparation/prepare_holoscan.sh`:

```bash
SCANNET_ROOT="/path/to/ScanNet/scannetv2"
RSCAN_ROOT="/path/to/3RScan"
MATTERPORT3D_ROOT="/path/to/Matterport3D/v1"
ARKITSCENES_ROOT="/path/to/ARKitScenes"
STRUCTURED3D_ROOT="/path/to/Structured3D/Panorama/Structured3D"
OUTPUT_ROOT="/path/to/processed/HoloScan"
```

### Generate aligned point clouds

Activate the training and inference environment, then run the conversion pipeline:

```bash
conda activate holocap
bash data_preparation/prepare_holoscan.sh
```

The script converts the raw meshes into axis-aligned PLY point clouds and creates the point-cloud links required by training and inference. Processed files are written as:

```text
OUTPUT_ROOT/
├── scannet/holoscan_axisaligned_plys/
├── 3rscan/holoscan_axisaligned_plys/
├── matterport3d/holoscan_axisaligned_plys/
├── arkitscenes/holoscan_axisaligned_plys/
└── structured3d/holoscan_axisaligned_plys/
```

The repository data layout after preparation is:

```text
data/
├── annotations/
│   ├── val.json
│   └── test.json
├── holocap/
│   ├── dataset_info.json
│   ├── holocap_train.json
│   ├── scannet_pcds -> OUTPUT_ROOT/scannet/holoscan_axisaligned_plys
│   ├── 3rscan_pcds -> OUTPUT_ROOT/3rscan/holoscan_axisaligned_plys
│   ├── matterport3d_pcds -> OUTPUT_ROOT/matterport3d/holoscan_axisaligned_plys
│   ├── arkitscenes_pcds -> OUTPUT_ROOT/arkitscenes/holoscan_axisaligned_plys
│   └── structured3d_pcds -> OUTPUT_ROOT/structured3d/holoscan_axisaligned_plys
└── splits/
    ├── val.json
    └── test.json
```
