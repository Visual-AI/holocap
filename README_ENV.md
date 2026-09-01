## README for environment configuration

For details other than environment configuration, please refer to [README.md](README.md).

HoloCap uses two environments because model training/inference requires Transformers 4.46.1, while HoloScore requires Transformers 4.52.3 for Gemma3 and Qwen3-Embedding.

### Install (for training and inference)

1. Clone this repository and navigate to the HoloCap folder.

```bash
git clone https://github.com/Visual-AI/HoloCap.git
cd HoloCap
```

2. Create the `holocap` environment with Python 3.11.13 and CUDA Toolkit 12.4.

```bash
conda create -n holocap python=3.11.13 -y
conda activate holocap

conda install -y \
  -c nvidia/label/cuda-12.4.0 \
  cuda-toolkit=12.4.0 \
  conda-forge::sparsehash=2.0.4
```

3. Install the pinned SpatialLM dependencies.

```bash
git clone https://github.com/manycore-research/SpatialLM.git SpatialLM-upstream
cd SpatialLM-upstream
git checkout 029808be90daef80cf2b850e254716ca7d64254f

python -m pip install \
  "pip==25.1" \
  "setuptools==80.9.0" \
  "wheel==0.45.1" \
  "poetry==2.1.3" \
  "poethepoet==0.33.1"

poetry config virtualenvs.create false --local
poetry install --no-root
cd ..
```

4. Install the HoloCap training and inference packages.

```bash
python -m pip install \
  "transformers==4.46.1" \
  "tokenizers==0.20.3" \
  "huggingface-hub==0.33.2" \
  "peft==0.17.1" \
  "datasets==3.6.0" \
  "accelerate==1.7.0" \
  "numpy==1.26.4" \
  "scipy==1.16.0" \
  "pandas==2.3.1" \
  "open3d==0.18.0" \
  "einops==0.8.1" \
  "safetensors==0.4.5" \
  "addict==2.4.0" \
  "PyYAML==6.0.2" \
  "packaging==25.0" \
  "tqdm==4.67.1" \
  "omegaconf==2.3.0" \
  "tensorboardX==2.6.4" \
  "wandb==0.22.1"
```

5. Install the pinned Sonata components.

```bash
python -m pip install "ninja==1.11.1.4"
python -m pip install "flash-attn==2.8.0.post2" --no-build-isolation
python -m pip install \
  "torch-scatter==2.1.2+pt24cu124" \
  -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
python -m pip install \
  "timm==1.0.16" \
  "ccimport==0.4.4" \
  "pccm==0.4.16" \
  "pybind11==2.13.6" \
  "fire==0.7.0" \
  "cumm-cu120==0.4.11" \
  "spconv-cu120==2.3.6"
```

6. Install the geometry packages used by entity filtering and relation candidate generation.

```bash
python -m pip install --no-deps \
  "portalocker==3.2.0" \
  "iopath==0.1.10" \
  "trimesh==4.8.3" \
  "rtree==1.4.1"

CUDA_HOME="$CONDA_PREFIX" FORCE_CUDA=1 \
python -m pip install --no-deps --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@3aee2a6005cbf32275d705b4440deb3813f6a71e"
```

PyTorch3D must be compiled inside this environment against PyTorch 2.4.1 and CUDA 12.4. Do not reuse a build from another Conda environment.

7. Verify the installation from the HoloCap repository root.

```bash
python -c "import torch, transformers, peft, open3d, flash_attn, torch_scatter, spconv, trimesh, rtree; from pytorch3d.ops import box3d_overlap; from spatiallm.model.sonata_encoder import Sonata; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('transformers:', transformers.__version__); print('holocap imports OK')"
```

The base model `manycore-research/SpatialLM1.1-Qwen-0.5B` is downloaded automatically from Hugging Face when first used.

### Install for HoloScore

1. Create the `holocap-eval` environment with Python 3.10.18 and CUDA Toolkit 11.8.

```bash
conda create -n holocap-eval python=3.10.18 -y
conda activate holocap-eval

conda install -y \
  -c nvidia/label/cuda-11.8.0 \
  cuda-toolkit=11.8.0

python -m pip install \
  "pip==25.1" \
  "setuptools==78.1.1" \
  "wheel==0.45.1"
```

2. Install the PyTorch 2.6.0 cu118 stack.

```bash
python -m pip install \
  "torch==2.6.0" \
  "torchvision==0.21.0" \
  "torchaudio==2.6.0" \
  --index-url https://download.pytorch.org/whl/cu118
```

3. Install the HoloScore packages.

```bash
python -m pip install \
  "transformers==4.52.3" \
  "tokenizers==0.21.2" \
  "huggingface-hub==0.33.2" \
  "accelerate==1.9.0" \
  "sentence-transformers==5.1.1" \
  "FactualSceneGraph==0.5.0" \
  "bitsandbytes==0.48.1" \
  "numpy==2.1.2" \
  "scipy==1.15.3" \
  "pandas==2.3.1" \
  "scikit-learn==1.7.0" \
  "spacy==3.8.7" \
  "tabulate==0.9.0" \
  "Pillow==11.0.0" \
  "open3d==0.19.0" \
  "opencv-contrib-python==4.12.0.88" \
  "trimesh==4.8.3" \
  "rtree==1.4.1" \
  "nltk==3.9.2" \
  "tqdm==4.67.1" \
  "iopath==0.1.10" \
  "portalocker==3.2.0" \
  "safetensors==0.5.3" \
  "packaging==25.0" \
  "PyYAML==6.0.2" \
  "einops==0.8.1" \
  "typing-extensions==4.15.0"
```

FactualSceneGraph source: [zhuang-li/FACTUAL](https://github.com/zhuang-li/FACTUAL).

4. Install the CUDA extensions used by HoloScore.

```bash
python -m pip install "ninja==1.11.1.4"
python -m pip install "flash-attn==2.8.0.post2" --no-build-isolation

CUDA_HOME="$CONDA_PREFIX" FORCE_CUDA=1 \
python -m pip install --no-deps --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@3aee2a6005cbf32275d705b4440deb3813f6a71e"
```

5. Download the NLTK resources.

```bash
python -m nltk.downloader wordnet omw-1.4 punkt punkt_tab
```

The following model weights are downloaded automatically when HoloScore is first run:

- `google/gemma-3-1b-it`
- `Qwen/Qwen3-Embedding-0.6B`
- `sentence-transformers/all-mpnet-base-v2`

6. Verify the HoloScore installation.

```bash
python -c "import torch, transformers, open3d, cv2, trimesh, rtree, bitsandbytes, flash_attn; from pytorch3d.ops import box3d_overlap; from transformers import Gemma3ForCausalLM, Qwen3Model; from factual_scene_graph.evaluation.soft_spice_evaluation import encode_phrases; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('transformers:', transformers.__version__); print('holocap-eval imports OK')"
```
