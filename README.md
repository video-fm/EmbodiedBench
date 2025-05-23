<h1 align="center">
<img src="docs/images/embodied-logo.png" alt="embodied-logo" width="40" height="40" style="vertical-align: middle; margin-top: -12px;">
ESCA: Contextualizing Embodied Agents via Scene-Graph Generation
</h1>




# Overview 
We introduce **ESCA** (Embodied and Scene-Graph Contextualized Agent), a framework designed to contextualize Multi-modal Large Language Models (MLLMs) through open-domain scene graph generation. ESCA provides structured visual grounding, helping MLLMs make sense of complex and ambiguous sensory environments. At its core is SGClip, a CLIP-based model that captures semantic visual features, including entity classes, physical attributes, actions, interactions, and inter-object relations.

ESCA operates as a multi-stage pipeline, generating task-aware scene graphs that reflect both visual content and human instructions. Through experiments on two challenging embodied environments, we demonstrate that ESCA consistently improves the performance of all evaluated MLLMs, including both open-source and proprietary models.


## 🚀 **Key Features** 

- 🛠️ **Structured Scene Understanding:**  
  ESCA decomposes visual understanding into four modular stages: concept extraction, object identification, scene graph prediction, and visual summarization.

- 🎯 **SGClip Model:**  
  A CLIP-based foundation model for structured scene understanding that supports open-domain concept coverage and probabilistic predictions.

- ⚡ **Transfer Protocol:**  
  A general transfer protocol based on customizable prompt templates that enables ESCA to generalize across different downstream tasks.

- 🏹 **ESCA-Video-87K Dataset:**  
  A large-scale dataset with 87K video clips, paired with natural language captions, object traces, and spatial-temporal programmatic specifications.

- 🔧 **Neurosymbolic Learning:**  
  A model-driven, self-supervised learning pipeline that eliminates the need for manual scene graph annotations.

# 🖥️ Installation
This repository is built on top of [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench). We maintain all the original setup instructions while adding our ESCA-specific components.

**Note: we need to install three conda environments, one for EB-ALFRED and EB-Habitat, one for EB-Navigation, and one for EB-Manipulation. Please use ssh download instead of HTTP download to avoid error during git lfs pull.**

Download repo
```bash
cd EmbodiedBench
```

**You have two options for installation: you can either use 
```bash install.sh``` or manually run the provided commands. After completing the installation with `bash install.sh`, you will need to start the headless server and verify that each environment is properly set up.**

Environment for ```EB-Navigation```
```bash
conda env create -f conda_envs/environment_eb-nav.yaml 
conda activate embench_nav
pip install -e .
```
3️⃣ Environment for ```EB-Manipulation```
```bash
conda env create -f conda_envs/environment_eb-man.yaml 
conda activate embench_man
pip install -e .
```

**Note: EB-Alfred, EB-Habitat and EB-Manipulation require downloading large datasets from Hugging Face or GitHub repositories. Ensure Git LFS is properly initialized by running the following commands:**
```bash
git lfs install
git lfs pull
```

## Environment Setup

### GroundingDINO Setup
1. Clone the GroundingDINO repository:
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
```

2. Download the required model files:
```bash
# Create checkpoints directory
mkdir -p GroundingDINO/checkpoints

# Download the model checkpoint
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0/groundingdino_swint_ogc.pth -P GroundingDINO/checkpoints/
```

3. Set the environment variable for GroundingDINO path:
```bash
export GROUNDING_DINO_PATH=/path/to/your/GroundingDINO
```

### LASER Model Setup
1. Please download the supplementary LASER code


2. Set the environment variable for the LASER model path (replace the path with the actual location of the `src/models` directory):
```bash
export LASER_MODEL_PATH=/path/to/your/LASER-unified/src/models
```

**Note:**
You must set this environment variable in every shell/session where you run the code, or add it to your `.bashrc`/`.zshrc` for convenience.

### LASER Model Weights

Please download the LASER model weights from the following website:

[Download LASER model weights here](<INSERT_LINK_HERE>)

After downloading, move the model file(s) to the `models` directory in your EmbodiedBench folder:

```bash
mv /path/to/downloaded/model_file.model /path/to/EmbodiedBench/models/
```

## Start Headless Server
Please run startx.py script before running experiment on headless servers. The server should be started in another tmux window. We use X_DISPLAY id=1 by default.
```bash
python -m embodiedbench.envs.eb_alfred.scripts.startx 1
```

## Running ESCA Evaluators

### EB-Navigation Evaluator
To run the EB-Navigation evaluator with ESCA:

```bash
conda activate embench_nav
python -m embodiedbench.evaluator.eb_gd_esca_navigation_evaluator
```

Both evaluators support various configuration options through command-line arguments or config files. Key parameters include:
- `model_name`: The MLLM to use (e.g., 'gpt-4o', 'gemini-2.0-flash')
- `n_shots`: Number of examples for few-shot learning
- `detection_box`: Enable/disable detection box visualization
- `sg_text`: Enable/disable scene graph text output
- `gd_only`: Use only Grounding DINO for object detection without the scene graph generation of ESCA
- `top_k`: Number of top predictions to consider
- `aggr_thres`: Aggregation threshold for predictions

# Citation
If you use this codebase, please cite both our paper and the original EmbodiedBench paper:

```
@article{esca2025,
  title={ESCA: Contextualizing Embodied Agents via Scene-Graph Generation},
  author={Anonymous Author(s)},
  journal={arXiv preprint},
  year={2025}
}

@article{yang2025embodiedbench,
  title={EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents},
  author={Yang, Rui and Chen, Hanyang and Zhang, Junyu and Zhao, Mark and Qian, Cheng and Wang, Kangrui and Wang, Qineng and Koripella, Teja Venkat and Movahedi, Marziyeh and Li, Manling and others},
  journal={arXiv preprint arXiv:2502.09560},
  year={2025}
}
```
