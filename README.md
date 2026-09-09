############################################################################
  The Ceiling Effect in Few-Shot Personalization: How Baseline Accuracy
    Shapes the Limits of User Adaptation in Wearable Activity Recognition
 
             Rabdaa Amjad, Muhammad Hassan Khan,* Muhammad Adeel Nisar


      Faculty of Computing & Information Technology, University of the Punjab,
                              Lahore 54590, Pakistan
                     *Corresponding Author: hassankhan@pucit.edu.pk

############################################################################
 

## Overview

This repository contains the complete codebase used to perform data preprocessing, multi-stream backbone pre-training, few-shot meta-learning, subject-specific personalization, and empirical result visualization for the study *"The Ceiling Effect in Few-Shot Personalization: How Baseline Accuracy Shapes the Limits of User Adaptation in Wearable Activity Recognition"*. 

The framework systematically evaluates three multi-stream backbone architectures (**1D Convolutional Neural Network**, **Bidirectional LSTM**, and **Sensor-as-Token Transformer**) paired with four few-shot adaptation strategies (**Prototypical Networks**, **First-Order MAML**, **Latent Embedding Exploitation (LEE)**, and **Supervised Contrastive Learning (SupCon)**) across five heterogeneous multimodal wearable benchmark datasets:
1. **WISDM** (51 subjects, 18 activities, Smartphone + Smartwatch)
2. **CogAge Atomic** (8 subjects, 61 fine-grained gestures, Glasses + Band + Phone)
3. **CogAge Composite** (6 subjects, 7 complex multi-step ADL tasks)
4. **HumCare ADL** (60 subjects, 29 daily living activities, Phone + Watch + Glasses)
5. **HumCare AF** (85 subjects, 28 activity & fall classes)

---

## Required Environment and Libraries

The codebase is written in **Python 3.10+** and supports execution in **Google Colab** (with GPU acceleration, e.g., T4/V100) or on a **local workstation** with CUDA support.

### Key Python Dependencies:
* **PyTorch (>= 2.0.1)**: Deep learning framework used to construct and train stream encoders, fusion heads, attention blocks, and few-shot episodic loops.
* **torchvision / torchaudio**: Auxiliary tensor transformations and signal pipelines.
* **scikit-learn (>= 1.2.0)**: Subject-independent partitioning, stratified batching, evaluation metrics (accuracy, Macro-F1, confusion matrices).
* **NumPy (>= 1.23.0) & SciPy (>= 1.10.0)**: Numerical array processing, linear interpolation resampling (50 Hz), Z-score channel normalization.
* **Pandas (>= 1.5.0)**: Benchmark logging, table compilation, and metadata management.
* **Matplotlib (>= 3.7.0) & Seaborn (>= 0.12.0)**: Vector-grade publication figures, $K$-shot trajectories, regression scatter plots, and heatmaps.
* **tqdm**: Progress monitoring during pre-training and episodic evaluation loops.

---

## Setup of the Environment

### 1. Repository Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/rabdaa/FewShot-HAR-Personalization.git
cd FewShot-HAR-Personalization
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the target datasets from their official sources:
* **WISDM**: Download from the official WISDM repository (Kwapisz et al.).
* **CogAge (Atomic & Composite)**: Download from Frédéric Li et al. (Sensors 2020).
* **HumCare ADL & AF**: Download from the HumCare project portal (http://faculty.pucit.edu.pk/hassankhan/humcare/datasets.html).

Place raw datasets inside a directory named `data/` or in your Google Drive (e.g., `/MyDrive/HAR_Datasets/`).

### 3. Google Drive / Local Path Configuration
* If running on **Google Colab**: Run the initial mount cell (`from google.colab import drive; drive.mount('/content/drive')`).
* If running **Locally**: Set `DATA_ROOT = "./data"` and `OUTPUT_ROOT = "./fsl_har_results"` in `Notebooks/00_config_and_utils.ipynb`.

---

## Code Execution Steps & Pipeline

The workflow is structured into 16 sequential Jupyter notebooks located in the `Notebooks/` directory:

### Phase 1: Configuration & Preprocessing
* **`00_config_and_utils.ipynb`**: Global configuration, seed fixing, sensor stream mapping, evaluation metrics, and general helper functions.
* **`01_preprocess_wisdm.ipynb`**: Resampling to 50 Hz, 5-second windowing (250 samples, 50% overlap), Z-score normalization, and 38/6/7 subject splitting on WISDM.
* **`02_preprocess_cogage_atomic.ipynb`**: Multimodal stream alignment (Glasses, Band, Phone, Magnetometer), length equalization, and 6/1/1 subject splitting on CogAge Atomic.
* **`03_preprocess_cogage_composite.ipynb`**: 10-stream multimodal processing and 4/1/1 subject partitioning for CogAge Composite.
* **`04_preprocess_humcare.ipynb`**: 9-stream standardized preprocessing (Phone, Watch, Glasses) and subject-level train/val/test partitioning for HumCare ADL (45/7/8) and HumCare AF (63/10/12).

### Phase 2: Multi-Stream Backbone Pre-training
* **`05_train_backbone_cnn.ipynb`**: Pre-trains the multi-stream 1D CNN backbone with Cosine Annealing, Adam optimizer, label smoothing (0.1), and early stopping.
* **`06_train_backbone_lstm.ipynb`**: Pre-trains the multi-stream Bidirectional LSTM backbone ($64$ units/direction, 128-d late fusion).
* **`07_train_backbone_transformer.ipynb`**: Pre-trains the Sensor-as-Token Transformer backbone (convolutional tokenizer + 2 encoder layers, 4 heads, AdamW, 5 warmup epochs).

### Phase 3: Few-Shot Learning & Adaptation Strategies
* **`08_protonet.ipynb`**: Episodic meta-training and metric prototype classification using Prototypical Networks ($N=5\text{-way}, 15\text{-query}$).
* **`09_lee.ipynb`**: Test-time Latent Embedding Exploitation (LEE) with preserved source-population cosine regularizer ($\lambda=0.5$).
* **`10_maml.ipynb`**: First-Order MAML (FOMAML) episodic meta-training ($3$ inner gradient steps with $\alpha=0.005$, outer Adam $\beta=5 \times 10^{-4}$).
* **`11_supervised_contrastive_fsl.ipynb`**: Non-episodic Supervised Contrastive Learning (SupCon) with intra-class spread penalty ($\tau=0.07, \alpha_\text{spread}=0.5$).

### Phase 4: Benchmark Evaluation & Result Analysis
* **`12_generalization_evaluation.ipynb`**: Cross-subject zero-shot generalization evaluation on unobserved test subjects across $K \in \{1, 5, 10\}$ shots.
* **`13_personalization_evaluation.ipynb`**: Subject-specific few-shot calibration evaluation, computing personalization accuracy and absolute gains ($\Delta\%$).
* **`14_experiment_runner.ipynb`**: Automated batch runner executing all $5 \times 3 \times 4 = 60$ dataset-backbone-method experimental configurations.
* **`15_results_analysis.ipynb`**: Generates all LaTeX tables (Tables I–IX) and publication figures (Figures 1–7), including the Personalization Ceiling Effect correlation regression ($r \approx -0.81$).

---

## Notes on Reproducibility

1. **Exact Reproduction**: To reproduce the published paper results exactly, keep all random seeds fixed to `42` (as defined in `00_config_and_utils.ipynb`) and run all notebooks sequentially.
2. **Subject-Independent Integrity**: All preprocessing statistics (mean and standard deviation for Z-score standardization) are computed strictly on the training partition of each dataset to prevent target subject data leakage.
3. **Hardware Runtime**: Pre-training and few-shot evaluation across all five datasets require approximately 4–6 hours on a single NVIDIA T4 GPU.

---

## Acknowledgments

This research was partially supported by the Higher Education Commission (HEC), Pakistan, under the project *"HumCare: Human Activity Analysis in Healthcare"* (Grant No. 15041).

---

## Citation

If you find this codebase or paper useful in your research, please cite:

```bibtex
@article{fsl_har_ceiling_effect,
  title   = {The Ceiling Effect in Few-Shot Personalization: How Baseline Accuracy Shapes the Limits of User Adaptation in Wearable Activity Recognition},
  author  = {Amjad, Rabdaa and Khan, Muhammad Hassan and Nisar, Muhammad Adeel},
  journal = {Multimedia Tools and Applications},
  year    = {2026}
}
```
