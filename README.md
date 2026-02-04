# Max Healthcare Assessment - Solution

**Author:** Aman Verma

---

## Project Overview

This repository contains the complete solution for the Max Healthcare assessment. It implements a robust machine learning pipeline to handle noisy labels in medical imaging classification.

### Key Features
- **Comprehensive EDA**: Analysis of class imbalance, pixel distributions, and sample visualization.
- **Multiple Strategies**: Comparison of 4 distinct approaches:
  1. **Baseline**: Weighted Cross-Entropy to handle class imbalance.
  2. **Label Smoothing**: To mitigate overfitting to noisy labels.
  3. **Loss-Based Filtering**: Dynamically removing likely mislabeled samples during training.
  4. **Modern Architecture**: ResNet-18 for superior feature extraction.
- **Live Inference**: Production-ready script (`live_inference.py`) for real-time evaluation.

---

## Quick Start

### 1. Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install numpy matplotlib seaborn torch torchvision Pillow pandas tqdm
```

### 2. Live Inference (On-Campus Evaluation)
To run the inference script on a new dataset (e.g., hidden test set):

```bash
python live_inference.py --model_path best_model_submission.pth --data_path <path_to_test_data.npz>
```

**Expected Output:**
```
============================================================
LIVE INFERENCE - PlaceComm'26 Assessment
============================================================
loading data...
INFERENCE COMPLETE
Accuracy: XX.XX%
============================================================
```

### 3. Training & Reproduction
To reproduce the training results and generate all plots:

1. Open the notebook:
   ```bash
   jupyter notebook Assessment_Complete_Solution.ipynb
   ```
2. Run all cells (`Cell` -> `Run All`).
3. Outputs will be saved in the current directory:
   - `best_model_submission.pth`: The final selected model.
   - `comparison_results.png`: Performance chart of all strategies.
   - `training_curves.png`: Loss and accuracy curves.

---

##  Methodology Comparison

| Strategy | Description | Why it works / Results |
|----------|-------------|------------------------|
| **Baseline** | Simple CNN + Weighted CE Loss | Establishes a lower bound; handles imbalance well. |
| **Label Smoothing** | Targets = 0.95/0.05 instead of 1/0 | Prevents the model from being "too confident" on noisy labels. |
| **Filtering** | Drop high-loss samples after warmup | Explicitly removes data that looks like "noise". |
| **ResNet-18** | Deep Residual Network | Best performance due to advanced feature learning capabilities. |

**Selected Model for Submission:** `best_model_submission.pth` (ResNet-18)

---

##  Repository Structure

- `Assessment_Complete_Solution.ipynb`: Main notebook code, analysis, and training.
- `live_inference.py`: Standalone script for testing/grading.
- `best_model_*.pth`: Saved model weights for each strategy.
- `*.png`: Generated visualizations and plots.
- `candidate_dataset (1).npz`: Training dataset.

---

