# DM-GNN: Dual-Stream Multi-Dependency Graph Neural Network

This repository contains a **Python implementation** of **DM-GNN (Dual-stream Multi-dependency Graph Neural Network)** for **cancer survival analysis** based on the paper:

**[Dual-stream Multi-dependency Graph Neural Network Enables Precise Cancer Survival Analysis](https://doi.org/10.1016/j.media.2024.103252)**  
*(Published in Medical Image Analysis, 2024)*  

## Disclaimer
This repository is an independent **reproduction** of the methodology described in the aforementioned paper. The original authors were not involved in this implementation. All credit for the **scientific concepts** goes to the original authors. This implementation is for **educational and research purposes only** and does not claim originality in methodology.

## Overview
DM-GNN is a deep learning framework designed to enhance **cancer survival analysis** using **histopathology whole-slide images (WSIs)**. Unlike existing methods, which struggle to model complex dependencies between diverse tissue patches, DM-GNN:

- Uses a **dual-stream structure** to capture both **morphological affinity** and **global dependencies**.
- Employs **graph neural networks (GNNs)** to process WSIs as **graphs**.
- Leverages **affinity-guided attention recalibration (AARM)** to improve prediction stability.

## Dataset
The WSI dataset used in the paper are sourced from the TCGA, which can be downloaded from the gdc portal https://portal.gdc.cancer.gov, the slide images are ".svs" files.

## Key Features
- **Dual-stream network**: 
  - **Feature Updating Branch (FUB)**: Models morphological similarity.
  - **Global Analysis Branch (GAB)**: Captures co-activating dependencies.
- **Graph-based computation**: Uses **Graph Convolutional Networks (GCNs)** to model relationships between image patches.
- **Weakly Supervised Learning**: Only requires slide-level labels, eliminating the need for detailed annotations.

## Files in this Repository
- `DM_Gnn.py` → **Main Python file** containing the full implementation, including:
  - **WSI processing** (tissue segmentation, patch extraction, feature extraction)
  - **Graph-based survival analysis** using GCNs
  - **Dual-stream learning architecture (FUB & GAB)**
  - **Affinity matrix computation & attention recalibration (AARM)**
  - **Model training, evaluation, and Kaplan-Meier analysis**
- `LICENSE` → **MIT License** file.

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

