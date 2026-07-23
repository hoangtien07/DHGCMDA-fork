# DHGCMDA: Dual-View Heterogeneous Graph Contrastive Learning for miRNA-Disease Association Prediction

> **Active status:** read [docs/status/PROJECT_STATUS.md](docs/status/PROJECT_STATUS.md) before
> interpreting results or launching an experiment. Historical `M_GSM` is GIP (association-derived),
> not a real sequence view; legacy `0.697` results are not leakage-controlled headlines.

A deep learning framework for predicting miRNA-disease associations using dual-view heterogeneous graph neural networks with cross-modal contrastive learning.

## Highlights

- **Dual-View Architecture**: Leverages both sequence and functional views for miRNAs, gene and semantic views for diseases
- **Cross-Modal Contrastive Learning**: Inter-view contrastive learning between miRNA and disease representations
- **Multi-Type Association Prediction**: Predicts 4 biological association types (Circulation, Epigenetics, Target, Genetics)
- **Hypergraph Neural Networks**: Captures high-order relationships using KNN and K-means based hypergraph construction
- **Focal Loss**: Addresses class imbalance in association types
- **Attention Mechanism**: Multi-head attention for feature fusion across different views

## Architecture

The model consists of three main components:

1. **Dual-View Feature Extraction**
   - miRNA View 1: historical GIP similarity (m_ss; not sequence-derived)
   - miRNA View 2: Functional similarity (mi_fun)
   - Disease View 1: Gene similarity (d_gs)
   - Disease View 2: Semantic similarity (dis_sem)

2. **Heterogeneous Graph Neural Network**
   - Hypergraph construction using KNN and K-means
   - Multi-layer graph convolution with attention
   - Cross-modal message passing

3. **Contrastive Learning Framework**
   - Intra-view contrastive learning within each modality
   - Inter-view contrastive learning between miRNA and disease
   - Multi-type association prediction with Bayesian priors

## Requirements

```
python>=3.8
torch>=1.10.0
torch-geometric>=2.0.0
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
scipy>=1.6.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CDMBlab/DHGCMDA.git
cd DHGCMDA
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install numpy pandas scikit-learn scipy
```

## Dataset

### Dataset Structure

Place your data in the `v2.0_495m383D` directory with the following files:

```
v2.0_495m383D/
├── D_SSM1.txt                                           # Disease semantic similarity
├── D_SSM2.txt                                           # Disease gene similarity  
├── M_FSM.txt                                            # miRNA functional similarity
├── M_GSM.txt                                            # miRNA sequence similarity
└── multi_all_mirna_disease_pairs_without_negative.csv   # Association pairs
```

### Data Format

- **Similarity matrices**: Space or tab-separated `.txt` files (NxN matrices)
- **Association file**: CSV format with columns: `[miRNA_id, disease_id, association_type]`
  - Association types: 1=Circulation, 2=Epigenetics, 3=Target, 4=Genetics

### Dataset Statistics

- **miRNAs**: 495
- **Diseases**: 383
- **Associations**: ~1,498
- **Association Types**: 4 (Circulation: 24.5%, Epigenetics: 10.5%, Target: 19.6%, Genetics: 45.5%)

## Usage

### Quick Start

1. **Verify data loading**:
```bash
python check_v2_0_495m383D.py
```

2. **Analyze class distribution and generate recommended weights**:
```bash
python wieghts.py
```

3. **Train the model**:
```bash
python main_experiments_hetero1.py
```

### Custom Configuration

Modify parameters in `param.py` or pass command-line arguments:

```bash
python main_experiments_hetero1.py \
    --epoch 650 \
    --lr 0.0001 \
    --nlayer 2 \
    --n_head 8 \
    --dropout 0.3 \
    --validation 5 \
    --enable_inter_view_cl True \
    --inter_view_weight 0.3
```

### Advanced Usage

**Custom hypergraph construction**:
```bash
python main_experiments_hetero1.py \
    --K_neigs 13 15 \
    --clusters 9 11
```

**Focal loss tuning**:
```bash
python main_experiments_hetero1.py \
    --use_focal_loss True \
    --focal_gamma 2.5
```

## Model Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epoch` | 650 | Number of training epochs |
| `lr` | 0.0001 | Learning rate |
| `nlayer` | 2 | Number of HGT layers |
| `n_head` | 8 | Number of attention heads |
| `dropout` | 0.3 | Dropout rate |
| `alpha` | 0.5 | Weight for negative samples |
| `validation` | 5 | Number of CV folds |
| `inter_view_weight` | 0.3 | Weight for inter-view contrastive loss |
| `focal_gamma` | 2.5 | Gamma for focal loss |

### Class Weights

The model uses balanced class weights to prevent class collapse:
- Automatically computed based on inverse class frequency
- Adjustable via `--class_weights` parameter
- Run `python wieghts.py` to analyze your data distribution

## Results

### Performance Metrics

The model is evaluated using:
- **AUC** (Area Under ROC Curve)
- **AUPR** (Area Under Precision-Recall Curve)
- **F1-Score** (Macro and Per-Class)
- **Precision & Recall** (Per-Class)

### Expected Performance

| Metric | 5-Fold CV | Independent Test |
|--------|-----------|------------------|
| AUC | ~0.95 | ~0.93 |
| AUPR | ~0.89 | ~0.87 |
| F1 (Macro) | ~0.62 | ~0.60 |

### Visualization

Training progress and results are saved in:
- `./results/`: Model checkpoints and predictions
- `./logs/`: Training logs and metrics

## Project Structure

```
DHGCMDA/
├── main_experiments_hetero1.py    # Main training script
├── hetero_model.py                 # Model architecture
├── prepareData.py                  # Data preprocessing
├── trainData.py                    # Data loader with dual-view support
├── param.py                        # Hyperparameter configuration
├── Calculate_Metrics.py            # Evaluation metrics
├── check_v2_0_495m383D.py         # Data validation script
├── wieghts.py                      # Class weight analyzer
├── layers.py                       # Neural network layers
├── utils.py                        # Utility functions
├── ConstructHW.py                  # Hypergraph construction wrapper
├── hypergraph_construct_KNN.py     # KNN-based hypergraph
├── hypergraph_construct_kmeans.py  # K-means-based hypergraph
├── create_hetero_data.py          # Heterogeneous graph data structure
├── v2.0_495m383D/                 # Dataset directory
│   ├── D_SSM1.txt
│   ├── D_SSM2.txt
│   ├── M_FSM.txt
│   ├── M_GSM.txt
│   └── multi_all_mirna_disease_pairs_without_negative.csv
├── results/                        # Output directory
└── logs/                          # Log directory
```

## Key Features Explained

### 1. Dual-View Architecture

Each biological entity is represented from multiple perspectives:
- **miRNA**: Sequence-based (GSM) + Functional (FSM)
- **Disease**: Gene-based + Semantic (SSM)

This multi-view approach captures complementary information.

### 2. Cross-Modal Contrastive Learning

The model learns to:
- **Align** positive miRNA-disease pairs in the embedding space
- **Separate** negative pairs with a margin
- **Maximize** mutual information across views

### 3. Multi-Type Association Prediction

Predicts specific biological mechanisms:
1. **Circulation**: miRNAs in body fluids as biomarkers
2. **Epigenetics**: miRNA regulation of gene expression
3. **Target**: Direct miRNA-gene targeting
4. **Genetics**: Genetic variants in miRNA genes

### 4. Hypergraph Neural Network

Captures high-order relationships:
- **KNN**: Local neighborhood structure
- **K-means**: Global cluster structure
- Combines both for comprehensive representation

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size or model dimensions
   - Use gradient accumulation
   - Enable mixed precision training

2. **Class imbalance warnings**
   - Run `python wieghts.py` to analyze distribution
   - Adjust `--focal_gamma` and `--class_weights`

3. **Data loading errors**
   - Verify file paths with `python check_v2_0_495m383D.py`
   - Check file formats match expected structure

4. **Poor performance on minority classes**
   - Increase `--focal_gamma` (e.g., 3.0)
   - Adjust class weights based on `wieghts.py` output

