# Week 1: Autoencoder for Gene Expression (LINCS L1000)

This week focuses on learning how to implement and train an autoencoder on biological gene expression data from the LINCS L1000 dataset (GSE92742).

## Contents

- `torch_autoencoder.ipynb`: PyTorch implementation with a manual training loop
- `lightning_autoencoder.ipynb`: PyTorch Lightning version with simplified training
- `data_loader.py`: Utility script to download and preprocess LINCS L1000 gene expression data
- `README.md`: This file

## Dataset

The LINCS L1000 dataset contains gene expression profiles from perturbation experiments (drugs, gene knockdowns). This project uses the GEO accession GSE92742 for Level 3 data.

## How to Use

1. Install dependencies listed in `requirements.txt`
2. Run the notebook(s) to train and evaluate the autoencoder
3. Visualize latent space with UMAP included in notebooks

## Dependencies

- torch
- pytorch-lightning
- umap-learn
- pandas
- scikit-learn
- matplotlib
- numpy

## References

- LINCS Project: https://lincsproject.org/
- GEO GSE92742: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742

