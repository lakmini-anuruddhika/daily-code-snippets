# Seeds Clustering Assignment

This repository contains the implementation of K-means and K-medoids clustering on the **UCI Seeds Dataset**.

## Contents

- `Seeds_Clustering.ipynb` : Jupyter notebook with all code and analysis
- `seeds_dataset.txt` : Dataset used for clustering (https://archive.ics.uci.edu/ml/datasets/seeds)

## Description

- The notebook performs:
  - Data standardization
  - Elbow method to determine optimal k for K-means
  - K-means clustering and evaluation (Silhouette, Purity)
  - K-medoids clustering and evaluation (Silhouette, Purity)
  - Comparison of K-means vs K-medoids