# Probability-Guided Contrastive Learning for Long-Tailed Domain Generalization (PgCL)

This repository contains the official implementation of the paper **"Probability-Guided Contrastive Learning for Long-Tailed Domain Generalization"**. The code and instructions provided here allow you to reproduce the experiments and results presented in the paper.

## Abstract

Domain Generalization (DG) aims to improve the generalization of models trained on specific source domains to perform well on unseen target domains. Contrastive Learning (CL) is one method to achieve this by leveraging the semantic relationships between sample pairs from different domains to learn domain-invariant representations. However, domain generalization cannot be achieved solely by applying contrastive-based methods. 

In this paper, we propose a novel technique called **Probability-Guided Contrastive Learning (PgCL)**, which selects contrastive pairs based on an estimation of the data distribution of samples from each class in the feature space. We introduce a simple assumption that normalized features in contrastive learning are sampled from an infinite number of contrastive pairs and modeled using a mixture of von Mises-Fisher (vMF) distributions on unit space in domain generalization. From this, we derive a closed-form of the expected contrastive loss for efficient optimization. 

We empirically study the error bound of PgCL and demonstrate its superior performance compared to state-of-the-art methods on several domain generalization datasets.

## Keywords

- Domain Generalization
- Probability-Guided Contrastive Learning
- Contrastive Pairs
- von Mises-Fisher Distributions

## Introduction

In many applications, deep neural networks (DNNs) have demonstrated remarkable effectiveness under the assumption that training and test data are identically distributed and independent. However, in real-world scenarios, this assumption is often violated, leading to poor performance when DNNs are tested on out-of-distribution (OOD) target data. This phenomenon, known as domain shift, hinders the generalization ability of DNNs.

Domain Generalization (DG) addresses this issue by leveraging the diversity of source domains to enhance model generalization. However, the presence of long-tail distributions in real-world data, where some classes have significantly fewer samples than others, poses an additional challenge. Contrastive Learning (CL) offers a promising solution by optimizing a distance metric that brings positive pairs closer together and pushes negative pairs apart, thereby learning generalized features across different domains.

Nevertheless, CL has limitations, particularly in DG scenarios, where a large batch size is required to generate enough contrastive pairs, leading to high computational and memory demands. Additionally, long-tailed data often results in a bias towards head classes, negatively impacting the performance on minority classes.

## Method

### Probability-Guided Contrastive Learning (PgCL)

To address these challenges, we propose PgCL, a novel method that:

1. **Estimates the Data Distribution**: We estimate the data distribution of samples from each class in the feature space using a mixture of von Mises-Fisher (vMF) distributions.
2. **Selects Contrastive Pairs**: Contrastive pairs are selected based on the probability of belonging to different classes, ensuring a more balanced representation of classes in the contrastive learning process.
3. **Optimizes Efficiently**: We derive a closed-form of the expected contrastive loss, enabling efficient optimization even with long-tailed data distributions.

## Results

PgCL demonstrates significant improvements over state-of-the-art methods in various domain generalization benchmarks, particularly in handling long-tailed data distributions.

```python
bash run.sh
```

or

```python
python train.py
```