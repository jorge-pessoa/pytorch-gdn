# PyTorch GDN

## Generalized divisive normalization layer

This utility provides a PyTorch implementation of the GDN non-linearity based on the papers: 


"Density Modeling of Images using a Generalized Normalization Transformation"


Johannes Ballé, Valero Laparra, Eero P. Simoncelli


https://arxiv.org/abs/1511.06281


"End-to-end Optimized Image Compression"


Johannes Ballé, Valero Laparra, Eero P. Simoncelli

## Method

The GDN activation implements the function:

`y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))`



