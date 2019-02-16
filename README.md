# PyTorch GDN

## Generalized divisive normalization layer

This utility provides a PyTorch implementation of the GDN non-linearity based on the papers: 


"Density Modeling of Images using a Generalized Normalization Transformation"


Johannes Ballé, Valero Laparra, Eero P. Simoncelli


https://arxiv.org/abs/1511.06281


"End-to-end Optimized Image Compression"


Johannes Ballé, Valero Laparra, Eero P. Simoncelli


https://arxiv.org/abs/1611.01704

## Usage

The GDN layer can be used as a normal non-linearity in PyTorch but must be instantiated with the number of channels at the application and the torch device where it will be used:

```
device = torch.device('cuda')
n_ch = 8

gdn = GDN(n_ch, device)

input = torch.randn(1, 8, 32, 32).to(device)
output = gdn(input)
```

Other parameters that can be used with the GDN are:


```
gdn = GDN(8, device
          beta_min=1-e6,
          gamma_init=.1,
          reparam_offset=2**-18
)
```

