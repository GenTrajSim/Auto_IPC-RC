# Paper Method Summary

This document summarizes the key method details needed for code refactoring and reproduction.

## Primary reference

Paper:

Evidence for the generic existence of two local structures in liquid water

DOI:

https://doi.org/10.1038/s41567-026-03301-8

## Primary reproduction target

The first reproduction target is main-text Fig. 2d under:

```text
P = 1800 bar
T = 188 K
```

The goal is to reproduce 2D probability density distributions:

```text
P(rho_local_hat, PC)
```

for different alpha and phi settings, especially:

```text
PCI:
alpha = 0.4
phi = 0.490*pi

PCII:
alpha = 0.2
phi = 0.455*pi
```

## Original data format

Original local descriptor shape for one configuration:

```text
[A, 30, 4]
```

where:

```text
A = 300 water molecules / oxygen centers
30 = maximum number of neighboring oxygen atoms
4 = (s(r), x, y, z)
```

Batch shape:

```text
[B, A, 30, 4]
```

## Original model idea

The model is an unsupervised autoencoder.

Encoder predicts:

```text
rho_local_hat_i
u_local_hat_i
```

from local structure descriptors.

System-level predictions:

```text
rho_global_hat = mean_i rho_local_hat_i
U_global_hat = sum_i u_local_hat_i
```

These are fitted to MD system density and potential.

Decoder predicts implicit physical coordinate:

```text
PC_i
```

from local density and local potential branches.

## Loss function

Total loss:

```text
L_total = k_rho * L_rho + k_pot * L_pot + k_alpha * L_alpha + k_phi * L_phi
```

Default paper weights:

```text
k_rho : k_pot : k_alpha : k_phi = 1 : 1 : 100 : 1
```

## Density loss

The model predicts local density for each atom and averages over atoms:

```text
rho_global_hat = mean_i rho_local_hat_i
```

Then fit to normalized MD system density.

## Potential loss

The model predicts local potential for each atom and sums over atoms:

```text
U_global_hat = sum_i u_local_hat_i
```

Then fit to normalized MD system potential.

## Normalization

The paper uses centered range normalization:

```text
x_hat = (x - mean_train) / (max_train - min_train)
```

Do not silently replace this with standard min-max normalization.

## Alpha constraint

Alpha controls Pearson correlation between local density and PC:

```text
alpha = corr(rho_local_hat, PC)
```

## Phi constraint

Phi controls generalized angle / slope of the distribution:

```text
phi = arctan(alpha * ||PC_centered|| / ||rho_centered||)
```

Equivalent slope target:

```text
alpha * ||PC_centered|| / ||rho_centered|| = tan(phi)
```

## Multi-head extension

Original code trains separate models for PCI and PCII.

New model should use:

```text
shared encoder + K independent decoder heads
```

Output:

```text
pc_heads: [K, B, A]
```

Initial heads:

```text
Head 0: PCI, alpha=0.4, phi=0.490*pi
Head 1: PCII, alpha=0.2, phi=0.455*pi
```

## A/B classification

Use plane coordinate:

```text
M_i = c1 * PCII_i + c2 * rho_local_i + c3 * PCI_i
```

A/B rule:

```text
A if M_i > M_cut
B if M_i < M_cut
```

For P1800_T188, supplementary table gives approximate:

```text
c1 = 0.030851
c2 = 1.0
c3 = 0.005987
M_cut = 0.032535
```

## Physical validation

If two-state model is meaningful:

```text
V_system = V_A * x_A + V_B * (1 - x_A)
```

Therefore:

```text
V_system = (V_A - V_B) * x_A + V_B
```

So Vsystem vs xA should be strongly linear.

Expected paper-level examples:

```text
P1800_T188: R2 around 0.9889
P1725_T188: R2 around 0.9858
```

