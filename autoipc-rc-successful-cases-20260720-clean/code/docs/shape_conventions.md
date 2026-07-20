# Shape Conventions

This project uses the following shape conventions.

## Symbols

```text
B: batch size / number of configurations
A: number of center atoms
N: number of neighbors
D: descriptor dimension
K: number of decoder heads
```

## Original oxygen-only water case

Single configuration:

```text
coord: [A, N, D] = [300, 30, 4]
```

Batch:

```text
coord: [B, A, N, D] = [B, 300, 30, 4]
```

## Descriptor fields

Original descriptor fields:

```text
D = 4 = (s(r), x, y, z)
```

## Encoder outputs

Local density:

```text
rho_local: [B, A]
```

Local potential:

```text
pot_local: [B, A]
```

Global density:

```text
rho_global: [B]
```

Global potential:

```text
pot_global: [B]
```

## Multi-head PC outputs

Per-head PC:

```text
pc_k: [B, A]
```

Stacked PCs:

```text
pc_heads: [K, B, A]
```

By convention:

```text
pc_heads[0] = PCI
pc_heads[1] = PCII
```

## Flattening convention

For neural network dense layers:

```text
coord [B, A, N, D]
-> embedding
-> [B*A, embedding_dim]
```

After local outputs:

```text
[B*A, 1] -> reshape -> [B, A]
```

## Multi-element future convention

For configurable center species:

```text
coord: [B, A_center, N_neighbor, D]
```

where A_center depends on selected center species.

Default water mode:

```text
center species = ["O"]
neighbor species = ["O"] or ["O", "H"]
```

