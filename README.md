# Auto_IPC-RC
An atuoencoder architecture for mining reaction coordinate (or implicit physical characteristic )
 - DATASET PATH
 ```text
├── Auto_IPC-RC/
└── dp_LDL/
 ```
The training data "dp_LDL" (~8 GB, for tip4p/ice P-T-rho-potential dataset) can be downloaded from the link within the article.
 - DATASET FORMAT
1. In "new_coord" Path, each $xx.npy (i.e., 1.npy, 2.npy, ...) file represents the $xx step (i.e., 1.npy, 2.npy, ...) during the MD simulation.

$xx.npy shape: (300,30,4), where "300" is the total number of water molecules, "30" is the max number of water molecules (oxygen atoms) in a local structure, and "4" refers to (x<sup>x<s/up>, y, z, r) 

2. In "box"
xx.npy shape: (9,) - (time, boxx, boxy, boxz, temp, press, rho, pot, ent, pb) - can read from boxdata.csv.
