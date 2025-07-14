# Auto_IPC-RC
An atuoencoder architecture for mining reaction coordinate (or implicit physical characteristic )
 - DATASET PATH
 ```text
├── Auto_IPC-RC/
└── dp_LDL/
 ```
The training data "dp_LDL" (~8 GB, for tip4p/ice P-T-rho-potential dataset) can be downloaded from the link within the article.
new_coord  
xx.npy shape: (300,30,444)
box
xx.npy shape: (9,) - (time, boxx, boxy, boxz, temp, press, rho, pot, ent, pb) - can read from boxdata.csv.
