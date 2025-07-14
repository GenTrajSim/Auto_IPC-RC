# Auto_IPC-RC
An atuoencoder architecture for mining reaction coordinate (or implicit physical characteristic )

## TRAINING PROCESS
In the ./Linear_*$alpha*/slop_*$phi*/ Path (i.e., Auto_IPC-RC/Linear_0.3/slope_455/), the "dp_LDL_simple_fhi47_100_linear.py" is the training main program. 
- *$alpha* and *$phi* Setting
```bash
###  $alpha setting ###
loss_correlation = (tf.reduce_mean((correlation-0.3)**2) ) #line 555 
loss_spearman_cor = (tf.reduce_mean((spearman_cor-0.3)**2) ) #line 556  
```
```bash
###  $phi setting ###
aa_loss1 = tf.reduce_mean( (k_cor - tf.math.tan(455.*pi/1000.0))**2 ) #line 564
```
- Executive command
```bash
python3 dp_LDL_simple_fhi47_100_linear.py
```
- Training output
In the ./Linear_*$alpha*/slop_*$phi*/log_simple_fhi47_100_linear Path, "test.log", "train.log" and "xe.log" are the outputs. And using "draw.py" can obtain the training process.

## TESTING 
In the ./Linear_*$alpha*/slop_*$phi*/ Path (i.e., Auto_IPC-RC/Linear_0.4/slope_490/), the "dp_LDL_simple_fhi47_100_linear_test_auto.py" is the testing main program. 
- Executive command
```bash
python dp_LDL_simple_fhi47_100_linear_test_auto.py 1800_188 0
```
Also can execute the auto_PT.pl or auto_PT2.pl in Auto_IPC-RC, e.g.,
```bash
perl auto_PT.pl
```
- Testing output
In the ./Linear_*$alpha*/slop_*$phi*/logtest/ Path, "xe.log" or "xe_1800_188.log"/"xe_1800_188.txt" (rename by auto_PT/auto_PT2.pl) are the outputs.

## DATA SET
 - DATASET PATH
 ```text
├── Auto_IPC-RC/
└── dp_LDL/
 ```
The training data "*dp_LDL*" (~8 GB, for tip4p/ice *P-T-rho-potential* dataset) can be downloaded from the link within the article.
 - DATASET FORMAT
1. In the "new_coord" Path, each *$xx*.npy (i.e., 1.npy, 2.npy, ...) file represents the $xx step (i.e., 1.npy, 2.npy, ...) during the MD simulation.

$xx.npy shape: (300,30,4), where "300" is the total number of water molecules, "30" is the max number of water molecules (oxygen atoms) in a local structure, and "4" refers to (*s(r)<sup>^</sup>, x<sup>^</sup>, y<sup>^</sup>, z<sup>^</sup>*)-the detail can read [End-to-end Symmetry Preserving Inter-atomic Potential Energy Model for Finite and Extended Systems](https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html)

2. In the "box" Path, each *$xx*.npy (i.e., 1.npy, 2.npy, ...) file also represents the $xx step (i.e., 1.npy, 2.npy, ...) during the MD simulation. And these files correspond one by one to the files in the coord path.

*$xx*.npy shape: (9,) - (*time, boxx, boxy, boxz, temp, press, rho, pot, ent, pb*) for a configuration from *$xx* step - can read from boxdata.csv. Herein, our model only learn the *rho* and *pot* information for a configuration from *$xx* step.
