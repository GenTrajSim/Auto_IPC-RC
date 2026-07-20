# Original Data Format

Audit date 2026-07-02

## Required roots

Training root one is ../../dp LDL/BalanceDataset.
Training root two is ../../dp LDL/Dataset.
Required test condition is ../../dp LDL/Dataset/P1800 T188.
All three directories exist on host xs. P1800 T188 contains data1 with box coord and new coord directories.

## Directory convention

Observed condition directories use pressure and temperature names such as P500 T225. A condition contains one or more data directories. Samples are paired by identical filename under new coord and box. The loader recursively scans every new coord NPY file and keeps only pairs with an existing box NPY file.

## Loaded values

The descriptor is loaded from new coord and cast to float32. Density target is box element index 5. Potential target is box element index 6. The model expects one descriptor shaped 300 by 30 by 4 and a batch shaped B by 300 by 30 by 4. Actual shape and dtype validation across every required sample has not yet been run.

## Preprocessing represented in the reference

The base descriptor has four channels. Channel one is a switched inverse distance. Channels two through four are unit vector components. Neighbors are sorted by distance the center atom is excluded and missing neighbors are zero padded. Optional element one hot channels can extend the feature dimension.

## Normalization

Reference code uses centered min max scaling equal to two times x minus midpoint divided by range. Training min and max therefore map to minus one and plus one. Project controls instead require x minus training mean divided by training range. The authoritative rule must be selected before training and the same saved statistics must be reused for validation test and inference.

## Completeness risks

1. Loader exceptions are swallowed and failed samples are silently omitted.
2. No complete manifest with expected accepted and rejected counts exists yet.
3. Shape consistency box length finite values duplicate paths and target ranges are not proven.
4. The test condition is inside Dataset which is also required in full for training. This is train test leakage unless the test condition is excluded.
5. Using both complete roots with preload may exceed RAM. A streaming loader is likely required but must preserve a deterministic manifest.

## Required preflight before training

Produce a read only manifest for both roots. Validate every pair and fail on any missing corrupt nonfinite or wrong shape sample. Record per condition counts and hashes. Freeze the test manifest. Compute normalization statistics from the approved training manifest only. This preflight is not a training run but has not been executed because the train test overlap decision is blocking.
