# Original Training Commands

Audit date 2026-07-02

No command in this document was executed.

## Reference training script

The supplied shell script sets TRAIN ROOT to two comma separated placeholder roots for BalanceDataset and Dataset. It then invokes Python unbuffered on train.py with paper run v8 config JSON enables preload and writes under runs paper run v8.

Equivalent intended arguments are

python -u train.py --config paper run v8 config.json --train-root BalanceDataset comma Dataset --preload --out-dir runs paper run v8

Important limitation. Comma separated roots are handled only in preload mode. Preload reads the complete datasets into RAM. The JSON requests 500 epochs and batch 250. This command must not run until blockers are resolved and the user explicitly approves training.

## Reference evaluation script

The supplied script invokes compare dist P1800 T188.py with run v8 a single P1800 T188 data1 directory two external reference text files and an output directory. It sets a noninteractive plotting backend.

This does not yet prove use of every file under P1800 T188 if more data directories exist. Both external reference files are unresolved. Evaluation output directory creation also requires user approval when run as a long task.

## Reference prediction script

The supplied script invokes predict.py with the JSON config an existing epoch 500 weights file one placeholder descriptor NPY input and one prediction NPZ output.

It must not overwrite an existing output or checkpoint.

## Required future sequence

1. Resolve all questions in blocking questions.
2. Add fail fast data manifest validation and focused unit tests.
3. Run only short model construction and loss tests after approval where required.
4. Select a new empty run directory and record configuration data manifests statistics seed environment and code revision.
5. Request explicit approval for the full training command.
6. Test on all approved P1800 T188 files only after training and report the controlled validation gates.

## Approval boundary

Package installation training checkpoint creation or overwrite long jobs deletion PLUMED GROMACS and WTMetaD remain prohibited without explicit user confirmation.
