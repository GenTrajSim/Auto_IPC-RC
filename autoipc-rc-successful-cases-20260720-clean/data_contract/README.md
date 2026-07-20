# Data Contract

`frames.jsonl.gz` is a compressed manifest snapshot for the successful cases. It contains 290,635 frame records and references the original frame files.

```bash
gzip -dk frames.jsonl.gz
export MANIFEST_JSONL="$PWD/frames.jsonl"
export NORMALIZATION_JSON="$PWD/normalization.json"
```

The original frame files are not duplicated in this release. Keep their paths available or regenerate the manifest for the target workstation.
