# Hardware replication

The archived measurements were collected on a Windows laptop with an NVIDIA GeForce RTX 4050 Laptop GPU. Hardware acquisition uses 64-bit Python and the NVIDIA driver. The benchmark uses Python's standard library and the CUDA/NVML interfaces exposed by that driver.

## Restore the executed program

After running `python reproduce.py` from the repository root, restore the exact follow-up program:

```sh
python restore_executed.py data/gpu_audit_v3_20261106_153155_70b1e6.csv executed/gpu_audit_v3.py
python executed/gpu_audit_v3.py --self-test
python executed/gpu_audit_v3.py --plan
```

Restoration checks the embedded checksum and refuses to overwrite an existing destination. It does not execute the recovered program.

## Collect a new run

On a compatible Windows NVIDIA system:

```sh
python executed/gpu_audit_v3.py --smoke --output-dir new_runs
python executed/gpu_audit_v3.py --output-dir new_runs
```

Keep AC power connected. The smoke test checks setup and is not study data. The full protocol contains 144 trials in four fresh processes. The archived run lasted 69.18 minutes; allow the protocol estimate of 75-95 minutes or longer on a different configuration. Preserve incomplete acquisitions and their status records.

A fresh run must be analyzed as a separate dataset. The supplied analysis is pinned to the two archived CSV hashes and will reject a replacement acquisition.

## Inspect the benchmark

`gpu_audit_v3.py` and `kernels.cu` are the readable representations used for inspection. Run `python benchmark/gpu_audit_v3.py --self-test` for host-side checks. This copy was not the file executed during the archived acquisition. Its equivalence checks and source identities are recorded in `provenance/`; use the recovered original for exact-source replication.
