# GPU energy measurement: polling and reuse

Supplementary material for **Polling-Dependent NVML Energy Estimates on a Laptop GPU**.

Ishan Teredesai and Tushar Rahangadle  
Department of Information Technology, Yeshwantrao Chavan College of Engineering, Nagpur, India

This repository contains the CUDA benchmark, acquisition records and analysis needed to reproduce the manuscript's results. The study examines how GPU energy estimates depend on counter polling and collection mode, using exact frame reuse as a software-optimization case. The completed study and interrupted pilot remain separate datasets; only the completed study supplies the revised manuscript's reported results.

## Reproduce the results

Use Python 3.12. Download this repository with **Code > Download ZIP** and extract it, or clone it:

```sh
git clone https://github.com/titcsl/reuse-model.git
cd reuse-model
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\activate` in Windows Command Prompt. Then run:

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

Reproduction requires no GPU or TeX installation. The script decompresses the two acquisition files, verifies their original SHA-256 hashes, recomputes the statistics, generates the tables and figures, and checks the recorded workload plans and source provenance. Allow at least 500 MB of free disk space in addition to the Python environment. Existing files with unexpected input hashes are rejected.

## Repository contents

| Location | Contents |
|---|---|
| [`benchmark/`](benchmark/) | Runnable CUDA benchmark and kernel source |
| [`data/`](data/) | Original acquisition records, losslessly compressed as CSV.gz |
| [`results/`](results/) | Trial-level results, matched comparisons, fixed protocols and verification reports |
| [`figures/`](figures/) | Figures generated from the recorded measurements |
| [`provenance/`](provenance/) | Source identities and analysis environment |
| `analyze.py` | Acquisition parsing, energy accounting and statistical summaries |
| `render_results.py` | Table and figure generation |
| `verify_artifact.py` | Source, protocol, trace and headline-result checks |
| `restore_executed.py` | Recovery of the exact program embedded in an acquisition |
| `reproduce.py` | Entry point for reproducing the complete analysis |

`SHA256SUMS.txt` records the distributed files. Generated raw CSVs, TeX tables and vector figures are excluded from version control.

## Expected checks

| Check | Completed follow-up | Interrupted pilot |
|---|---:|---:|
| Completed trials | 144 / 144 | 88 / 306 |
| Measured iterations | 204,297 | 84,000 |
| Correctness comparisons | 1,788 | 1,548 |
| Final-image checks | 144 | 88 |
| Main-policy contrasts | 36 | 48 |
| Opposite energy-estimate signs | 10 | 23 |

The pilot and follow-up are analyzed separately. The follow-up uses four fresh processes on one laptop in one sitting. It includes polling-frequency sweeps, no-launch and copy controls, A/A comparisons, explicit power-field queries and sustained rendering.

For a new hardware acquisition, follow [`benchmark/README.md`](benchmark/README.md). A new run is a separate dataset; the fixed-data analysis intentionally checks the two archived acquisitions.

## Scope

The benchmark uses a controlled renderer with explicit tile dependencies. The results describe one NVIDIA GeForce RTX 4050 Laptop GPU and driver configuration. Both energy estimates come from NVML telemetry; neither was calibrated against an external meter. Frame-time improvements do not establish physical energy savings.

The inspected benchmark copy and the exact executed program have separate source identities. See [`provenance/README.md`](provenance/README.md) for recovery and verification details.

## Contact

- Ishan Teredesai: teredesaishan@gmail.com
- Tushar Rahangadle: tusharrahangadle@gmail.com
