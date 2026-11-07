# Source provenance

`readable_source.json` records the original executed source, the inspected benchmark copy, kernel identities and embedded compiled PTX. The original program is recovered from the acquisition log with `restore_executed.py`; its bytes are checked against the checksum recorded during acquisition.

The inspected copy differs in documentation and formatting, including the kernel-source checksum required by its text representation. `verify_artifact.py` compares normalized executable Python structure, CUDA executable tokens and embedded PTX, then regenerates all 140 paced workload traces and checks the 144 trial plans.

`analysis_environment.json` records the environment used to produce the analysis. Acquisition-specific hardware, driver and process metadata remain in the raw CSVs and the derived metadata files under `results/`.
