# Exact frame reuse: GPU telemetry audit

Research artifact for **Auditing Energy Claims for Exact Frame Reuse on a Laptop GPU**, prepared for GREENS 2027 Emerging Research. Revision prepared on 7 November 2026. This is a submission manuscript; acceptance is not claimed.

**Authors:** Ishan Teredesai (`001@ddm.in`) and Tushar Rahangadle (`002@ddm.in`), Department of Information Technology, Yeshwantrao Chavan College of Engineering, Nagpur, India.

## Downloads

- [Paper PDF](GREENS_2027_Emerging_Research.pdf)
- [Complete LaTeX project](GREENS_2027_LaTeX.zip)
- [Main LaTeX source](GREENS_2027_Emerging_Research.tex) — use the complete project to compile its table and figure dependencies.
- [Full source, data and reproducibility artifact](GREENS_2027_Source_and_Artifact.zip)
- [SHA-256 checksums](SHA256SUMS.txt)

The full artifact is distributed as a ZIP. Download and extract it to access the benchmark source, both raw CSVs, generated trial tables, protocols, figures, audit reports and build scripts. The large acquisition CSVs are inside that archive.

## What the study tests

An implemented CUDA renderer compares full rendering, GPU validation, application-known dirty tiles and compact dirty-tile dispatch. Explicit dependencies make invalidation inspectable and allow exact image comparisons. The controlled workload is used to audit timing and energy telemetry; it is not a production game-engine benchmark.

The completed follow-up contains 144 trials in four fresh process sessions, 204,297 measured iterations and 1,788 correctness comparisons with zero mismatches. All 144 final-image checks pass. Cumulative energy and integrated power disagree in sign for 10 of 36 main-policy contrasts across 12 blocks. The separate, interrupted pilot contains 88 completed trials and 23 disagreements among 48 contrasts; it is retained as context rather than pooled with the follow-up.

The polling sweep includes 52 trials, with 24 idle/copy controls, eight A/A trials, eight explicit-power-field trials and four sustained-load trials alongside 48 main comparisons. The paper explains the acquisition-sensitive counter behavior and the limits of both estimators.

## Reproduce the reported analysis

After extracting the full source-and-artifact ZIP, open a terminal in that extracted directory. Use Python 3.12 for the pinned analysis dependencies:

```sh
python -m pip install -r requirements.txt
python build.py --analysis-only
```

This uses the existing CSVs and requires no GPU. It verifies input and embedded-source hashes, recomputes energy accounting, regenerates tables and figures, and checks the recorded plans and paced traces. The artifact's `README.md` includes virtual-environment setup. Its `REVIEWER_START_HERE.md` maps claims to the evidence files.

To rebuild the paper after installing a TeX distribution with pdfLaTeX:

```sh
python build.py
```

Alternatively, import the complete LaTeX ZIP into Overleaf and compile `GREENS_2027_Emerging_Research.tex` with pdfLaTeX.

## Hardware replication and provenance

The artifact's separate acquisition instructions describe restoring the exact program embedded in the CSV, host checks, a smoke test and a new full hardware run. A new acquisition is a separate dataset and is not needed to regenerate the current paper.

The readable Python/CUDA sources contain no comments or docstrings. Their documented formatting changes do not change the executable CUDA tokens or embedded compiled PTX. The exact original source remains recoverable from the unchanged raw CSV; original evidence and third-party notices are preserved. Source identities and the checked derivative are documented in `provenance/readable_source.json` inside the artifact.

## Interpretation and disclosure

The four sessions are from one laptop in one sitting. Individual frames and shared-baseline contrasts are not independent replications. Both energy estimators use exposed NVML telemetry; no external calibration, physical energy-saving guarantee or statistical-equivalence result is claimed.

OpenAI ChatGPT assisted during preparation of the paper, including drafting and revising the abstract and Sections I–VIII and developing the benchmark, analysis code and figures. Numerical results were computed from the recorded acquisition data. The paper includes this disclosure. The unchanged IEEEtran distribution retains its original notices.
