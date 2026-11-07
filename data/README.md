# Acquisition records

| File | Run | Status |
|---|---|---|
| `gpu_audit_v3_20261106_153155_70b1e6.csv.gz` | 6 November 2026 follow-up | 144 of 144 trials completed |
| `gpu_audit_v2_20261102_144917_80abb6.csv.gz` | 2 November 2026 pilot | 88 of 306 trials completed before interruption |

These gzip files preserve the original CSV bytes. `datasets.json` gives the compressed size, original size and SHA-256 digest of each decompressed record. Run `python reproduce.py` from the repository root to decompress and verify them automatically.

Each CSV is an acquisition log with multiple record types, identified by `record_type`. The log includes metadata, the fixed protocol, per-trial plans, frame timing, telemetry queries, energy boundaries, correctness checks and run/session status. Fields that do not apply to a record type are empty. Structured per-record payloads are held in the `details` column.

The original executed Python program is stored in the metadata as a compressed payload with its checksum. No original records, including incomplete pilot records, have been removed. See `restore_executed.py` for source recovery and `analyze.py` for record handling and units.
