# Regression control

A fixed, known control trace that guards against unintended changes to analysis
results. Run it after any code change: it re-analyzes the control shot, compares the
outputs against a stored baseline, and appends a timestamped entry to the version
history.

## Control trace

`C1--20251022--00001` (Brass) — run **SPADE-only** on the bundled velocity file
`data/C1--20251022--00001--vel-smooth-with-uncert.csv`, so the check is fully
deterministic (no ALPSS re-extraction, whose STFT/spline stages can vary across
numpy/scipy versions and platforms). Its spall strength, `2.892868603294747 GPa`, was
independently reproduced bit-for-bit.

## Files

| File | Purpose |
|---|---|
| `run_control.py` | Runs the control and compares vs baseline; appends to history. |
| `control_config.yml` | Frozen SPADE-only config (snapshot). Kept fixed so the control isolates **code** changes; `@DATA@/@PARAMS@/@OUTPUT@` are resolved at run time. |
| `baseline.json` | The golden expected values (metrics + the commit that set them). |
| `history.jsonl` | Append-only, one JSON line per run: timestamp, git commit, PASS/FAIL, and the measured values — a timestamped version history. |
| `data/` | The control velocity trace, its noise fraction, and a one-row param fixture (maps the shot to Brass). |

## Usage

```bash
# after a code change — compare against the baseline (exit 0 = PASS, 1 = FAIL)
helix_toolbox_env/bin/python3 regression/run_control.py

# intentionally accepted a change? re-set the baseline (old one stays in history)
helix_toolbox_env/bin/python3 regression/run_control.py --update-baseline
```

Options: `--rtol` (default 1e-6), `--atol` (default 1e-9), `--keep-output`.

## Workflow

1. Make code changes.
2. Run `run_control.py`. A PASS means every tracked metric (spall, HEL, peak stress,
   rise times, plastic/compressive strain rates, shock-front width, …) is within
   tolerance of the baseline.
3. A **DRIFT/FAIL** means a result moved. If unintended, investigate. If the change is
   correct and reviewed, run with `--update-baseline` — the previous values remain in
   `history.jsonl`, so the full timeline of how the control evolved is preserved.

The exit code (0 PASS / 1 FAIL) lets this gate a pre-commit hook or CI job.

## MPDV input compatibility checks

Run `QT_QPA_PLATFORM=offscreen helix_toolbox_env/bin/python3 regression/test_mpdv.py`
for central-probe input/metadata filtering and GUI configuration tests. Fixtures
are generated in temporary directories; the original experimental logs are not
modified. This suite complements the scientific control above.

The pre-MPDV working-tree control on 2026-09-18 already differed from the stored
baseline in HEL strength, HEL strain rate, and HEL uncertainty. The MPDV change
must preserve that pre-edit result as well as raw Single PDV extraction. Do not
reset the scientific baseline merely to accept these pre-existing differences.

## File compatibility suite

Run all input, metadata, GUI, and availability regressions with:

```bash
QT_QPA_PLATFORM=offscreen helix_toolbox_env/bin/python3 -m unittest discover -s regression -p 'test_*.py'
```

Coverage includes configured two-column scope exports, LeCroy `Time,Ampl`
headers, combined `Time Tags (...)` exports selecting labeled Channel 1,
reordered channel columns, UTF-8 BOM/CRLF/whitespace, sample offsets, CSV/Excel
metadata, shared filenames across probe fields, mixed single/central-probe
selection, batch selection, saved velocity inputs, and missing/cloud-only inputs.
The bundled raw scope fixture is compared directly against the previous reader.
Combined-format tests use synthetic values, not research data.

Multiplexed MPDV demultiplexing remains pending. These checks do not promise
support for arbitrary unlabeled multichannel tables or binary scope formats.
