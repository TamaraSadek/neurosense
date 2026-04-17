# NeuroSense — Tremor Classifier Pipeline

## Project Overview
Parkinson's Disease tremor detection + ON/OFF motor state classification from IMU accelerometer data (i-PROGNOSIS SData, Zenodo 4311175). Reference repo: https://github.com/alpapado/deep_pd

**Approach: fully unsupervised signal processing.** UPDRS is never a training signal — it is used ONLY post-hoc for face validity. Both a threshold pipeline (HC baseline) and a clustering pipeline (GMM) run in parallel so they can be compared.

## Files
- `tremor_pipeline.ipynb` — canonical pipeline notebook (Sections 1–7), actively being built. **Section 1 complete.**
- `neurosense_tremor.ipynb` — earlier prototype (KNN + pseudo-labels). **Deprecated.** Kept only for reference; do not edit.
- `imu_sdata.pickle` — accelerometer data, 81 subjects.
- `subject_metadata.pickle` — 1716 subjects (only 81 overlap with `imu_sdata`). Always key through `imu_sdata.keys()`.

---

## Dataset structure — verified by inspection (Section 1)

### `imu_sdata[subject_id]` is a 4-tuple
For every subject, `len(imu_sdata[sid][0]) == len(imu_sdata[sid][2]) == len(imu_sdata[sid][3])`. Each element of indices [0]/[2]/[3] corresponds to **one phone call** (what the original paper calls a "session").

| Index | Contents | Shape / value |
|-------|----------|---------------|
| `[0]` | Pre-computed FFT per call. List of arrays, one per call | `(N_windows_in_call, 76, 3)` — **DO NOT USE** (opaque 76-bin pre-processed) |
| `[1]` | Per-subject UPDRS summary dict, 8 keys: `updrs16`, `updrs20_right/left`, `updrs21_right/left`, `fpp_ams_tremor`, `fpp_ams_tremor_sev`, `tremor_manual` | — (use `subject_metadata` instead, richer) |
| `[2]` | Per-call hex IDs (32-char MD5-like). **Phone-call UUIDs**, one per call | list of `str` |
| `[3]` | Raw time-domain accelerometer per call. **USE THIS.** | List of arrays each `(N_windows_in_call, 500, 3)`, float64, units ≈ g |

**Terminology:** a "session" in our downstream code = one element of `[3]` = one phone call. Each phone call contains 3–16 windows (5-sec each at 100 Hz).

### Metadata keys (from `subject_metadata[sid]`)
- Demographics: `serial, age, gender_id, education_id, healthstatus_id, usability_id, languagecode, iscontacted, isactive, study_id`
- `healthstatus_id`: **0=PD (49), 1=Healthy+family hx (13), 2=Healthy (19)**. Total 81.
- `med_eval_1`, `med_eval_2` — 55 / 52 UPDRS keys each. **Present for all 81 IMU subjects.** Key we care about:
  - `tremor_manual_2_2_2020_new` — binary per-patient tremor ground truth (0/1). Present for **80/81** subjects.

### UPDRS tremor label distribution (crosstab)

|                  | tremor=0 | tremor=1 |
|------------------|----------|----------|
| PD (0)           | 25       | 23       |
| HC+fam hx (1)    | 13       | 0        |
| HC (2)           | 18       | **1**    |

### Session / window stats

| Metric | min | p25 | median | p75 | max |
|--------|-----|-----|--------|-----|-----|
| Sessions per subject | 1 | 7 | 26 | 106 | **1753** |
| Windows per session | 3 | 4 | 6 | 9 | 16 |
| Total windows per subject | 4 | 52 | 175 | 670 | 12,122 |

Total windows: **57,010**. No session has <3 windows on this dataset.

---

## Subject exclusions (decided — apply at top of Section 2)

Build `exclude_subjects` = set of subject IDs to drop BEFORE feature extraction. Filter `subject_ids` through it and propagate downstream.

1. **Subject(s) with <5 total windows across all sessions** — matches original paper's `min_windows=5`. Identify dynamically (do not hardcode). Expected: 1 subject.
2. **HC subject with `tremor_manual_2_2_2020_new == 1`** — ambiguous label (healthy by `healthstatus_id` but flagged tremor+); would contaminate both the HC baseline AND the HC validation bucket. Identify dynamically. Expected: 1 subject.

Print the IDs + reasons so the log is auditable. Expected final count: **79 subjects kept, 2 dropped.**

---

## Pipeline plan — Sections 2 through 7

Every section MUST:
1. Run to completion in-notebook (notebook is authoritative, not scripts).
2. Print enough output that the user can verify without running the code themselves.
3. End with a short markdown "Findings / what I'd flag" cell (user's voice: what surprised you, assumptions, uncertainties, what you'd do differently).
4. Be self-contained — do not reference future sections' variables.
5. Persist its output DataFrame so later sections can reload it without re-running upstream (save as pickle alongside the notebook).

### Section 2 — Per-window tremor index
**Constants at top (put them here, not scattered):**
```
FS = 100
WIN_SAMPLES = 500
TREMOR_LO, TREMOR_HI = 4.0, 6.0
BROAD_LO,  BROAD_HI  = 3.0, 15.0
ENTROPY_LO, ENTROPY_HI = 3.0, 8.0
HP_CUTOFF = 1.0
WELCH_NPERSEG = 256
```

**Per-window pipeline:**
- `mag = sqrt(sum(window**2, axis=1))` — orientation-invariant magnitude
- `sos = butter(4, 1.0, 'high', fs=100, output='sos')` → `mag_hp = sosfilt(sos, mag)` (removes gravity)
- `freqs, psd = signal.welch(mag_hp, fs=100, nperseg=256)` — more stable than raw FFT for 500 samples
- `tremor_power = np.trapezoid(psd[4–6 Hz mask], freqs[4–6 Hz mask])`  ← **`np.trapezoid`, not `np.trapz`** (numpy 2.x removed trapz; the old notebook errored on this)
- `dominant_freq` = freq with max PSD in 3–7 Hz
- `in_pd_band = float(4.0 <= dominant_freq <= 6.0)`
- `psd_entropy_band = psd[3–8 Hz]; psd_norm = psd_entropy_band / (psd_entropy_band.sum() + 1e-10)`
- `spec_entropy = entropy(psd_norm + 1e-10) / log(n_bins)` → normalized to [0,1]
- `total_power = np.trapezoid(psd[3–15 Hz], ...)`
- `snr = tremor_power / (total_power + 1e-10)`
- `tremor_index = tremor_power * snr * (1.0 - spec_entropy)`

**Before batch-running everything** — one-window sanity plot:
- Pick one window from a PD subject with `tremor_manual_2_2_2020_new==1` AND high `sum_part_3`.
- Pick one window from any HC subject (post-exclusion).
- Plot side-by-side: (a) magnitude time series, (b) PSD with 4–6 Hz band shaded, (c) table of the 6 features.
- Output whether the expected 4–6 Hz peak shows up in the PD window and NOT in the HC window.

**Batch run** on all non-excluded subjects. Build flat df and pickle as `df_windows.pkl`:
```
subject_id, session_idx, window_idx, health_status, tremor_manual_label,
tremor_power, dominant_freq, spectral_entropy, snr, tremor_index, in_pd_band
```
This will be ~55k rows. Use a flat list-of-dicts accumulator, then `pd.DataFrame(...)` once at the end (much faster than `df.append`).

**Diagnostic plot:** `tremor_index` distribution PD vs HC — log-scale x-axis histogram with two colors overlaid (alpha=0.5). State the separation verbally.

### Section 3 — Per-session aggregation
- Group `df_windows` by `(subject_id, session_idx)`. Aggregate:
  - `mean_tremor_index, max_tremor_index, pct_windows_in_pd_band` (= mean of `in_pd_band`), `mean_snr, mean_spectral_entropy, n_windows`
- Merge back `health_status` and `tremor_manual_label` (first).
- Confidence tier: `n_windows < 3 → 'insufficient'` (unreachable but keep), `3–5 → 'low'`, `>=6 → 'high'`.
- Save as `df_sessions.pkl`.
- Plot `mean_tremor_index` PD vs HC at session level. Compare qualitatively to the window-level plot from Section 2.

### Section 4 — Severity labels, two paths

**Path A — threshold (HC baseline):**
```
hc_sess = df_sessions[df_sessions.health_status != 0]
hc_75 = np.percentile(hc_sess.mean_tremor_index, 75)

pd_above = df_sessions.query("health_status == 0 and mean_tremor_index > @hc_75").mean_tremor_index
mild_thr     = np.percentile(pd_above, 33)
moderate_thr = np.percentile(pd_above, 66)

def label_threshold(x):
    if x <= hc_75: return 'None'
    if x <= mild_thr: return 'Mild'
    if x <= moderate_thr: return 'Moderate'
    return 'Severe'

df_sessions['severity_label_threshold'] = df_sessions.mean_tremor_index.map(label_threshold)
```
Print: `hc_75, mild_thr, moderate_thr`, and label counts.

**Path B — GMM clustering:**
```
feats = ['mean_tremor_index','max_tremor_index','pct_windows_in_pd_band','mean_snr','mean_spectral_entropy']
X = StandardScaler().fit_transform(df_sessions[feats].values)
gmm = GaussianMixture(n_components=4, random_state=42, covariance_type='full').fit(X)
cluster = gmm.predict(X)

# order clusters by mean tremor_index ascending → None..Severe
order = np.argsort([df_sessions.mean_tremor_index.values[cluster==k].mean() for k in range(4)])
cluster_to_label = {order[i]: ['None','Mild','Moderate','Severe'][i] for i in range(4)}
df_sessions['severity_label_cluster'] = [cluster_to_label[c] for c in cluster]
```
(Note: 5 features, NOT 6 — skip `log(n_windows)` so session length doesn't dominate clustering.)

**Comparison:**
- Print per-label counts for both paths.
- `pd.crosstab(df_sessions.severity_label_threshold, df_sessions.severity_label_cluster, margins=True)` — raw + row-normalized.
- Print 5 sessions with biggest disagreement (map severities to 0..3 and `abs(diff)`).
- Recommend which path to trust, with reasoning.

Save as `df_sessions.pkl` (overwrite with new columns).

### Section 5 — Simulated timeline
```
SIMULATED_TIMES = ['08:00','10:00','12:00','14:00','16:00']
```
- For each subject, sort sessions by `session_idx`. `day = i // 5 + 1`, `slot = SIMULATED_TIMES[i % 5]`. Store as `day, simulated_time` columns.
- Pick one PD subject whose `total_windows` is closest to the PD median. Plot their full timeline with Path A labels:
  - x = sequential slot (day*5 + slot_idx)
  - y = severity as ordinal (0..3)
  - colored markers: `None=green, Mild=yellow, Moderate=orange, Severe=red`
  - `low` confidence sessions hatched / alpha=0.4
- Print plain-English summary of that subject's day-by-day pattern (e.g., "Day 1 mostly ON with one Moderate at 14:00; Day 2 degrades after lunch...").

### Section 6 — UPDRS validation
- Compute `patient_mean_tremor_index` = `df_sessions.groupby('subject_id').mean_tremor_index.mean()` (mean of session means — equal weight per session).
- Merge `health_status` and `tremor_manual_label`.
- Three groups:
  - HC = all `health_status != 0` (post-exclusion — 31 subjects after dropping the HC-with-tremor=1)
  - PD-tremor-neg: `health_status == 0, tremor_manual_label == 0` (~25)
  - PD-tremor-pos: `health_status == 0, tremor_manual_label == 1` (~23)
- `scipy.stats.mannwhitneyu(pd_pos, pd_neg, alternative='greater')` → report U, p.
- `sklearn.metrics.roc_auc_score(labels, scores)` on PD-pos vs PD-neg.
- Box plot + overlay strip plot, 3 groups. Include n per group on x-tick labels.
- Print final table:
  ```
  Group              n    mean_score   median
  HC                31    ...          ...
  PD tremor-         ...
  PD tremor+         ...
  Mann-Whitney:  U=...  p=...
  AUC (tremor+ vs tremor-):  ...
  ```
- Commentary — be honest. Targets are p<0.05 and AUC>0.65 but report whatever you get. If weak, hypothesize why (small n, single clinic snapshot vs long aggregate, feature sensitivity, heterogeneous PD subtypes).

### Section 7 — Report payload
- Pick one PD subject with `tremor_manual_label==1` and `>=10` sessions (biggest plausible story).
- Map severity → motor state: `None/Mild → 'ON'`, `Moderate/Severe → 'OFF'`.
- Build JSON matching this schema exactly:
  ```json
  {
    "subject_id": "...",
    "health_status": 0,
    "updrs_tremor_label": 1,
    "patient_mean_tremor_index": 0.38,
    "sessions": [
      {"session_idx": 0, "day": 1, "simulated_time": "08:00", "n_windows": 6,
       "mean_tremor_index": 0.43, "max_tremor_index": 0.71, "pct_in_pd_band": 0.5,
       "mean_snr": 0.41, "severity_label_threshold": "Moderate",
       "severity_label_cluster": "Moderate", "motor_state": "OFF", "confidence": "high"}
    ],
    "daily_summaries": [
      {"day": 1, "on_time_pct": 40, "peak_tremor_time": "10:00",
       "severity_distribution": {"None":1,"Mild":1,"Moderate":2,"Severe":1},
       "sessions_with_data": 5, "sessions_insufficient": 0}
    ],
    "validation": {
      "updrs_tremor": 1,
      "patient_score_percentile_vs_hc": 87,
      "patient_score_percentile_vs_pd": 62
    }
  }
  ```
- Print with `json.dumps(payload, indent=2, default=str)`.
- Short commentary: what's missing that a neurologist would want to see?

---

## Dependencies
`numpy, scipy, scikit-learn, pandas, matplotlib, pickle (stdlib)`. **No PyTorch.**

## Environment
Virtual environment: `C:\Users\USAID\OneDrive - American University of Beirut\Desktop\OneDrive - American University of Beirut\Jad Daorah (Student)'s files - Beirut 332\venv`
Use this venv for all installs and when running any code in this project.

---

## Hard rules (do not violate)

1. **UPDRS is only used in Section 6 validation.** Never as a training/labeling signal anywhere upstream.
2. **Use `imu_sdata[sid][3]`** for raw time-domain data. Never `[0]`.
3. **Apply `exclude_subjects`** everywhere; no silent re-inclusion downstream.
4. **Sessions are the unit of analysis** — do not flatten all windows of a subject into a single pool before per-session aggregation (the subject with 1753 sessions would dominate everything otherwise).
5. **`np.trapezoid`** not `np.trapz` (numpy 2.x).
6. **Both severity paths (A and B) run to completion** in Section 4 — even if one looks worse.
7. Every section ends with a markdown "Findings / what I'd flag" cell.
8. Match existing Section 1 style: flat dataframes, short helpers at top of section, print-heavy for verifiability.
9. **Pause after each section** (build Section N, report findings, wait for user confirmation before Section N+1).
10. Never modify `neurosense_tremor.ipynb` — it's deprecated reference.

## Coding style
- Default to no comments. Only comment non-obvious WHYs (e.g., `# Welch more stable than raw FFT for 500-sample windows`).
- No abstractions for single-use code. No classes.
- Use `df.merge` / `groupby.agg` over manual loops when possible.
- Plots: one figure per question, `tight_layout()`, titles that state the key stat (e.g., `f'PD vs HC tremor_index (AUC={auc:.2f})'`).
- Keep function signatures flat — no `**kwargs` pass-through, no factory patterns.
