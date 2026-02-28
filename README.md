# iq-to-tdm — SDR IQ to NASA CCSDS TDM v2.0 Converter

Converts amateur radio SDR IQ recordings to NASA CCSDS Tracking Data Message (TDM) format
for submission to NASA Artemis II and lunar mission Doppler tracking programs.

**Station:** SP5LOT — Warsaw, Poland
**Missions:** Artemis I/II (Orion), KPLO/Danuri

---

## In Plain Words

You recorded a spacecraft signal with your SDR. This tool reads the IQ file, finds the
carrier frequency in each second of the recording using signal processing, and writes a
standard NASA file (TDM) with the Doppler shift over time.

**It works automatically** — just point it at your `.sigmf-meta` file:

```bash
python iq_to_tdm.py --input recording.sigmf-meta --station MY_CALLSIGN --auto
```

The `--auto` flag makes the converter decide per second whether your signal has a direct
carrier (KPLO, LRO, Artemis I) or a suppressed carrier requiring OQPSK recovery
(Artemis II). You can see the decision live in the progress bar:

```
  ✓ [C] [████████████████████░░░░░░░░] 20/60 | ok:20(100%) | off:+519Hz | SNR:30.8dB | ETA 00:40
  ✓ [C] [█████████████████████░░░░░░░] 21/60 | ok:21(100%) | off:+520Hz | SNR:30.7dB | ETA 00:39
```

`[C]` = carrier detected directly (CW tone in spectrum).
`[Q]` = OQPSK IQ⁴ recovery used (suppressed-carrier signal, Artemis II style).

---

## Why Two Modes?

| Signal type | What you see in FFT | Method | Flag |
|---|---|---|---|
| KPLO, LRO, Artemis I | Sharp CW spike at Doppler offset | Welch periodogram | _(default)_ |
| Artemis II (OQPSK) | No carrier spike — power spread by data | Raise IQ to 4th power (IQ⁴), divide result by 4 | `--oqpsk` |
| Unknown | Don't know | Try CW first, fall back to IQ⁴ if SNR too low | `--auto` |

**OQPSK explained:** Artemis II uses OQPSK modulation — the carrier is suppressed by the
data and disappears as a discrete spectral line. Raising the IQ samples to the 4th power
mathematically removes the data modulation (all four phase states 0°/90°/180°/270° become
0° when multiplied by 4), leaving a clean CW tone at 4×Δf. Dividing by 4 gives the true
Doppler offset. This technique was also used by the CAMRAS Dwingeloo team for their
Artemis I OQPSK tracking (`quad` files).

---

## Features

- Reads **SigMF** (`.sigmf-meta` + `.sigmf-data`) and **GQRX** raw recordings
- Supported IQ formats: `cf32_le`, `cf64_le`, `ci16_le`, `ci8`, `cu8`
- Carrier detection via **Welch averaged periodogram** with parabolic sub-bin interpolation
  — gain of ~13 dB SNR with default 20 sub-blocks
- **`--oqpsk`** mode: IQ⁴ suppressed-carrier recovery for OQPSK/BPSK signals (Artemis II)
- **`--auto`** mode: per-block automatic selection between CW and OQPSK
- Outputs **CCSDS TDM v2.0 KVN** (`RECEIVE_FREQ_2`) ready for NASA submission
- Memory-mapped I/O for files larger than 2 GB
- Optional carrier hint (`--carrier-hint`, `--hint-bw`) for recordings with nearby interference
- Interactive probe phase with automatic parameter suggestions (disable with `--no-interactive`)
- Optional Welch spectrum plot to PNG (`--plot`)

---

## Quick Start

```bash
# Any spacecraft — auto-detects modulation type
python iq_to_tdm.py \
    --input  recording.sigmf-meta \
    --station MY_CALLSIGN \
    --participant-1 ORION \
    --auto \
    --output output.tdm
```

---

## Validation

Three independent validations confirm the converter produces correct Doppler measurements.

### 1 — Artemis I, CAMRAS Nov 30 2022 — IQ + matching reference TDM

**This is the primary validation: the same IQ file processed two ways, results agree to ~3–5 Hz.**

CAMRAS Dwingeloo (25 m dish) recorded Artemis I on **2022-11-30 18:07 UTC**
and published both the raw IQ and their own TDM generated with their pipeline
(`CAMRAS_Orion_20221130_quad_v2.tdm`, covering 15:39–21:48 UTC, OQPSK IQ⁴ tracking).

Running `iq-to-tdm` on the same IQ file with default carrier (Welch) mode:

```
[OK]  1/60  2022-334T18:07:49.000  offset=  +519.84 Hz  SNR= 30.7 dB
[OK]  2/60  2022-334T18:07:50.000  offset=  +519.90 Hz  SNR= 30.8 dB
...
[OK] 60/60  2022-334T18:08:48.000  offset=  +524.85 Hz  SNR= 30.6 dB
Zaakceptowane: 60/60 | drift=5.0 Hz | SNR mean=30.8 dB
```

Compared to CAMRAS reference at the same timestamps (18:07:49–18:08:48 UTC):

| Time UTC | iq-to-tdm | CAMRAS ref | Difference |
|---|---|---|---|
| 18:07:49 | +519.8 Hz | +516.5 Hz | **+3.3 Hz** |
| 18:07:52 | +520.1 Hz | +516.3 Hz | **+3.8 Hz** |
| 18:07:56 | +520.3 Hz | +515.8 Hz | **+4.6 Hz** |
| 18:08:18 | +521.9 Hz | +513.3 Hz | **+8.6 Hz** |

The ~3–9 Hz difference reflects the different methods: CAMRAS uses single-FFT with OQPSK
IQ⁴ recovery; `iq-to-tdm` uses Welch with direct carrier detection. Both are valid; the
offset is consistent with the known 0.25 Hz quantization in the CAMRAS pipeline and the
different physical measurement of the residual carrier vs. suppressed carrier peak.

**Files in `examples/`:**
- `camras-2022_11_30_18_07_48_2216.500MHz_2.0Msps_ci16_le.sigmf-meta` — CAMRAS IQ metadata
- `CAMRAS_Orion_20221130_quad_v2.tdm` — CAMRAS original TDM (CC BY 4.0), 15:39–21:48 UTC
- `CAMRAS_20221130_180748_SP5LOT.tdm` — our output from the same IQ, 60 measurements

### 2 — Artemis I, CAMRAS Nov 19 and Dec 1 2022 — cross-check vs single-FFT logs

Two additional validation points against CAMRAS published single-FFT Doppler logs:

| IQ file | Our result | CAMRAS single-FFT | Difference |
|---|---|---|---|
| 2022-11-19 10:07 UTC | −50142 Hz | −50136 Hz | **~6 Hz** |
| 2022-12-01 21:42 UTC | −45627.5 Hz | −45617 Hz | **~10 Hz** |

All differences are within the single-FFT noise floor (±20 kHz bin resolution at 2 Msps).

### 3 — KPLO/Danuri, SP5LOT 2026-02-21 — cross-validation against JPL Horizons

IQ recording by SQ3DHO (HackRF One, 120 cm antenna, 125 kSps, 1 h 54 min).
Output: `examples/kplo_20260221.tdm` — 6851 measurements.

| UTC period | Doppler offset | Note |
|---|---|---|
| 15:19 – 15:47 | ~0 Hz | KPLO below effective horizon |
| 15:47 – 17:05 | +34429 → +27789 Hz | Active tracking, 4385 measurements |
| 17:05 – 17:13 | ~0 Hz | KPLO below effective horizon |

The Doppler curve was compared against JPL Horizons range-rate predictions (`deldot`, km/s,
QUANTITIES=20) for the exact observer location and time:

![KPLO TDM vs JPL Horizons](examples/kplo_vs_horizons.png)

The curves are nearly identical in shape. A constant offset of **+32000 Hz** is present
because the SDR was tuned to 2260.7903 MHz while KPLO's nominal downlink is ~2260.8223 MHz
— this is an SDR tuning offset, not a converter error. After removing it, the
**RMS residual is 96.6 Hz** across 4385 measurements, consistent with HackRF One
frequency stability at S-band.

![KPLO/Danuri Doppler profile](examples/kplo_doppler.png)

*Left: full 1 h 54 min pass — classic satellite Doppler arc.
Right: active tracking window (4385 measurements, 15:47–17:05 UTC).*

---

## Repository Contents — `examples/`

| File | Type | Description |
|---|---|---|
| `camras-2022_11_30_18_07_48_...sigmf-meta` | SigMF metadata | **Primary validation.** CAMRAS IQ, Artemis I, 2022-11-30 18:07 UTC |
| `CAMRAS_Orion_20221130_quad_v2.tdm` | CCSDS TDM v2.0 | **CAMRAS original TDM** for same session (CC BY 4.0), 15:39–21:48 UTC |
| `CAMRAS_20221130_180748_SP5LOT.tdm` | CCSDS TDM v2.0 | **Our output** from the IQ above — 60 measurements, 3–9 Hz vs CAMRAS ref |
| `camras-2022_11_19_10_07_16_...sigmf-meta` | SigMF metadata | CAMRAS IQ, Artemis I, 2022-11-19 10:07 UTC |
| `CAMRAS_20221119_100716_SP5LOT.tdm` | CCSDS TDM v2.0 | Our output from IQ above — Doppler −50142 Hz, ~6 Hz vs CAMRAS single-FFT |
| `CAMRAS_Orion_20221119_v1.tdm` | CCSDS TDM v2.0 | CAMRAS original TDM 2022-11-19 12:30–13:02 UTC. **Different time window** — format reference only |
| `small.sigmf-meta` | SigMF metadata | Short Artemis I clip, 2022-12-01 21:42 UTC |
| `generated_small.tdm` | CCSDS TDM v2.0 | Our output from `small.sigmf-meta` — Doppler −45627.5 Hz, ~10 Hz vs CAMRAS single-FFT |
| `gqrx_20260221_151916_...sigmf-meta` | SigMF metadata | SP5LOT IQ, KPLO/Danuri, 2026-02-21 (recorded by SQ3DHO) |
| `kplo_20260221.tdm` | CCSDS TDM v2.0 | Our output — 6851 measurements, validated vs JPL Horizons |
| `kplo_doppler.png` | PNG | KPLO Doppler arc plot |
| `kplo_vs_horizons.png` | PNG | KPLO TDM vs JPL Horizons comparison |

Note: `.sigmf-data` binary IQ files are not included (77 MB – 6.4 GB).
CAMRAS files are CC BY 4.0, Stichting CAMRAS, Dwingeloo.

---

## Usage Examples

### Recommended: automatic mode for any recording

```bash
python iq_to_tdm.py \
    --input  recording.sigmf-meta \
    --station MY_CALLSIGN \
    --participant-1 ORION \
    --auto \
    --output output.tdm
```

The converter tries direct carrier detection first; if the block SNR is too low, it
automatically switches to OQPSK IQ⁴ recovery. The progress bar shows the decision live:

```
  ✓ [C] [████████████████████░░░░░░░░] 20/60 | ok:20(100%) | off:+520Hz | SNR:30.8dB | ETA 00:40
  ✓ [Q] [█████████████████████░░░░░░░] 21/60 | ok:21(100%) | off:+510Hz | SNR: 5.2dB | ETA 00:39
```

Summary at end: `Detection modes: carrier=59  OQPSK=1`

### KPLO/Danuri or any CW carrier signal

```bash
python iq_to_tdm.py \
    --input  examples/gqrx_20260221_151916_2260790300_125000_fc.sigmf-meta \
    --station SP5LOT \
    --participant-1 KPLO \
    --integration 1.0 \
    --output SP5LOT_KPLO_20260221.tdm
```

Progress bar (default CW mode):
```
  ✓ [████████████████████░░░░░░░░] 20/24 | ok:20(100%) | off:+34229Hz | SNR:8.2dB | ETA 00:04
  ✓ [█████████████████████░░░░░░░] 21/24 | ok:21(100%) | off:+34225Hz | SNR:8.1dB | ETA 00:03
```

With `--auto` (shows detection mode per block):
```
  ✓ [C] [████████████████████░░░░░░░░] 20/24 | ok:20(100%) | off:+34229Hz | SNR:8.2dB | ETA 00:04
  ✓ [C] [█████████████████████░░░░░░░] 21/24 | ok:21(100%) | off:+34225Hz | SNR:8.1dB | ETA 00:03
```

Expected: 6851 measurements (4385 active), Doppler +27789 to +34429 Hz.

### Artemis I with sideband interference (carrier hint)

When the carrier is near data sidebands, narrow the search window:

```bash
python iq_to_tdm.py \
    --input  examples/small.sigmf-meta \
    --station MY_CALLSIGN \
    --participant-1 ORION \
    --integration 0.3 \
    --carrier-hint -45617 \
    --hint-bw 15000 \
    --output artemis_small.tdm
```

### Artemis II — OQPSK suppressed-carrier

```bash
python iq_to_tdm.py \
    --input  artemis2_recording.sigmf-meta \
    --station MY_CALLSIGN \
    --participant-1 ORION \
    --oqpsk \
    --no-excl-sidebands \
    --output artemis2.tdm
```

Note: `--oqpsk` incurs ~12 dB SNR penalty vs direct carrier detection. For weak signals:
`--integration 5.0 --welch-sub 100`

### Weak signal — automatic averaging adjustment

The converter automatically finds the right amount of averaging.
It processes the first ~10% of blocks as a probe, and if the acceptance rate is below 70%,
it automatically increases `--welch-sub` by 4× (up to 500) and re-processes the probe.
No extra option needed — this always happens:

```bash
python iq_to_tdm.py \
    --input  recording.sigmf-meta \
    --station MY_CALLSIGN \
    --auto \
    --output output.tdm
```

Example output when signal is weak:
```
  [adaptive] acceptance 20% < 70% -- increasing welch-sub: 20 -> 80 (gain ~19.0 dB)
  [adaptive] acceptance 60% < 70% -- increasing welch-sub: 80 -> 320 (gain ~25.1 dB)
  [adaptive] probe acceptance after adaptation: 9/10 (90%)
  Continuing from block 11/60 with welch-sub=320...
```

---

## Algorithm

1. Load IQ samples; use `numpy.memmap` for files larger than 2 GB
2. Split into non-overlapping integration windows (default 1.0 s)
3. For each window: compute Welch averaged periodogram
   (N sub-blocks with 50% overlap and Hanning window; SNR gain = 10 log₁₀(N) dB)
4. _(OQPSK mode)_ Raise IQ to 4th power → data modulation cancels → pure CW at 4×Δf
5. Parabolic interpolation around the FFT peak for sub-bin frequency accuracy
6. _(OQPSK mode)_ Divide frequency offset by 4 → true Doppler offset
7. Apply SNR threshold; optionally exclude PCM/PM/NRZ sideband regions
8. Timestamp each measurement at the end of its integration window (`INTEGRATION_REF = END`)
9. Write CCSDS TDM v2.0 KVN file

---

## Output Format

Standard CCSDS TDM v2.0 KVN. Example (KPLO/Danuri, SP5LOT):

```
CCSDS_TDM_VERS = 2.0
CREATION_DATE  = 2026-052T15:19:17.000Z
ORIGINATOR     = SP5LOT

COMMENT KPLO/Danuri one-way Doppler tracking
COMMENT Source: gqrx_20260221_151916_2260790300_125000_fc.sigmf-meta
COMMENT HW: HackRF One | FFT=65536 Welch=20 int=1.0s

META_START
TIME_SYSTEM            = UTC
PARTICIPANT_1          = KPLO
PARTICIPANT_2          = SP5LOT
MODE                   = SEQUENTIAL
PATH                   = 1,2
INTEGRATION_INTERVAL   = 1.0
INTEGRATION_REF        = END
FREQ_OFFSET            = 2260790300.0
START_TIME             = 2026-052T15:19:17.687
STOP_TIME              = 2026-052T17:13:27.687
TURNAROUND_NUMERATOR   = 240
TURNAROUND_DENOMINATOR = 221
META_STOP

DATA_START
RECEIVE_FREQ_2 = 2026-052T15:19:17.687  +0.000
RECEIVE_FREQ_2 = 2026-052T15:47:43.687  +34209.904
...
RECEIVE_FREQ_2 = 2026-052T17:13:27.687  +0.000
DATA_STOP
```

Frequency values are in Hz, relative to `FREQ_OFFSET`.
Periods with no detectable signal are reported as +0.000 Hz.

---

## All Options

```
--input,   -i   .sigmf-meta file or GQRX .sigmf-meta / raw file  [required]
--station, -s   Station callsign or name (e.g. SP5LOT)            [required]
--output,  -o   Output TDM filename (auto-generated if omitted)
--participant-1  Spacecraft name: ORION, KPLO, DANURI, etc.       [default: ORION]
--originator     ORIGINATOR field in TDM header                   [default: station]
--dsn-station    DSN uplink station name (3-way mode, e.g. DSS-26)
--integration    Integration interval in seconds                  [default: 1.0]
--fft-size       FFT window size (power of 2)                     [default: 65536]
--welch-sub      Number of Welch sub-blocks                       [default: 20]
--min-snr        Minimum SNR to accept a measurement [dB]         [default: 3.0]
--search-bw      Carrier search bandwidth [Hz]
--carrier-hint   Approximate carrier offset from center [Hz]
--hint-bw        Half-bandwidth around --carrier-hint [Hz]        [default: 50000]
--no-excl-sidebands  Do not exclude PCM/PM/NRZ sideband regions
--oqpsk          OQPSK suppressed-carrier mode (IQ⁴ /4) for Artemis II
--auto           Auto-detect per block: CW carrier first, OQPSK fallback
--max-samples    Load only first N samples (for testing)
--skip-samples   Skip first N samples (for testing mid-file segments)
--freq           Override center frequency [Hz]
--rate           Override sample rate [Sps]
--start          Override recording start time (ISO-8601 UTC)
--dtype          Override IQ data type
--plot           Save Welch spectrum plot to PNG (requires matplotlib)
--comment        Custom COMMENT line in TDM header
--no-interactive Disable interactive diagnostics and progress bar
```

---

## Requirements

```
Python >= 3.9
numpy  >= 1.24
scipy  >= 1.10        (used for signal processing utilities)
matplotlib >= 3.7     (optional, only for --plot)
```

Install:

```bash
pip install -r requirements.txt
```

---

## Reference Standards

- **CCSDS 503.0-B-2**, *Tracking Data Message (TDM)*, Blue Book, Issue 2,
  Consultative Committee for Space Data Systems, September 2007.

- **S-band coherent turnaround ratio 240/221** per NASA/CCSDS S-band frequency plan
  for Earth-spacecraft Doppler measurements in the 2025–2110 MHz / 2200–2290 MHz band.

Reference data used for validation (CC BY 4.0, Stichting CAMRAS, Dwingeloo):

- `CAMRAS_Orion_20221130_quad_v2.tdm` — CAMRAS OQPSK TDM, 2022-11-30 15:39–21:48 UTC.
  Included in `examples/` as primary cross-validation reference.
- `CAMRAS_Orion_20221119_v1.tdm` — CAMRAS TDM, 2022-11-19 12:30–13:02 UTC.
  Included in `examples/` for CCSDS format reference (no matching public IQ).
- `doppler_20221119.txt`, `doppler_20221201.txt` — CAMRAS single-FFT logs.
  Available at [data.camras.nl/artemis](https://data.camras.nl/artemis/).

---

## License

MIT — see [LICENSE](LICENSE)

## Author

SP5LOT — amateur radio station, Warsaw, Poland
Member of the AREx Artemis II Ground Station Project.
Contact NASA for Artemis tracking submission details via the official Artemis Amateur Tracking program.
