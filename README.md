# iq-to-tdm — SDR IQ to NASA CCSDS TDM v2.0 Converter

Converts amateur radio SDR IQ recordings to NASA Consultative Committee for Space Data Systems
(CCSDS) Tracking Data Message (TDM) format, suitable for submission to NASA one-way Doppler
tracking programs for Artemis and lunar missions.

**Station:** SP5LOT — Warsaw, Poland (52.1539°N, 21.1918°E)
**Missions:** Artemis II (Orion spacecraft), KPLO/Danuri

---

## Features

- Reads **SigMF** (`.sigmf-meta` + `.sigmf-data`) and **GQRX** raw recordings
- Supported IQ formats: `cf32_le`, `cf64_le`, `ci16_le`, `ci8`, `cu8`
- Weak-signal carrier detection via **Welch averaged periodogram** with parabolic
  sub-bin interpolation — gain of ~13 dB SNR with default 20 sub-blocks
- Outputs **CCSDS TDM v2.0 KVN** (`RECEIVE_FREQ_2`) ready for NASA submission
- Memory-mapped I/O for files larger than 2 GB
- Optional carrier hint (`--carrier-hint`, `--hint-bw`) for recordings with nearby
  interference or known approximate carrier offset
- Interactive probe phase: automatically suggests parameter adjustments when the
  initial acceptance rate is low (disable with `--no-interactive`)
- Optional Welch spectrum plot to PNG (`--plot`)

---

## Validation

### Artemis I — CAMRAS Dwingeloo Radio Telescope (DRO departure burn, 2022-12-01)

The primary validation uses the publicly available CAMRAS IQ clip
`examples/small.sigmf-meta` recorded at the 25-metre Dwingeloo Radio Telescope
on **2022-12-01 at 21:42:38 UTC** (Artemis I DRO departure burn day).
Equipment: DIFI-compliant receiver, external PPS reference, 2.0 Msps, ci16_le. CC BY 4.0.

Processing with `iq-to-tdm` (integration 0.3 s, `--carrier-hint -45617 --hint-bw 15000`):

**`examples/generated_small.tdm`** — Doppler **-45627.5 Hz** relative to 2216.5 MHz.

Independent cross-check against CAMRAS single-FFT log (`doppler_20221201.txt`,
available at [data.camras.nl/artemis](https://data.camras.nl/artemis/)):
at the same timestamp the single-FFT measurement reads **-45617 Hz**
— a difference of **~10 Hz**, within the single-FFT noise floor (±20 kHz).
This confirms the converter correctly identifies the carrier frequency.

### Artemis I — CAMRAS IQ 2022-11-19

The IQ recording `camras-2022_11_19_10_07_16_2216.500MHz_2.0Msps_ci16_le.sigmf-meta`
(2022-11-19, 10:07 UTC) is included as a second validation point.
Processing produces **`examples/CAMRAS_20221119_100716_SP5LOT.tdm`** — 9 measurements,
Doppler stable at **-50142 ± 1 Hz** relative to 2216.5 MHz.

Cross-check against CAMRAS single-FFT log (`doppler_20221119.txt`,
available at [data.camras.nl/artemis](https://data.camras.nl/artemis/)):
at the same timestamps the single-FFT reads **-50135 to -50138 Hz** (median -50136 Hz)
— a difference of **~6 Hz**, consistent with the Dec 1 result above.

Note: the CAMRAS reference TDM `CAMRAS_Orion_20221119_v1.tdm` covers 12:30–13:02 UTC
on the same day — a **different time window**. No CAMRAS IQ recording for the 12:30 UTC
session is publicly available (confirmed against the full CAMRAS archive index at
[gitlab.camras.nl/dijkema/artemistracking](https://gitlab.camras.nl/dijkema/artemistracking)).
The reference TDM is included for CCSDS format reference only. Both files use TDM v2.0;
conventions differ (`FREQ_OFFSET = 0`, `INTEGRATION_REF = MIDDLE` vs. our
`FREQ_OFFSET = center_freq`, `INTEGRATION_REF = END` per NASA guidance).

### KPLO/Danuri — SP5LOT, 2026-02-21

IQ recording: `gqrx_20260221_151916_2260790300_125000_fc.sigmf-meta`
Recorded by: SQ3DHO
Receiver: HackRF One, center 2260.7903 MHz, 125 kSps, recording duration 1 h 54 min.
Approximate location: 52.36°N, 16.68°E (Dopiewo, Greater Poland Voivodeship, Poland).

Output: **`examples/kplo_20260221.tdm`** — 6851 measurements, 1-second integration.

| UTC period | Doppler offset | Note |
|---|---|---|
| 15:19 – 15:47 | ~0 Hz | KPLO below effective horizon / DC region |
| 15:47 – 17:05 | +34429 → +27789 Hz | Active tracking, 4385 measurements |
| 17:05 – 17:13 | ~0 Hz | KPLO below effective horizon |

SNR 7–12 dB throughout the visible pass. The result is fully reproducible:
re-running the converter on the same file produces bit-identical output.

---

## Repository Contents — `examples/`

| File | Type | Description |
|---|---|---|
| `camras-2022_11_19_10_07_16_2216.500MHz_2.0Msps_ci16_le.sigmf-meta` | SigMF metadata | CAMRAS IQ recording, Artemis I, 2022-11-19 10:07 UTC |
| `CAMRAS_Orion_20221119_v1.tdm` | CCSDS TDM v2.0 | CAMRAS original TDM, same day 12:30–13:02 UTC |
| `CAMRAS_20221119_100716_SP5LOT.tdm` | CCSDS TDM v2.0 | Output of this converter from the CAMRAS IQ above |
| `gqrx_20260221_151916_2260790300_125000_fc.sigmf-meta` | SigMF metadata | SP5LOT IQ recording, KPLO/Danuri, 2026-02-21 (recorded by SQ3DHO) |
| `kplo_20260221.tdm` | CCSDS TDM v2.0 | Output of this converter from the SP5LOT KPLO IQ above |
| `small.sigmf-meta` | SigMF metadata | CAMRAS IQ clip, Artemis I DRO departure burn, 2022-12-01 |
| `generated_small.tdm` | CCSDS TDM v2.0 | Output of this converter from the small clip above |

Note: `.sigmf-data` binary IQ files are not included due to size (77 MB and 6.4 GB).
The CAMRAS IQ data files are publicly available from Stichting CAMRAS under CC BY 4.0.

---

## Usage Examples

### CAMRAS Artemis I IQ recording

```bash
python iq_to_tdm.py \
    --input  examples/camras-2022_11_19_10_07_16_2216.500MHz_2.0Msps_ci16_le.sigmf-meta \
    --station CAMRAS \
    --participant-1 ORION \
    --integration 1.0 \
    --output CAMRAS_20221119_100716.tdm
```

Expected output: 9 measurements, Doppler ~-50142 Hz, SNR ~5 dB.

### SP5LOT KPLO/Danuri recording

```bash
python iq_to_tdm.py \
    --input  examples/gqrx_20260221_151916_2260790300_125000_fc.sigmf-meta \
    --station SP5LOT \
    --participant-1 KPLO \
    --integration 1.0 \
    --output SP5LOT_KPLO_20260221.tdm
```

Expected output: 6851 measurements (4385 with active signal), Doppler +27789 to +34429 Hz.

### Weak signal with nearby sideband interference

Use `--carrier-hint` (approximate carrier offset from center, in Hz) and
`--hint-bw` to narrow the search window when the carrier is partially obscured
by data sidebands or other interference:

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

---

## Algorithm

1. Load IQ samples; use `numpy.memmap` for files larger than 2 GB
2. Split into non-overlapping integration windows (default 1.0 s)
3. For each window: compute Welch averaged periodogram
   (N sub-blocks with 50% overlap and Hanning window; SNR gain = 10 log10(N) dB)
4. Parabolic interpolation around the FFT peak for sub-bin frequency accuracy
5. Apply SNR threshold; optionally exclude PCM/PM/NRZ sideband regions
6. Timestamp each measurement at the end of its integration window
   (`INTEGRATION_REF = END`)
7. Write CCSDS TDM v2.0 KVN file

---

## Output Format

Standard CCSDS TDM v2.0 KVN. Example (KPLO/Danuri, SP5LOT):

```
CCSDS_TDM_VERS = 2.0
CREATION_DATE  = 2026-052T15:19:17.000Z
ORIGINATOR     = SP5LOT

COMMENT Artemis II one-way Doppler tracking
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
Periods with no detectable signal (spacecraft below effective horizon, or SNR
below threshold) are reported as +0.000 Hz.

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
--max-samples    Load only first N samples (for testing)
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

The output format follows:

- **CCSDS 503.0-B-2**, *Tracking Data Message (TDM)*, Blue Book, Issue 2,
  Consultative Committee for Space Data Systems, September 2007.
  Defines TDM structure, keywords (`RECEIVE_FREQ_2`, `FREQ_OFFSET`,
  `INTEGRATION_REF`, `TURNAROUND_NUMERATOR`, `TURNAROUND_DENOMINATOR`, etc.),
  time system, and KVN encoding used throughout this tool.

- **S-band coherent turnaround ratio 240/221** per NASA/CCSDS S-band frequency
  plan for Earth-spacecraft Doppler measurements in the 2025–2110 MHz / 2200–2290 MHz
  band.

The following reference data files were used for development and validation
(not included in this repository):

- `CAMRAS_Orion_20221119_v1.tdm` — Stichting CAMRAS, Dwingeloo, Netherlands.
  Artemis I tracking data, 2022-11-19, published under CC BY 4.0.
  (The file is also included in `examples/` for format reference.)
- CAMRAS SigMF IQ archive for Artemis I (multiple sessions, 2022-11-17 through
  2022-12-10) — Stichting CAMRAS, CC BY 4.0.
- `doppler_20221201.txt` — CAMRAS single-FFT carrier frequency log,
  2022-12-01, used to determine the approximate carrier offset for
  `--carrier-hint` parameter tuning.

---

## License

MIT — see [LICENSE](LICENSE)

## Author

SP5LOT — amateur radio station, Warsaw, Poland
Member of the AREx Artemis II Ground Station Project.
Contact NASA for Artemis tracking submission details via the official Artemis Amateur Tracking program.
