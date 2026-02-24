# Poradnik użytkownika / User Guide

- [🇵🇱 Wersja polska](#-wersja-polska)
- [🇬🇧 English version](#-english-version)

---

## 🇵🇱 Wersja polska

### Minimum — dwie wymagane opcje

```bash
python iq_to_tdm.py \
  --input nagranie.sigmf-meta \
  --station SP5LOT
```

- `--input` / `-i` — plik nagrania: `.sigmf-meta` (SigMF) lub plik GQRX
- `--station` / `-s` — twój callsign lub nazwa stacji (np. `SP5LOT`, `MY_STATION`)

To wystarczy dla typowego nagrania SigMF z kompletem metadanych.

---

### Gdzie trafi wynik

```bash
--output SP5LOT_20260221.tdm
```

Bez tej opcji plik nazywa się automatycznie: `SP5LOT_20260221_151916.tdm`
(callsign + data + godzina z metadanych).

---

### Jaki statek kosmiczny nagrywałeś

```bash
--participant-1 KPLO        # domyślnie: ORION
```

Wpisuje się w nagłówek TDM jako `PARTICIPANT_1`. Użyj nazwy misji: `ORION`, `KPLO`, `DANURI`, itp.

```bash
--dsn-station DSS-26        # tryb 3-way (stacja DSN + Twoja + statek)
```

Tylko jeśli nagrywałeś sygnał re-transmitowany przez DSN. Dla typowego odbioru bezpośredniego — nie potrzebne.

```bash
--originator NASA_JSC       # domyślnie: wartość z --station
```

Pole `ORIGINATOR` w nagłówku TDM. Zazwyczaj niepotrzebne — domyślnie wpisuje się Twój callsign.

---

### Słaby sygnał — popraw wykrywalność

Gdy konwerter odrzuca dużo bloków (niska akceptacja), masz trzy narzędzia:

**1. Dłuższa integracja** — uśredniasz więcej próbek na jeden pomiar:

```bash
--integration 5.0    # domyślnie: 1.0 s
```

Daje lepszy SNR, ale mniej punktów w TDM (co 5 sekund zamiast co sekundę).
NASA akceptuje 1 s i 10 s.

**2. Więcej sub-bloków Welcha** — więcej uśredniania FFT:

```bash
--welch-sub 50       # domyślnie: 20
```

Każde podwojenie = +3 dB SNR. Przy małej antenie spróbuj 50–200.

**3. Niższy próg SNR** — akceptuj słabsze pomiary:

```bash
--min-snr 1.5        # domyślnie: 3.0 dB
```

Ostrożnie — zbyt nisko i wyniki będą zaszumione.

---

### Sygnał jest, ale konwerter szuka nie tam

Gdy widzisz sygnał na wodospadzie ale konwerter go nie łapie — wąskie okno wyszukiwania pomaga:

```bash
--carrier-hint -45000        # nośna jest ~45 kHz PONIŻEJ center
--carrier-hint +12000        # nośna jest ~12 kHz POWYŻEJ center
```

Wartość odczytujesz z wodospadu (SDR#, GQRX) jako offset od częstotliwości centralnej.

Razem z `--hint-bw` (szerokość okna wokół hintu):

```bash
--carrier-hint -45000 --hint-bw 10000   # szukaj tylko w oknie ±10 kHz
```

Domyślnie `--hint-bw` = 50 000 Hz (±50 kHz). Zwęź jeśli w pobliżu nośnej są silne sidebands.

```bash
--search-bw 20000     # szukaj tylko w środkowych ±20 kHz pasma
```

Alternatywa gdy nie znasz dokładnego offsetu, ale wiesz że sygnał jest blisko center.

```bash
--no-excl-sidebands   # nie omijaj pasm sideband
```

Normalnie konwerter wyklucza obszary znanych sideband Oriona (72 kHz, 2 MHz...).
Użyj tej opcji tylko jeśli nośna przypadkowo pokrywa się z tym pasmem.

---

### Nagranie GQRX bez metadanych w nazwie pliku

GQRX zapisuje freq i rate w nazwie pliku (`gqrx_20260221_151916_2260790300_125000_fc.raw`) —
konwerter to odczytuje automatycznie.

Ale jeśli plik jest przemianowany lub dane brakujące:

```bash
--freq  2216500000           # częstotliwość centralna w Hz (= 2216.5 MHz)
--rate  2000000              # sample rate w Hz (= 2.0 Msps)
--start 2026-02-21T15:19:16Z # czas startu nagrania (ISO-8601 UTC)
--dtype ci16_le              # format próbek: cf32_le, ci16_le, ci8, cu8, cf64_le
```

---

### Narzędzia diagnostyczne

```bash
--plot                  # zapisz wykres widma Welcha do PNG (wymaga matplotlib)
```

Przydatne gdy nie wiesz gdzie jest nośna — rysunek pokaże Ci widmo z zaznaczonymi
sideband Oriona.

```bash
--max-samples 500000    # załaduj tylko pierwsze N próbek
```

Do szybkiego testu na dużym pliku. Nie używaj w produkcji.

```bash
--comment "Artemis II obserwacja próbna 21.02.2026"
```

Własny komentarz w nagłówku TDM. Domyślnie wpisuje się opis automatyczny.

---

### Szybka ściągawka — typowe scenariusze

| Sytuacja | Kluczowe opcje |
|----------|---------------|
| Standardowe nagranie SigMF | tylko `--input` + `--station` |
| KPLO / Danuri (nie Orion) | + `--participant-1 KPLO` |
| Słaby sygnał, mała antena | + `--welch-sub 100 --integration 5` |
| Nośna widoczna na wodospadzie | + `--carrier-hint <Hz>` |
| Plik GQRX bez metadanych | + `--freq` + `--rate` + `--start` |
| Nie wiem gdzie jest sygnał | + `--plot` → patrz na wykres |
| Skrypt/automatyzacja | + `--no-interactive` |

---
---

## 🇬🇧 English version

### Minimum — two required options

```bash
python iq_to_tdm.py \
  --input recording.sigmf-meta \
  --station MY_CALLSIGN
```

- `--input` / `-i` — recording file: `.sigmf-meta` (SigMF) or GQRX file
- `--station` / `-s` — your callsign or station name (e.g. `W1ABC`, `MY_STATION`)

This is enough for a typical SigMF recording with complete metadata.

---

### Output file

```bash
--output MY_STATION_20260221.tdm
```

Without this option the filename is auto-generated: `MY_STATION_20260221_151916.tdm`
(callsign + date + time from metadata).

---

### Which spacecraft did you record?

```bash
--participant-1 KPLO        # default: ORION
```

Written into the TDM header as `PARTICIPANT_1`. Use the mission name: `ORION`, `KPLO`, `DANURI`, etc.

```bash
--dsn-station DSS-26        # 3-way mode (DSN station + yours + spacecraft)
```

Only if you recorded a signal re-transmitted via DSN. Not needed for direct reception.

```bash
--originator NASA_JSC       # default: value from --station
```

The `ORIGINATOR` field in the TDM header. Usually not needed — defaults to your callsign.

---

### Weak signal — improve detection

If the converter rejects many blocks (low acceptance rate), you have three tools:

**1. Longer integration** — average more samples per measurement:

```bash
--integration 5.0    # default: 1.0 s
```

Gives better SNR but fewer data points (one every 5 s instead of every second).
NASA accepts 1 s and 10 s intervals.

**2. More Welch sub-blocks** — more FFT averaging:

```bash
--welch-sub 50       # default: 20
```

Every doubling = +3 dB SNR. With a small antenna try 50–200.

**3. Lower SNR threshold** — accept weaker measurements:

```bash
--min-snr 1.5        # default: 3.0 dB
```

Be careful — too low and the results will be noisy.

---

### Signal is there but the converter misses it

If you can see the signal on the waterfall but the converter doesn't detect it,
a narrow search window helps:

```bash
--carrier-hint -45000        # carrier is ~45 kHz BELOW center
--carrier-hint +12000        # carrier is ~12 kHz ABOVE center
```

Read this value from your waterfall (SDR#, GQRX) as an offset from the center frequency.

Combined with `--hint-bw` (search window half-width around the hint):

```bash
--carrier-hint -45000 --hint-bw 10000   # search only within ±10 kHz
```

Default `--hint-bw` = 50 000 Hz (±50 kHz). Narrow it down if there are strong
sidebands close to the carrier.

```bash
--search-bw 20000     # search only within the central ±20 kHz of the band
```

Alternative when you don't know the exact offset but know the signal is near center.

```bash
--no-excl-sidebands   # do not exclude sideband regions
```

By default the converter excludes known Orion sideband areas (72 kHz, 2 MHz…).
Use this only if the carrier happens to fall within those bands.

---

### GQRX recording without metadata in the filename

GQRX encodes frequency and sample rate in the filename
(`gqrx_20260221_151916_2260790300_125000_fc.raw`) — the converter reads this automatically.

If the file has been renamed or metadata is missing, supply it manually:

```bash
--freq  2216500000           # center frequency in Hz (= 2216.5 MHz)
--rate  2000000              # sample rate in Hz (= 2.0 Msps)
--start 2026-02-21T15:19:16Z # recording start time (ISO-8601 UTC)
--dtype ci16_le              # IQ data type: cf32_le, ci16_le, ci8, cu8, cf64_le
```

---

### Diagnostic tools

```bash
--plot                  # save a Welch spectrum plot to PNG (requires matplotlib)
```

Useful when you don't know where the carrier is — the plot shows the spectrum
with Orion's sideband positions marked.

```bash
--max-samples 500000    # load only the first N samples
```

For quick testing on large files. Do not use in production.

```bash
--comment "Artemis II test observation 2026-02-21"
```

Custom comment line in the TDM header. By default an automatic description is used.

---

### Quick reference — common scenarios

| Situation | Key options |
|-----------|-------------|
| Standard SigMF recording | `--input` + `--station` only |
| KPLO / Danuri (not Orion) | + `--participant-1 KPLO` |
| Weak signal, small antenna | + `--welch-sub 100 --integration 5` |
| Carrier visible on waterfall | + `--carrier-hint <Hz>` |
| GQRX file without metadata | + `--freq` + `--rate` + `--start` |
| Don't know where the signal is | + `--plot` → inspect the spectrum |
| Script / automation | + `--no-interactive` |
