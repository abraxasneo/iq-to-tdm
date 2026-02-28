#!/usr/bin/env python3
"""
IQ → NASA CCSDS TDM Converter  (wersja z Welch averaging dla słabych sygnałów)
================================================================================
Konwertuje nagrania IQ (SigMF lub GQRX) na plik TDM dla NASA Artemis II.

KLUCZOWA RÓŻNICA OD PROSTEJ WERSJI:
  - Używa metody Welcha (uśrednianie wielu FFT) dla niskiego SNR
  - Wyszukuje WĄSKĄ nośną (residual carrier PCM/PM/NRZ), nie sidebands
  - Działa z małymi antenami (9–20 m apertura)

SYGNAŁ ORIONA (Artemis II):
  - Modulacja: PCM/PM/NRZ (Phase Modulation + Non-Return-to-Zero)
  - Nośna: 2216.5 MHz (residual carrier – NIE jest w pełni stłumiona)
  - Data rate: 72 ksps (contingency) lub 2 Msps (nominal) lub 4/6 Msps (SQPSK)
  - Na widmie widzisz: wąska linia nośnej + sinc-shaped sidebands po bokach
  - MY MIERZYMY TYLKO NOŚNĄ – sidebands nas nie interesują

Użycie:
  # SigMF:
  python iq_to_tdm.py --input nagranie.sigmf-meta --station MY_CALL

  # GQRX raw (freq i rate z nazwy pliku):
  python iq_to_tdm.py --input gqrx_20260210_120000_2216500000_2000000_fc.raw --station MY_CALL

  # GQRX raw (ręczne parametry):
  python iq_to_tdm.py --input nagranie.raw \\
      --freq 2216500000 --rate 2000000 --start "2026-02-10T12:00:00Z" \\
      --station MY_CALL

  # Słaby sygnał – więcej uśredniania:
  python iq_to_tdm.py --input nagranie.sigmf-meta --station MY_CALL \\
      --integration 10 --welch-sub 50 --min-snr 2.0

  # Znana pozycja nośnej na wodospadzie (np. 15 kHz poniżej center):
  python iq_to_tdm.py --input nagranie.sigmf-meta --station MY_CALL \\
      --carrier-hint -15000
"""

import argparse
import json
import math
import numpy as np
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Stałe NASA / S-band
# ─────────────────────────────────────────────────────────────────────────────

TURNAROUND_NUMERATOR   = 240   # S-band
TURNAROUND_DENOMINATOR = 221

# Typowe data rates Oriona → sidebands w tych odległościach od nośnej
ORION_DATA_RATES_HZ = [72_000, 2_000_000, 4_000_000, 6_000_000]
# Szerokość ochronna wokół sideband (Hz) – obszar wyłączony z wyszukiwania nośnej
SIDEBAND_GUARD_HZ = 5_000


# ─────────────────────────────────────────────────────────────────────────────
# Parsowanie SigMF
# ─────────────────────────────────────────────────────────────────────────────

def read_sigmf_meta(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    g   = meta.get("global", {})
    cap = (meta.get("captures") or [{}])[0]

    datatype    = g.get("core:datatype", "cf32_le").lower()
    sample_rate = float(g.get("core:sample_rate", 0))
    center_freq = float(cap.get("core:frequency", 0))
    dt_str      = cap.get("core:datetime") or g.get("core:datetime") or ""
    start_time  = _parse_dt(dt_str) if dt_str else None

    return {
        "datatype":    datatype,
        "sample_rate": sample_rate,
        "center_freq": center_freq,
        "start_time":  start_time,
        "hw":          g.get("core:hw", ""),
        "description": g.get("core:description", ""),
    }


def load_iq(data_path, datatype, max_samples=None):
    """
    Ładuje plik IQ i zwraca tablicę complex64.
    Dla plików >2 GB używa np.memmap zamiast np.fromfile,
    żeby nie wczytywać całości do RAM naraz.
    """
    raw_map = {
        "cf32_le": np.float32, "cf32_be": np.float32,
        "cf64_le": np.float64,
        "ci16_le": np.int16,   "ci16_be": np.int16,
        "ci8":     np.int8,
        "cu8":     np.uint8,
    }
    dt = datatype.lower()
    if dt not in raw_map:
        raise ValueError(f"Nieobslugiwany datatype: {dt}")

    file_size = os.path.getsize(str(data_path))
    elem_dtype = raw_map[dt]
    elem_size  = np.dtype(elem_dtype).itemsize
    n_elems    = file_size // elem_size
    if max_samples:
        n_elems = min(n_elems, max_samples * 2)

    LARGE_FILE_BYTES = 2 * 1024 ** 3  # 2 GB
    if file_size > LARGE_FILE_BYTES:
        raw = np.memmap(str(data_path), dtype=elem_dtype, mode='r', shape=(n_elems,))
    else:
        raw = np.fromfile(str(data_path), dtype=elem_dtype)
        if max_samples:
            raw = raw[: max_samples * 2]

    if dt.startswith("cf32"):
        # memmap view nie wymaga .copy() – process_iq czyta blokami
        iq = raw.view(np.complex64)
        if "be" in dt:
            iq = iq.byteswap().newbyteorder()
    elif dt.startswith("cf64"):
        iq = raw.view(np.complex128).astype(np.complex64)
    elif dt.startswith("ci16"):
        iq = (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)) / 32768.0
        iq = iq.astype(np.complex64)
    elif dt.startswith("ci8"):
        iq = (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)) / 128.0
        iq = iq.astype(np.complex64)
    elif dt.startswith("cu8"):
        f  = raw.astype(np.float32) - 127.5
        iq = (f[0::2] + 1j * f[1::2]).astype(np.complex64)
    else:
        raise ValueError(f"Nie wiem jak zaladowac: {dt}")

    return iq


# ─────────────────────────────────────────────────────────────────────────────
# Parsowanie GQRX z nazwy pliku
# ─────────────────────────────────────────────────────────────────────────────

def parse_gqrx_filename(name):
    """gqrx_YYYYMMDD_HHMMSS_FREQ_RATE_fc.raw"""
    info = {}
    m = re.search(r"gqrx_(\d{8})_(\d{6})_(\d+)_(\d+)", name, re.IGNORECASE)
    if m:
        d, t = m.group(1), m.group(2)
        try:
            info["start_time"] = datetime(
                int(d[:4]), int(d[4:6]), int(d[6:8]),
                int(t[:2]), int(t[2:4]), int(t[4:6]),
                tzinfo=timezone.utc,
            )
        except ValueError:
            pass
        info["center_freq"] = float(m.group(3))
        info["sample_rate"] = float(m.group(4))
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Czas
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dt(s):
    s = s.strip().rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"Nie moge sparsowac daty: '{s}'")


def _dt_to_tdm(dt):
    """YYYY-DOYThh:mm:ss.mmm  (format CCSDS TDM day-of-year)"""
    doy = dt.timetuple().tm_yday
    return (f"{dt.year}-{doy:03d}T"
            f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}."
            f"{dt.microsecond // 1000:03d}")


# ─────────────────────────────────────────────────────────────────────────────
# DSP CORE: Welch periodogram
# ─────────────────────────────────────────────────────────────────────────────

def welch_psd(iq_block, fft_size, n_sub):
    """
    Uśredniony periodogram metodą Welcha z 50% overlap.

    Dlaczego to działa przy słabym sygnale:
      Szum jest losowy  → uśrednianie N widm redukuje go o sqrt(N)
      Nośna CW jest deterministyczna → jej moc NIE maleje przy uśrednianiu
      Zysk SNR = 10*log10(n_sub) dB
        n_sub=20  → +13 dB
        n_sub=100 → +20 dB
        n_sub=500 → +27 dB

    Args:
        iq_block : tablica complex64
        fft_size : rozmiar jednego FFT (powinien być potęgą 2)
        n_sub    : max liczba sub-bloków do uśrednienia

    Returns:
        Uśrednione widmo mocy (float64), długość fft_size
    """
    hop    = fft_size // 2   # 50% overlap
    window = np.hanning(fft_size).astype(np.float32)
    wsum   = float(np.sum(window ** 2))

    psd   = np.zeros(fft_size, dtype=np.float64)
    count = 0
    pos   = 0

    while pos + fft_size <= len(iq_block) and count < n_sub:
        seg  = iq_block[pos : pos + fft_size]
        spec = np.fft.fft(seg * window, n=fft_size)
        psd += (np.abs(spec) ** 2) / wsum
        pos  += hop
        count += 1

    if count == 0:
        raise ValueError("Za malo probek dla Welch PSD – zwieksz integration lub zmniejsz fft_size")

    return psd / count


def _parabolic(psd, k, fft_size, sr):
    """Interpolacja paraboliczna binu FFT → dokładniejszy offset [Hz]."""
    if 0 < k < fft_size - 1:
        y1, y2, y3 = psd[k-1], psd[k], psd[k+1]
        denom = 2*y2 - y1 - y3
        if denom > 0:
            delta = 0.5 * (y1 - y3) / denom
            return (k + delta) * sr / fft_size
    return k * sr / fft_size


def estimate_carrier(
    iq_block,
    sample_rate,
    center_freq,
    fft_size    = 65536,
    n_sub       = 20,
    search_bw   = None,
    carrier_hint = None,
    hint_bw     = 50_000,
    excl_sidebands = True,
    oqpsk       = False,
):
    """
    Znajdź residual carrier PCM/PM/NRZ metodą Welcha.

    Tryb --oqpsk (M-th power carrier recovery):
      Dla OQPSK (suppressowana nośna) podnosimy IQ do 4. potęgi.
      Modulacja QPSK znika (fazy 0/90/180/270° → wszystkie ×4 = 0°),
      pozostaje czysta linia CW na 4×Δf. Wynik dzielimy przez 4.

    Logika maski wyszukiwania:
      1. search_bw  → ogranicz do środkowego search_bw Hz
      2. carrier_hint → szukaj tylko ±hint_bw Hz wokół podpowiedzi (domyślnie ±50 kHz)
      3. excl_sidebands → wyklucz obszary przy ±dr od center (sidebands)

    Returns:
        (freq_abs_hz, snr_db)
    """
    if oqpsk:
        # Podnieś do 4. potęgi — usuwa modulację OQPSK, zostaje CW na 4×Δf
        iq_proc = (iq_block.astype(np.complex128) ** 4).astype(np.complex64)
    else:
        iq_proc = iq_block

    psd   = welch_psd(iq_proc, fft_size, n_sub)
    freqs = np.fft.fftfreq(fft_size, d=1.0/sample_rate)   # offset [Hz] od center
    bin_hz = sample_rate / fft_size

    # ── Maska wyszukiwania ──────────────────────────────────────────────────
    if carrier_hint is not None:
        # Znamy mniej więcej gdzie jest sygnał na wodospadzie
        mask = np.abs(freqs - carrier_hint) <= hint_bw
    elif search_bw is not None:
        mask = np.abs(freqs) <= search_bw / 2
    else:
        # Szukaj w środkowych 80% pasma (unikamy artefaktów krawędzi filtra)
        mask = np.abs(freqs) <= sample_rate * 0.40

    # ── Wyklucz obszary sideband ────────────────────────────────────────────
    # PCM/PM/NRZ: sidebands przy ±dr, ±3dr, ±5dr od nośnej
    # Dla bezpieczeństwa wykluczamy tylko ±dr (pierwsze, najsilniejsze)
    if excl_sidebands:
        for dr in ORION_DATA_RATES_HZ:
            guard = SIDEBAND_GUARD_HZ + dr * 0.05
            for sign in (+1, -1):
                mask[np.abs(freqs - sign * dr) < guard] = False

    if not np.any(mask):
        # Fallback: zniesienie maski sideband (może nie być danych w tym pasmie)
        mask = np.abs(freqs) <= sample_rate * 0.40

    psd_m = np.where(mask, psd, 0.0)
    peak  = int(np.argmax(psd_m))

    # Subbin accuracy przez interpolację paraboliczną
    raw_offset = _parabolic(psd, peak, fft_size, sample_rate)
    # fftfreq convention: bin > N/2 → ujemne częstotliwości
    if peak > fft_size // 2:
        raw_offset -= sample_rate

    # OQPSK: sygnał był na 4×Δf → podziel przez 4
    if oqpsk:
        raw_offset /= 4.0

    freq_abs = center_freq + raw_offset

    # ── SNR ─────────────────────────────────────────────────────────────────
    # Sygnał: kilka binów wokół szczytu (≈ szerokość nośnej CW)
    sig_w = max(3, int(200.0 / bin_hz))
    sig_p = float(np.mean(psd[max(0,peak-sig_w) : peak+sig_w+1]))

    # Szum: mediana dalszych binów (odporność na inne sygnały)
    noise_mask = mask.copy()
    excl_w = max(sig_w * 8, int(2000.0 / bin_hz))
    noise_mask[max(0,peak-excl_w) : peak+excl_w+1] = False

    noise_p = float(np.median(psd[noise_mask])) if np.any(noise_mask) \
              else float(np.median(psd[mask]))

    if noise_p > 0 and sig_p > noise_p:
        snr_db = 10.0 * math.log10(sig_p / noise_p)
    else:
        snr_db = 0.0

    return freq_abs, snr_db


# ─────────────────────────────────────────────────────────────────────────────
# Interaktywna diagnostyka fazy próbkowania
# ─────────────────────────────────────────────────────────────────────────────

def _interactive_probe(probe_raw, rejected_snrs, probe_n,
                       min_snr_db, carrier_hint, n_welch_sub, center_freq):
    """
    Analizuje wyniki pierwszych probe_n bloków i pyta użytkownika o korektę.

    Args:
        probe_raw     : lista (t, freq_abs, snr) – WSZYSTKIE bloki próbki
        rejected_snrs : lista snr bloków poniżej progu
        probe_n       : liczba bloków próbki
        min_snr_db    : bieżący próg SNR
        carrier_hint  : bieżący carrier hint (lub None)
        n_welch_sub   : bieżąca liczba sub-bloków Welcha
        center_freq   : częstotliwość centralna [Hz]

    Returns:
        dict z nowymi parametrami, np. {'min_snr_db': 1.5} lub {}
    """
    n_ok = len([r for r in probe_raw if r[2] >= min_snr_db])
    accept_rate = n_ok / max(probe_n, 1)

    SEP = "─" * 54
    print(f"\n  {SEP}")
    print(f"  Diagnostyka po {probe_n} blokach próbkowania:")
    print(f"  Akceptacja : {n_ok}/{probe_n} ({accept_rate*100:.0f}%)", end="")

    if accept_rate >= 0.70:
        print("  ✓ OK")
        print(f"  {SEP}\n")
        return {}

    print("  ← niska!")

    ok_snrs = [r[2] for r in probe_raw if r[2] >= min_snr_db]
    if ok_snrs:
        ok_offsets = [r[1] - center_freq for r in probe_raw if r[2] >= min_snr_db]
        print(f"  SNR ok     : min={min(ok_snrs):.1f}  avg={sum(ok_snrs)/len(ok_snrs):.1f}"
              f"  max={max(ok_snrs):.1f} dB")
        print(f"  Offset ok  : avg={sum(ok_offsets)/len(ok_offsets):+.0f} Hz  "
              f"drift={max(ok_offsets)-min(ok_offsets):.1f} Hz")
    if rejected_snrs:
        avg_rej = sum(rejected_snrs) / len(rejected_snrs)
        print(f"  SNR odrzuc.: avg={avg_rej:.1f} dB  (próg={min_snr_db:.1f} dB)")

    suggestions = []

    # Sugestia 1: obniż próg SNR
    if rejected_snrs:
        avg_rej = sum(rejected_snrs) / len(rejected_snrs)
        if avg_rej > min_snr_db * 0.4 and min_snr_db > 1.5:
            new_snr = max(1.0, round(avg_rej * 0.85, 1))
            suggestions.append({
                'desc': f"Obniż próg SNR: {min_snr_db:.1f} → {new_snr:.1f} dB",
                'param': 'min_snr_db', 'new': new_snr,
            })

    # Sugestia 2: zwiększ Welch sub-bloki
    if n_welch_sub < 100 and accept_rate < 0.5:
        new_sub = min(200, n_welch_sub * 4)
        gain = 10 * math.log10(new_sub / n_welch_sub)
        suggestions.append({
            'desc': f"Zwiększ Welch sub-bloki: {n_welch_sub} → {new_sub} (+{gain:.0f} dB SNR)",
            'param': 'n_welch_sub', 'new': new_sub,
        })

    # Sugestia 3: podaj carrier-hint
    if carrier_hint is None and accept_rate < 0.2:
        suggestions.append({
            'desc': "Podaj przybliżony offset nośnej od center [Hz] jako carrier-hint",
            'param': 'carrier_hint', 'new': None,  # wartość podana przez użytkownika
        })

    if not suggestions:
        print(f"\n  Brak konkretnych sugestii – kontynuuję z obecnymi parametrami.")
        print(f"  {SEP}\n")
        return {}

    print(f"\n  Proponowane zmiany (wybierz numer lub Enter = bez zmian):")
    for k, s in enumerate(suggestions, 1):
        print(f"  [{k}] {s['desc']}")
    print(f"  [0] Kontynuuj bez zmian")

    try:
        choice = input(f"\n  Twój wybór (0-{len(suggestions)}): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        print(f"  {SEP}\n")
        return {}

    if not choice or choice == '0':
        print(f"  {SEP}\n")
        return {}

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(suggestions):
            s = suggestions[idx]
            if s['new'] is None:  # carrier_hint – ask for value
                try:
                    val_str = input("  Podaj offset nośnej od center [Hz] (np. -15000): ").strip()
                    s['new'] = float(val_str)
                except (ValueError, EOFError):
                    print("  Niepoprawna wartość – bez zmian.")
                    print(f"  {SEP}\n")
                    return {}
            print(f"  ✓ Zastosowano: {s['desc']}")
            print(f"  {SEP}\n")
            return {s['param']: s['new']}
    except ValueError:
        pass

    print(f"  {SEP}\n")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Przetwarzanie blokowe
# ─────────────────────────────────────────────────────────────────────────────

def process_iq(
    iq, sample_rate, center_freq, start_time,
    integration_sec = 1.0,
    fft_size        = 65536,
    n_welch_sub     = 20,
    min_snr_db      = 3.0,
    search_bw       = None,
    carrier_hint    = None,
    hint_bw         = 50_000,
    excl_sidebands  = True,
    interactive     = True,
    oqpsk           = False,
    auto            = False,
):
    """
    Przesuwa okno integracji i zbiera pomiary częstotliwości nośnej.

    Dla każdego bloku czasu:
      - Welch PSD → SNR
      - Filtr SNR
      - Timestamp = KONIEC okna (INTEGRATION_REF = END, wymóg NASA)

    Returns: lista (datetime, freq_hz, snr_db)
    """
    spb    = int(sample_rate * integration_sec)   # samples per block
    eff_fft = min(fft_size, spb // 2)
    eff_fft = max(1024, 2 ** int(math.log2(eff_fft)))
    n_blocks = len(iq) // spb
    bin_hz   = sample_rate / eff_fft

    if search_bw is None:
        search_bw = sample_rate * 0.80

    snr_gain = 10 * math.log10(max(1, n_welch_sub))

    print(f"\n{'='*64}")
    print(f"  Probki IQ        : {len(iq):,}")
    print(f"  Sample rate      : {sample_rate/1e6:.3f} Msps")
    print(f"  Center freq      : {center_freq/1e6:.6f} MHz")
    print(f"  Integracja       : {integration_sec:.1f} s ({spb:,} probek/blok)")
    print(f"  FFT size         : {eff_fft:,}  ({bin_hz:.2f} Hz/bin)")
    print(f"  Welch sub-bloki  : {n_welch_sub}  (zysk SNR ~{snr_gain:.1f} dB)")
    print(f"  Min SNR          : {min_snr_db:.1f} dB")
    print(f"  Pasmo szukania   : +/-{search_bw/1e3:.0f} kHz")
    if carrier_hint is not None:
        print(f"  Carrier hint     : {carrier_hint:+.0f} Hz od center")
    print(f"  Wyklucz sidebands: {'TAK' if excl_sidebands else 'NIE'}")
    if auto:
        print(f"  Tryb             : AUTO (nośna → OQPSK fallback)")
    elif oqpsk:
        print(f"  Tryb             : OQPSK (IQ^4, /4)")
    else:
        print(f"  Tryb             : carrier (Welch)")
    print(f"  Bloków            : {n_blocks}")
    print(f"{'='*64}")

    measurements = []
    skipped = 0
    rejected_snrs = []   # SNR odrzuconych bloków (poniżej progu)
    probe_raw = []       # (t, freq_abs, snr) – wszystkie bloki fazy próbkowania
    n_carrier_mode = 0   # bloki wykryte przez nośną (auto)
    n_oqpsk_mode   = 0   # bloki wykryte przez OQPSK (auto)

    use_tty = interactive and sys.stdout.isatty() and sys.stdin.isatty()
    probe_n = min(20, max(10, n_blocks // 10))
    probe_done = (not interactive) or (probe_n >= n_blocks)

    t0_proc = time.time()

    for i in range(n_blocks):
        block = iq[i*spb : (i+1)*spb]
        # Timestamp = koniec okna integracji
        t = start_time + timedelta(seconds=(i + 1) * integration_sec)

        try:
            if auto:
                # Próba 1: szukaj nośnej
                freq_abs, snr = estimate_carrier(
                    block, sample_rate, center_freq,
                    fft_size=eff_fft, n_sub=n_welch_sub,
                    search_bw=search_bw, carrier_hint=carrier_hint,
                    hint_bw=hint_bw, excl_sidebands=excl_sidebands,
                    oqpsk=False,
                )
                block_mode = 'C'
                if snr < min_snr_db:
                    # Próba 2: OQPSK (IQ^4) — wyższy próg SNR żeby uniknąć fałszywych detekcji
                    freq_q, snr_q = estimate_carrier(
                        block, sample_rate, center_freq,
                        fft_size=eff_fft, n_sub=n_welch_sub,
                        search_bw=search_bw, carrier_hint=carrier_hint,
                        hint_bw=hint_bw, excl_sidebands=False,
                        oqpsk=True,
                    )
                    if snr_q >= min_snr_db + 2.0:   # +2 dB margines dla OQPSK
                        freq_abs, snr, block_mode = freq_q, snr_q, 'Q'
            else:
                freq_abs, snr = estimate_carrier(
                    block, sample_rate, center_freq,
                    fft_size=eff_fft, n_sub=n_welch_sub,
                    search_bw=search_bw, carrier_hint=carrier_hint,
                    hint_bw=hint_bw, excl_sidebands=excl_sidebands,
                    oqpsk=oqpsk,
                )
                block_mode = 'Q' if oqpsk else 'C'
        except Exception as e:
            if use_tty:
                print(f"\r  [ERR] blok {i+1:4d}: {e}     ")
            else:
                print(f"  [ERR] blok {i+1:4d}: {e}", file=sys.stderr)
            skipped += 1
            continue

        offset = freq_abs - center_freq
        accepted = snr >= min_snr_db

        if accepted:
            measurements.append((t, freq_abs, snr))
            if block_mode == 'C':
                n_carrier_mode += 1
            else:
                n_oqpsk_mode += 1
        else:
            skipped += 1
            rejected_snrs.append(snr)

        # Zbierz dane fazy próbkowania (przed diagnozą)
        if not probe_done:
            probe_raw.append((t, freq_abs, snr))

        # ── Pasek postępu (TTY) ───────────────────────────────────────────
        if use_tty:
            elapsed = time.time() - t0_proc
            if i > 0 and elapsed > 0:
                eta_s = elapsed / (i + 1) * (n_blocks - i - 1)
                eta_str = f"ETA {int(eta_s//60):02d}:{int(eta_s%60):02d}"
            else:
                eta_str = "ETA --:--"
            n_ok = len(measurements)
            accept_pct = n_ok / (i + 1) * 100
            status = "✓" if accepted else "✗"
            bw = 28
            filled = int(bw * (i + 1) / n_blocks)
            bar = '█' * filled + '░' * (bw - filled)
            print(f"\r  {status} [{bar}] {i+1}/{n_blocks} | "
                  f"ok:{n_ok}({accept_pct:.0f}%) | "
                  f"off:{offset:+.0f}Hz | SNR:{snr:.1f}dB | {eta_str}   ",
                  end='', flush=True)
        else:
            # Nie-TTY: linie co jakiś czas (zachowane oryginalne zachowanie)
            mode_tag = f'[{block_mode}]' if auto else '[OK]'
            if accepted:
                if len(measurements) <= 5 or (i+1) % 30 == 0 or i == n_blocks-1:
                    print(f"  {mode_tag}  {i+1:4d}/{n_blocks}  {_dt_to_tdm(t)}  "
                          f"offset={offset:+10.2f} Hz  SNR={snr:5.1f} dB")
            else:
                if len(rejected_snrs) <= 3 or (i+1) % 60 == 0:
                    print(f"  [--]  {i+1:4d}/{n_blocks}  "
                          f"offset={offset:+10.2f} Hz  SNR={snr:5.1f} dB  <-- ponizej progu")

        # ── Diagnostyka po fazie próbkowania ──────────────────────────────
        if not probe_done and (i + 1) == probe_n:
            probe_done = True
            if use_tty:
                print()  # newline po pasku postępu
            new_params = _interactive_probe(
                probe_raw, rejected_snrs[:], probe_n,
                min_snr_db, carrier_hint, n_welch_sub, center_freq,
            )
            if new_params:
                if 'min_snr_db' in new_params:
                    min_snr_db = new_params['min_snr_db']
                    # Przelicz bloki próbki z nowym progiem
                    measurements = [(t2, f, s) for t2, f, s in probe_raw if s >= min_snr_db]
                    skipped = sum(1 for _, _, s in probe_raw if s < min_snr_db)
                    rejected_snrs = [s for _, _, s in probe_raw if s < min_snr_db]
                    print(f"  Przeliczono {len(probe_raw)} bloków próbki → {len(measurements)} ok")
                if 'n_welch_sub' in new_params:
                    n_welch_sub = new_params['n_welch_sub']
                    snr_gain_new = 10 * math.log10(max(1, n_welch_sub))
                    print(f"  Nowe Welch sub-bloki: {n_welch_sub} (zysk ~{snr_gain_new:.1f} dB)")
                if 'carrier_hint' in new_params:
                    carrier_hint = new_params['carrier_hint']
                    print(f"  Nowy carrier hint: {carrier_hint:+.0f} Hz")
                print(f"  Kontynuuję od bloku {i+2}/{n_blocks}...\n")

    if use_tty:
        print()  # newline po ostatnim pasku postępu

    print(f"\n  Zaakceptowane: {len(measurements)}/{n_blocks}  (pominiete: {skipped})")

    if measurements:
        offsets = [m[1]-center_freq for m in measurements]
        snrs    = [m[2] for m in measurements]
        print(f"  Offset nośnej : min={min(offsets):+.1f}  max={max(offsets):+.1f}  "
              f"drift={max(offsets)-min(offsets):.1f} Hz")
        print(f"  SNR           : min={min(snrs):.1f}  max={max(snrs):.1f}  "
              f"sred={sum(snrs)/len(snrs):.1f} dB")
        if auto:
            print(f"  Tryby detekcji: carrier={n_carrier_mode}  OQPSK={n_oqpsk_mode}")

    return measurements


# ─────────────────────────────────────────────────────────────────────────────
# Zapis TDM
# ─────────────────────────────────────────────────────────────────────────────

def write_tdm(measurements, output_path, station_name, center_freq_hz,
              integration_sec, originator=None, dsn_station=None, comment=None,
              participant_1=None):
    """
    Zapisuje plik TDM CCSDS v2.0 KVN zgodny z FDSS-III-109-0044.
    """
    if not measurements:
        raise ValueError("Brak pomiarow do zapisania")

    freq_offset = round(center_freq_hz)   # FREQ_OFFSET w Hz (calkow.)
    t_start = measurements[0][0]
    t_stop  = measurements[-1][0]
    now_utc = datetime.now(timezone.utc)
    orig    = originator or station_name

    p1 = participant_1 or "ORION"
    if dsn_station:
        parts      = [f"PARTICIPANT_1          = {dsn_station}",
                      f"PARTICIPANT_2          = {p1}",
                      f"PARTICIPANT_3          = {station_name}"]
        path_str   = "1,2,3"
        data_kw    = "RECEIVE_FREQ_3"
    else:
        parts      = [f"PARTICIPANT_1          = {p1}",
                      f"PARTICIPANT_2          = {station_name}"]
        path_str   = "1,2"
        data_kw    = "RECEIVE_FREQ_2"

    lines = [
        "CCSDS_TDM_VERS = 2.0",
        f"CREATION_DATE  = {_dt_to_tdm(now_utc)}Z",
        f"ORIGINATOR     = {orig}",
        "",
    ]
    if comment:
        for ln in comment.strip().splitlines():
            lines.append(f"COMMENT {ln}")
        lines.append("")

    lines += [
        "META_START",
        "TIME_SYSTEM            = UTC",
        *parts,
        "MODE                   = SEQUENTIAL",
        f"PATH                   = {path_str}",
        f"INTEGRATION_INTERVAL   = {integration_sec:.1f}",
        "INTEGRATION_REF        = END",
        f"FREQ_OFFSET            = {freq_offset:.1f}",
        f"START_TIME             = {_dt_to_tdm(t_start)}",
        f"STOP_TIME              = {_dt_to_tdm(t_stop)}",
        f"TURNAROUND_NUMERATOR   = {TURNAROUND_NUMERATOR}",
        f"TURNAROUND_DENOMINATOR = {TURNAROUND_DENOMINATOR}",
        "META_STOP",
        "",
        "DATA_START",
    ]
    for (t, fa, snr) in measurements:
        lines.append(f"{data_kw} = {_dt_to_tdm(t)}  {fa - freq_offset:+.3f}")
    lines += ["DATA_STOP", ""]

    with open(output_path, "w", encoding="ascii") as f:
        f.write("\n".join(lines))

    dur = (t_stop - t_start).total_seconds()
    print(f"\n{'='*64}")
    print(f"  ZAPISANO TDM : {output_path}")
    print(f"  Pomiarow     : {len(measurements)}")
    print(f"  Czas         : {dur:.0f} s ({dur/60:.1f} min)")
    print(f"  Tryb         : {'3-way ('+dsn_station+')' if dsn_station else '1-way'}")
    print(f"{'='*64}")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostyczny wykres widma (opcjonalny, wymaga matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectrum(iq, sample_rate, center_freq, output_png,
                 fft_size=65536, n_sub=50, duration_sec=60.0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [INFO] matplotlib niedostepny – pomijam wykres")
        return

    n = min(len(iq), int(sample_rate * duration_sec))
    print(f"  Generuje widmo ({n/sample_rate:.0f} s, FFT={fft_size}, sub={n_sub})...")

    psd   = welch_psd(iq[:n], fft_size, n_sub)
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0/sample_rate)) / 1e3  # kHz
    psd_s = 10 * np.log10(np.fft.fftshift(psd) + 1e-20)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(freqs, psd_s, lw=0.5, color="steelblue")
    ax.set_xlabel("Offset od center [kHz]")
    ax.set_ylabel("PSD [dB rel.]")
    ax.set_title(
        f"Welch PSD  |  {center_freq/1e6:.4f} MHz  |  "
        f"{sample_rate/1e6:.1f} Msps  |  "
        f"int={n/sample_rate:.0f}s  sub={n_sub}"
    )
    ax.grid(True, alpha=0.3)

    colors = ["orange", "red", "purple", "brown"]
    for dr, col in zip(ORION_DATA_RATES_HZ, colors):
        for sign in (+1, -1):
            ax.axvline(sign*dr/1e3, color=col, lw=0.8, ls="--", alpha=0.5,
                       label=f"±{dr/1e3:.0f}k sideband" if sign==1 else None)
    ax.axvline(0, color="lime", lw=1.0, ls=":", label="DC/center")
    ax.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    plt.savefig(str(output_png), dpi=120)
    plt.close()
    print(f"  Widmo: {output_png}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True,
                   help=".sigmf-meta  LUB  .raw/.bin/.iq (GQRX)")
    p.add_argument("--freq",  type=float, help="Czestotliwosc centralna [Hz]")
    p.add_argument("--rate",  type=float, help="Sample rate [Sps]")
    p.add_argument("--start", type=str,   help="Start nagrania (ISO-8601 UTC)")
    p.add_argument("--dtype", type=str,
                   choices=["cf32_le","cf32_be","ci16_le","ci16_be","ci8","cu8","cf64_le"],
                   help="Typ danych IQ (domyslnie: cf32_le)")
    p.add_argument("--station",    "-s", required=True,
                   help="Nazwa stacji (np. SP5ABC_YAGI9M)")
    p.add_argument("--originator", type=str)
    p.add_argument("--dsn-station", type=str,
                   help="Stacja DSN uplink (np. DSS-26) → tryb 3-way")
    p.add_argument("--integration", type=float, default=1.0,
                   help="Czas integracji [s] (NASA: 1 lub 10, domyslnie: 1)")
    p.add_argument("--fft-size",    type=int,   default=65536,
                   help="Rozmiar FFT (domyslnie: 65536)")
    p.add_argument("--welch-sub",   type=int,   default=20,
                   help=(
                       "Liczba sub-blokow Welcha [domyslnie: 20]. "
                       "Wiecej = lepszy SNR dla slabych sygnalow. "
                       "Przy malej antenie sprobuj 50-200."
                   ))
    p.add_argument("--min-snr",     type=float, default=3.0,
                   help="Min SNR [dB] do akceptacji pomiaru (domyslnie: 3.0)")
    p.add_argument("--search-bw",   type=float, default=None,
                   help="Pasmo wyszukiwania nosnej [Hz]")
    p.add_argument("--carrier-hint", type=float, default=None,
                   help=(
                       "Przyblizony offset nosnej [Hz] od center. "
                       "Jesli wiesz gdzie jest sygnal na wodospadzie – "
                       "podaj tu (np. --carrier-hint -15000 = 15 kHz ponizej center). "
                       "Bardzo pomaga przy slabym sygnale!"
                   ))
    p.add_argument("--hint-bw", type=float, default=50_000,
                   help=(
                       "Polszerokos pasma wyszukiwania wokol --carrier-hint [Hz] "
                       "(domyslnie 50000 = ±50 kHz). Zwez jesli w poblizu nośnej "
                       "sa silne sidebands, np. --hint-bw 15000."
                   ))
    p.add_argument("--no-excl-sidebands", action="store_true",
                   help="NIE wyklucz obszarow sideband przy szukaniu nosnej")
    p.add_argument("--oqpsk", action="store_true",
                   help=(
                       "Tryb OQPSK/suppressowana nośna (Artemis II). "
                       "Podnosi IQ do 4. potęgi przed Welchem (M-th power carrier recovery), "
                       "usuwa modulację QPSK i ujawnia nośną jako CW na 4×Δf. "
                       "Wynik dzielony przez 4. Użyj z --no-excl-sidebands."
                   ))
    p.add_argument("--auto", action="store_true",
                   help=(
                       "Auto-detekcja modulacji. Dla każdego bloku próbuje najpierw "
                       "wykryć nośną CW (KPLO, LRO, Artemis I), a jeśli SNR za niski — "
                       "przełącza na tryb OQPSK IQ^4 (Artemis II). "
                       "Przydatne gdy nie wiadomo jakiego sygnału się spodziewać."
                   ))
    p.add_argument("--max-samples",  type=int,  default=None,
                   help="Max probek do zaladowania (do testow na duzych plikach)")
    p.add_argument("--output",  "-o", default=None, help="Plik wyjsciowy TDM")
    p.add_argument("--participant-1", type=str, default=None,
                   help="Spacecraft name for PARTICIPANT_1 (default: ORION)")
    p.add_argument("--comment", type=str)
    p.add_argument("--plot", action="store_true",
                   help="Zapisz widmo Welcha do PNG (wymaga matplotlib)")
    p.add_argument("--no-interactive", action="store_true",
                   help="Wyłącz interaktywną diagnostykę i pasek postępu (dla skryptów/cron)")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"BLAD: Nie znaleziono: {inp}")

    # ── Wykryj format ──────────────────────────────────────────────────────
    suffix   = inp.suffix.lower()
    is_sigmf = suffix == ".sigmf-meta"
    is_gqrx  = suffix in (".raw", ".bin", ".iq")
    if not is_sigmf and not is_gqrx:
        meta_g = inp.with_suffix(".sigmf-meta")
        if meta_g.exists():
            inp, is_sigmf = meta_g, True
        else:
            is_gqrx = True

    info = {}

    if is_sigmf:
        data_path = inp.with_suffix(".sigmf-data")
        if not data_path.exists():
            sys.exit(f"BLAD: Brak pliku danych: {data_path}")
        print(f"[SigMF] {inp.name}")
        info = read_sigmf_meta(inp)
        print(f"  Datatype  : {info['datatype']}")
        print(f"  Rate      : {info['sample_rate']/1e6:.3f} Msps")
        print(f"  Freq      : {info['center_freq']/1e6:.6f} MHz")
        if info["start_time"]:
            print(f"  Start     : {info['start_time'].isoformat()}")
        print(f"\nLaduje IQ z: {data_path}")
        iq = load_iq(data_path, info["datatype"], args.max_samples)
    else:
        data_path = inp
        fn_info   = parse_gqrx_filename(inp.name)
        info.update(fn_info)
        dt_str = args.dtype or "cf32_le"
        info.setdefault("datatype", dt_str)
        print(f"[GQRX] {inp.name}")
        if fn_info:
            print(f"  Z nazwy pliku: freq={info.get('center_freq',0)/1e6:.3f} MHz  "
                  f"rate={info.get('sample_rate',0)/1e6:.3f} Msps  "
                  f"start={info.get('start_time','?')}")
        print(f"\nLaduje IQ z: {data_path}")
        iq = load_iq(data_path, info["datatype"], args.max_samples)

    # ── Override z CLI ─────────────────────────────────────────────────────
    if args.freq:  info["center_freq"] = args.freq
    if args.rate:  info["sample_rate"] = args.rate
    if args.start: info["start_time"]  = _parse_dt(args.start)

    missing = [k for k in ("center_freq","sample_rate","start_time") if not info.get(k)]
    if missing:
        sys.exit(
            "BLAD – brakuje metadanych: " + ", ".join(missing) +
            "\nUzyj: --freq, --rate, --start"
        )

    cf, sr, t0 = info["center_freq"], info["sample_rate"], info["start_time"]

    # ── Opcjonalny wykres ─────────────────────────────────────────────────
    if args.plot:
        plot_out = Path(args.output or f"{args.station}_spectrum").with_suffix(".png")
        plot_spectrum(iq, sr, cf, plot_out,
                      fft_size=min(65536, 2**int(math.log2(max(1024, int(sr*30))))),
                      n_sub=50,
                      duration_sec=min(120.0, len(iq)/sr))

    # ── Główne przetwarzanie ───────────────────────────────────────────────
    meas = process_iq(
        iq, sr, cf, t0,
        integration_sec = args.integration,
        fft_size        = args.fft_size,
        n_welch_sub     = args.welch_sub,
        min_snr_db      = args.min_snr,
        search_bw       = args.search_bw,
        carrier_hint    = args.carrier_hint,
        hint_bw         = args.hint_bw,
        excl_sidebands  = not args.no_excl_sidebands,
        interactive     = not args.no_interactive,
        oqpsk           = args.oqpsk,
        auto            = args.auto,
    )

    if not meas:
        print("\nBLAD: Brak pomiarow – sygnal za slaby lub zly prog SNR.")
        print("Sprobuj: --min-snr 1.5  --welch-sub 100  --carrier-hint <Hz>")
        sys.exit(1)

    # ── Zapis TDM ─────────────────────────────────────────────────────────
    out = Path(args.output) if args.output else \
          Path(f"{args.station}_{t0.strftime('%Y%m%d_%H%M%S')}.tdm")

    mode_str = "OQPSK (M-th power /4)" if args.oqpsk else "carrier (Welch)"
    auto_cmt = (
        f"Artemis II one-way Doppler tracking\n"
        f"Source: {inp.name}\n"
        f"HW: {info.get('hw','?')} | "
        f"FFT={args.fft_size} Welch={args.welch_sub} int={args.integration}s | "
        f"mode={mode_str}"
    )
    write_tdm(meas, out, args.station, cf, args.integration,
              args.originator, args.dsn_station, args.comment or auto_cmt,
              participant_1=args.participant_1)

    print(f"\nNazwa wg konwencji NASA:")
    print(f"  {args.station}_Antenna1_{t0.strftime('%Y%m%d%H%M%S')}.tdm")
    print(f"\nAby przeslac plik do NASA, skontaktuj sie z programem Artemis Amateur Tracking.")


if __name__ == "__main__":
    main()
