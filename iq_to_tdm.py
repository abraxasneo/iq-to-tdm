#!/usr/bin/env python3
"""
IQ -> NASA CCSDS TDM Converter  (Welch averaging for weak signals)
==================================================================
Converts SDR IQ recordings (SigMF, GQRX, or WAV) to a NASA CCSDS TDM v2.0 file
for Artemis II / lunar mission one-way Doppler tracking.

Key features:
  - Welch averaged periodogram for low-SNR carrier detection (+13 dB with default 20 sub-blocks)
  - Searches for the narrow residual carrier (PCM/PM/NRZ), not sidebands
  - --oqpsk mode: IQ^4 suppressed-carrier recovery for OQPSK signals (Artemis II)
  - --auto mode: per-block automatic selection between CW and OQPSK
  - Works with small antennas (120 cm dish and up)

Usage:
  # SigMF (recommended):
  python iq_to_tdm.py --input recording.sigmf-meta --station MY_CALL --auto

  # WAV IQ (SDR Console, SDR#, HDSDR, SDRuno):
  python iq_to_tdm.py --input SDRSharp_20260210_120000Z_2216500000Hz_IQ.wav --station MY_CALL

  # GQRX raw (freq and rate from filename):
  python iq_to_tdm.py --input gqrx_20260210_120000_2216500000_2000000_fc.raw --station MY_CALL

  # GQRX raw (manual parameters):
  python iq_to_tdm.py --input recording.raw \\
      --freq 2216500000 --rate 2000000 --start "2026-02-10T12:00:00Z" \\
      --station MY_CALL

  # Weak signal -- more averaging:
  python iq_to_tdm.py --input recording.sigmf-meta --station MY_CALL \\
      --integration 10 --welch-sub 50 --min-snr 2.0

  # Known carrier position (e.g. 15 kHz below center):
  python iq_to_tdm.py --input recording.sigmf-meta --station MY_CALL \\
      --carrier-hint -15000
"""

import argparse
import json
import math
import numpy as np
import os
import re
import struct
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# NASA / S-band constants
# ---------------------------------------------------------------------------

TURNAROUND_NUMERATOR   = 240   # S-band coherent turnaround ratio
TURNAROUND_DENOMINATOR = 221

# Orion data rates -> sideband positions relative to carrier
ORION_DATA_RATES_HZ = [72_000, 2_000_000, 4_000_000, 6_000_000]
# Guard band around each sideband (Hz) -- excluded from carrier search
SIDEBAND_GUARD_HZ = 5_000


# ---------------------------------------------------------------------------
# SigMF parsing
# ---------------------------------------------------------------------------

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


class _LazyIntIQ:
    """
    Lazy int->complex64 adapter for large IQ files.

    Keeps only the raw int16/int8/uint8 memmap in memory and converts
    each slice on-the-fly.  Avoids allocating 2–3× the file size in RAM
    when converting ci16/ci8/cu8 recordings upfront.

    Supports len() and slice indexing — the only operations used by
    process_iq().
    """
    def __init__(self, raw, datatype):
        self._raw = raw       # int memmap, length = 2 × n_complex_samples
        self._dt  = datatype  # 'ci16_le', 'ci8', 'cu8', …

    def __len__(self):
        return len(self._raw) // 2

    def __getitem__(self, sl):
        if isinstance(sl, slice):
            start, stop, _ = sl.indices(len(self))
        else:
            start, stop = int(sl), int(sl) + 1
        chunk = self._raw[start * 2 : stop * 2]
        if self._dt.startswith('ci16'):
            return (chunk[0::2].astype(np.float32)
                    + 1j * chunk[1::2].astype(np.float32)) / 32768.0
        elif self._dt.startswith('ci8'):
            return (chunk[0::2].astype(np.float32)
                    + 1j * chunk[1::2].astype(np.float32)) / 128.0
        else:  # cu8
            f = chunk.astype(np.float32) - 127.5
            return (f[0::2] + 1j * f[1::2]).astype(np.complex64)


def load_iq(data_path, datatype, max_samples=None, skip_samples=None,
            data_offset=0):
    """
    Load IQ file and return complex64 array (or lazy wrapper for large int files).
    Uses np.memmap for files larger than 2 GB to avoid loading everything into RAM.
    For large ci16/ci8/cu8 files the raw memmap is wrapped in _LazyIntIQ so that
    only the current integration block is converted at a time (saves 2–3× peak RAM).

    data_offset: byte offset where IQ data begins (>0 for WAV files).
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
        raise ValueError(f"Unsupported datatype: {dt}")

    file_size = os.path.getsize(str(data_path))
    elem_dtype = raw_map[dt]
    elem_size  = np.dtype(elem_dtype).itemsize
    data_bytes = file_size - data_offset
    n_elems    = data_bytes // elem_size
    skip_elems = (skip_samples * 2) if skip_samples else 0
    if max_samples:
        n_elems = min(n_elems - skip_elems, max_samples * 2)
    else:
        n_elems = n_elems - skip_elems

    LARGE_FILE_BYTES = 2 * 1024 ** 3  # 2 GB
    if file_size > LARGE_FILE_BYTES:
        raw = np.memmap(str(data_path), dtype=elem_dtype, mode='r',
                        offset=data_offset + skip_elems * np.dtype(elem_dtype).itemsize,
                        shape=(n_elems,))
        # For integer types, avoid converting the whole file to complex64 upfront:
        # a 14 GB ci16 file would require ~29 GB of RAM for the conversion.
        # Return a lazy wrapper that converts one integration block at a time.
        if not dt.startswith('cf'):
            return _LazyIntIQ(raw, dt)
    else:
        raw = np.fromfile(str(data_path), dtype=elem_dtype, offset=data_offset)
        raw = raw[skip_elems : skip_elems + n_elems]

    if dt.startswith("cf32"):
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
        raise ValueError(f"Cannot load datatype: {dt}")

    return iq


# ---------------------------------------------------------------------------
# GQRX filename parsing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# WAV / RF64 IQ parsing  (SDR Console, SDR#, HDSDR, SDRuno)
# ---------------------------------------------------------------------------

def parse_wav_iq(wav_path):
    """
    Parse a 2-channel (I/Q) WAV or RF64 file.

    Extracts metadata from:
      - fmt  chunk : sample_rate, bits_per_sample, format_tag
      - auxi chunk : center_freq, start_time (SDR Console XML or SDRuno binary)
      - filename   : center_freq fallback (SDR Console / SDR# / HDSDR patterns)
      - ds64 chunk : true data size for RF64 files >4 GB

    Returns dict: datatype, sample_rate, center_freq, start_time, hw,
                  data_offset (bytes), data_size (bytes).
    """
    with open(wav_path, "rb") as f:
        head = f.read(12)
        if len(head) < 12:
            raise ValueError("WAV file too short")
        riff_id = head[0:4]
        wave_id = head[8:12]
        if wave_id != b"WAVE":
            raise ValueError(f"Not a WAVE file (got {wave_id!r})")
        is_rf64 = (riff_id == b"RF64")
        if riff_id not in (b"RIFF", b"RF64"):
            raise ValueError(f"Not RIFF/RF64 (got {riff_id!r})")

        ds64_data_size = None
        fmt_data = None
        auxi_data = None
        data_offset = None
        data_size = None

        while True:
            ch = f.read(8)
            if len(ch) < 8:
                break
            cid = ch[0:4]
            csz = struct.unpack("<I", ch[4:8])[0]
            pos = f.tell()

            if cid == b"ds64":
                raw = f.read(min(csz, 28))
                if len(raw) >= 16:
                    ds64_data_size = struct.unpack("<Q", raw[8:16])[0]
                f.seek(pos + csz + (csz % 2))
            elif cid == b"fmt ":
                fmt_data = f.read(csz)
                if csz % 2:
                    f.read(1)
            elif cid == b"auxi":
                auxi_data = f.read(csz)
                if csz % 2:
                    f.read(1)
            elif cid == b"data":
                data_offset = pos
                data_size = csz
                if is_rf64 and ds64_data_size is not None:
                    data_size = ds64_data_size
                break  # data chunk found — don't read payload
            else:
                f.seek(pos + csz + (csz % 2))

    if fmt_data is None:
        raise ValueError("WAV file has no fmt chunk")
    if data_offset is None:
        raise ValueError("WAV file has no data chunk")

    # -- Parse fmt chunk ----------------------------------------------------
    fmt_tag, n_channels, sample_rate, _, _, bits = \
        struct.unpack("<HHIIHH", fmt_data[:16])
    if n_channels != 2:
        raise ValueError(f"Expected 2 channels (I/Q stereo), got {n_channels}")

    if fmt_tag == 1 and bits == 16:
        datatype = "ci16_le"
    elif fmt_tag == 3 and bits == 32:
        datatype = "cf32_le"
    elif fmt_tag == 1 and bits == 8:
        datatype = "cu8"
    else:
        raise ValueError(
            f"Unsupported WAV format: tag={fmt_tag} bits={bits}")

    info = {
        "datatype":    datatype,
        "sample_rate": float(sample_rate),
        "center_freq": 0.0,
        "start_time":  None,
        "hw":          "",
        "data_offset": data_offset,
        "data_size":   data_size,
    }

    if auxi_data:
        _parse_auxi(auxi_data, info)

    if not info["center_freq"]:
        _parse_wav_filename(str(wav_path), info)

    return info


def _parse_auxi(auxi_data, info):
    """Route auxi chunk to XML or binary parser."""
    for encoding in ("utf-16le", "utf-8"):
        try:
            text = auxi_data.decode(encoding)
            if "<" in text:
                _parse_auxi_xml(text, info)
                return
        except (UnicodeDecodeError, ValueError):
            continue
    # Binary struct (SDRuno: >=40 bytes)
    if len(auxi_data) >= 40:
        _parse_auxi_binary(auxi_data, info)


def _parse_auxi_xml(xml_text, info):
    """Parse SDR Console auxi XML (RadioCenterFreq, UTC, …)."""
    from xml.etree import ElementTree as ET
    try:
        root = ET.fromstring(xml_text.strip())
        defn = root.find(".//Definition")
        attrs = defn.attrib if defn is not None else root.attrib
    except Exception:
        attrs = {}
        for m in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', xml_text):
            attrs[m.group(1)] = m.group(2)

    if "RadioCenterFreq" in attrs:
        try:
            info["center_freq"] = float(attrs["RadioCenterFreq"])
        except ValueError:
            pass
    if "SampleRate" in attrs:
        try:
            sr = float(attrs["SampleRate"])
            if sr > 0:
                info["sample_rate"] = sr
        except ValueError:
            pass

    for key in ("UTC", "CurrentTimeUTC"):
        if key in attrs and attrs[key]:
            try:
                info["start_time"] = _parse_dt(attrs[key])
                break
            except ValueError:
                pass

    if not info.get("start_time") and "UTCSeconds" in attrs:
        try:
            ts = float(attrs["UTCSeconds"])
            info["start_time"] = datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OSError):
            pass

    info["hw"] = attrs.get("Receiver", attrs.get("HW", ""))


def _parse_auxi_binary(auxi_data, info):
    """Parse SDRuno binary auxi struct (SYSTEMTIME + center freq)."""
    year, month, _, day, hour, minute, second, ms = \
        struct.unpack("<8H", auxi_data[0:16])
    if 1970 <= year <= 2100 and 1 <= month <= 12:
        try:
            info["start_time"] = datetime(
                year, month, day, hour, minute, second,
                ms * 1000, tzinfo=timezone.utc)
        except ValueError:
            pass
    center_freq = struct.unpack("<d", auxi_data[32:40])[0]
    if center_freq > 0:
        info["center_freq"] = center_freq


def _parse_wav_filename(name, info):
    """Try to extract center frequency and start time from WAV filename."""
    basename = os.path.basename(name)

    # GQRX: "gqrx_YYYYMMDD_HHMMSS_FREQ[_RATE_fc].wav"
    m = re.search(r'gqrx_(\d{8})_(\d{6})_(\d+?)(?:_(\d+))?(?:_fc)?\.wav',
                  basename, re.IGNORECASE)
    if m:
        info["center_freq"] = float(m.group(3))
        if m.group(4):
            info["sample_rate"] = float(m.group(4))
        d, t = m.group(1), m.group(2)
        try:
            info["start_time"] = datetime(
                int(d[:4]), int(d[4:6]), int(d[6:8]),
                int(t[:2]), int(t[2:4]), int(t[4:6]),
                tzinfo=timezone.utc)
        except ValueError:
            pass
        return

    # SDR Console: "162.550MHz.wav"
    m = re.search(r'([\d.]+)\s*MHz', basename, re.IGNORECASE)
    if m:
        try:
            info["center_freq"] = float(m.group(1)) * 1e6
        except ValueError:
            pass

    # SDR#: "_162550000Hz"
    if not info.get("center_freq"):
        m = re.search(r'[_-](\d+)\s*Hz', basename, re.IGNORECASE)
        if m:
            info["center_freq"] = float(m.group(1))

    # HDSDR: "_162550kHz"
    if not info.get("center_freq"):
        m = re.search(r'[_-](\d+)\s*kHz', basename, re.IGNORECASE)
        if m:
            info["center_freq"] = float(m.group(1)) * 1e3

    # SDR# / HDSDR: "SDRSharp_YYYYMMDD_HHMMSSZ_..."
    if not info.get("start_time"):
        m = re.search(r'(\d{8})[_-](\d{6})Z?[_-]', basename)
        if m:
            d, t = m.group(1), m.group(2)
            try:
                info["start_time"] = datetime(
                    int(d[:4]), int(d[4:6]), int(d[6:8]),
                    int(t[:2]), int(t[2:4]), int(t[4:6]),
                    tzinfo=timezone.utc)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _parse_dt(s):
    s = s.strip().rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                "%d-%m-%Y %H:%M:%S.%f", "%d-%m-%Y %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse datetime: '{s}'")


def _dt_to_tdm(dt):
    """YYYY-DOYThh:mm:ss.mmm  (CCSDS TDM day-of-year format)"""
    doy = dt.timetuple().tm_yday
    return (f"{dt.year}-{doy:03d}T"
            f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}."
            f"{dt.microsecond // 1000:03d}")


# ---------------------------------------------------------------------------
# DSP core: Welch periodogram
# ---------------------------------------------------------------------------

def welch_psd(iq_block, fft_size, n_sub):
    """
    Averaged periodogram using Welch method with 50% overlap.

    Why this works for weak signals:
      Noise is random  -> averaging N spectra reduces it by sqrt(N)
      CW carrier is deterministic -> its power does NOT decrease with averaging
      SNR gain = 10*log10(n_sub) dB
        n_sub=20  -> +13 dB
        n_sub=100 -> +20 dB
        n_sub=500 -> +27 dB

    Args:
        iq_block : complex64 array
        fft_size : single FFT size (should be power of 2)
        n_sub    : max number of sub-blocks to average

    Returns:
        Averaged power spectrum (float64), length fft_size
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
        raise ValueError("Too few samples for Welch PSD -- increase integration or reduce fft_size")

    return psd / count


def _parabolic(psd, k, fft_size, sr):
    """Parabolic interpolation of FFT bin -> more accurate offset [Hz]."""
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
    Find residual PCM/PM/NRZ carrier using Welch method.

    --oqpsk mode (M-th power carrier recovery):
      For OQPSK (suppressed carrier) raise IQ to 4th power.
      QPSK modulation cancels (phases 0/90/180/270 deg * 4 = 0 deg),
      leaving a pure CW line at 4*delta_f. Result divided by 4.

    Search mask logic:
      1. search_bw  -> restrict to central search_bw Hz
      2. carrier_hint -> search only within +/-hint_bw Hz of hint (default +/-50 kHz)
      3. excl_sidebands -> exclude regions near +/-data_rate from center

    Returns:
        (freq_abs_hz, snr_db)
    """
    if oqpsk:
        # Raise to 4th power -- removes OQPSK modulation, leaves CW at 4*delta_f
        iq_proc = (iq_block.astype(np.complex128) ** 4).astype(np.complex64)
    else:
        iq_proc = iq_block

    psd   = welch_psd(iq_proc, fft_size, n_sub)
    freqs = np.fft.fftfreq(fft_size, d=1.0/sample_rate)   # offset [Hz] from center
    bin_hz = sample_rate / fft_size

    # -- Search mask --------------------------------------------------------
    if carrier_hint is not None:
        # Approximate carrier position known from waterfall
        mask = np.abs(freqs - carrier_hint) <= hint_bw
    elif search_bw is not None:
        mask = np.abs(freqs) <= search_bw / 2
    else:
        # Search inner 80% of band (avoid filter edge artifacts)
        mask = np.abs(freqs) <= sample_rate * 0.40

    # -- Exclude sideband regions -------------------------------------------
    # PCM/PM/NRZ: sidebands at +/-dr, +/-3dr, +/-5dr from carrier
    # Only exclude +/-dr (first, strongest)
    if excl_sidebands:
        for dr in ORION_DATA_RATES_HZ:
            guard = SIDEBAND_GUARD_HZ + dr * 0.05
            for sign in (+1, -1):
                mask[np.abs(freqs - sign * dr) < guard] = False

    if not np.any(mask):
        # Fallback: drop sideband mask (may not apply to this band)
        mask = np.abs(freqs) <= sample_rate * 0.40

    psd_m = np.where(mask, psd, 0.0)
    peak  = int(np.argmax(psd_m))

    # Sub-bin accuracy via parabolic interpolation
    raw_offset = _parabolic(psd, peak, fft_size, sample_rate)
    # fftfreq convention: bin > N/2 -> negative frequencies
    if peak > fft_size // 2:
        raw_offset -= sample_rate

    # OQPSK: signal was at 4*delta_f -> divide by 4
    if oqpsk:
        raw_offset /= 4.0

    freq_abs = center_freq + raw_offset

    # -- SNR ----------------------------------------------------------------
    # Signal: a few bins around peak (approx CW carrier width)
    sig_w = max(3, int(200.0 / bin_hz))
    sig_p = float(np.mean(psd[max(0,peak-sig_w) : peak+sig_w+1]))

    # Noise: median of distant bins (robust against other signals)
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


# ---------------------------------------------------------------------------
# Interactive probe-phase diagnostics
# ---------------------------------------------------------------------------

def _interactive_probe(probe_raw, rejected_snrs, probe_n,
                       min_snr_db, carrier_hint, n_welch_sub, center_freq):
    """
    Analyse results of the first probe_n blocks and offer parameter adjustments.

    Args:
        probe_raw     : list of (t, freq_abs, snr) -- ALL probe blocks
        rejected_snrs : list of SNR values for rejected blocks
        probe_n       : number of probe blocks
        min_snr_db    : current SNR threshold
        carrier_hint  : current carrier hint (or None)
        n_welch_sub   : current number of Welch sub-blocks
        center_freq   : center frequency [Hz]

    Returns:
        dict with updated parameters, e.g. {'min_snr_db': 1.5} or {}
    """
    n_ok = len([r for r in probe_raw if r[2] >= min_snr_db])
    accept_rate = n_ok / max(probe_n, 1)

    SEP = "-" * 54
    print(f"\n  {SEP}")
    print(f"  Diagnostics after {probe_n} probe blocks:")
    print(f"  Acceptance : {n_ok}/{probe_n} ({accept_rate*100:.0f}%)", end="")

    if accept_rate >= 0.70:
        print("  OK")
        print(f"  {SEP}\n")
        return {}

    print("  <- low!")

    ok_snrs = [r[2] for r in probe_raw if r[2] >= min_snr_db]
    if ok_snrs:
        ok_offsets = [r[1] - center_freq for r in probe_raw if r[2] >= min_snr_db]
        print(f"  SNR ok     : min={min(ok_snrs):.1f}  avg={sum(ok_snrs)/len(ok_snrs):.1f}"
              f"  max={max(ok_snrs):.1f} dB")
        print(f"  Offset ok  : avg={sum(ok_offsets)/len(ok_offsets):+.0f} Hz  "
              f"drift={max(ok_offsets)-min(ok_offsets):.1f} Hz")
    if rejected_snrs:
        avg_rej = sum(rejected_snrs) / len(rejected_snrs)
        print(f"  SNR reject : avg={avg_rej:.1f} dB  (threshold={min_snr_db:.1f} dB)")

    suggestions = []

    # Suggestion 1: lower SNR threshold
    if rejected_snrs:
        avg_rej = sum(rejected_snrs) / len(rejected_snrs)
        if avg_rej > min_snr_db * 0.4 and min_snr_db > 1.5:
            new_snr = max(1.0, round(avg_rej * 0.85, 1))
            suggestions.append({
                'desc': f"Lower SNR threshold: {min_snr_db:.1f} -> {new_snr:.1f} dB",
                'param': 'min_snr_db', 'new': new_snr,
            })

    # Suggestion 2: increase Welch sub-blocks
    if n_welch_sub < 100 and accept_rate < 0.5:
        new_sub = min(200, n_welch_sub * 4)
        gain = 10 * math.log10(new_sub / n_welch_sub)
        suggestions.append({
            'desc': f"Increase Welch sub-blocks: {n_welch_sub} -> {new_sub} (+{gain:.0f} dB SNR)",
            'param': 'n_welch_sub', 'new': new_sub,
        })

    # Suggestion 3: provide carrier hint
    if carrier_hint is None and accept_rate < 0.2:
        suggestions.append({
            'desc': "Provide approximate carrier offset from center [Hz] as carrier-hint",
            'param': 'carrier_hint', 'new': None,  # value provided by user
        })

    if not suggestions:
        print(f"\n  No specific suggestions -- continuing with current parameters.")
        print(f"  {SEP}\n")
        return {}

    print(f"\n  Suggested changes (enter number or Enter = no change):")
    for k, s in enumerate(suggestions, 1):
        print(f"  [{k}] {s['desc']}")
    print(f"  [0] Continue without changes")

    try:
        choice = input(f"\n  Your choice (0-{len(suggestions)}): ").strip()
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
            if s['new'] is None:  # carrier_hint -- ask for value
                try:
                    val_str = input("  Enter carrier offset from center [Hz] (e.g. -15000): ").strip()
                    s['new'] = float(val_str)
                except (ValueError, EOFError):
                    print("  Invalid value -- no change.")
                    print(f"  {SEP}\n")
                    return {}
            print(f"  Applied: {s['desc']}")
            print(f"  {SEP}\n")
            return {s['param']: s['new']}
    except ValueError:
        pass

    print(f"  {SEP}\n")
    return {}


# ---------------------------------------------------------------------------
# Block processing
# ---------------------------------------------------------------------------

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
    Slide integration window and collect carrier frequency measurements.

    For each time block:
      - Welch PSD -> SNR
      - SNR filter
      - Timestamp = END of window (INTEGRATION_REF = END, required by NASA)

    Returns: list of (datetime, freq_hz, snr_db)
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
    print(f"  IQ samples       : {len(iq):,}")
    print(f"  Sample rate      : {sample_rate/1e6:.3f} Msps")
    print(f"  Center freq      : {center_freq/1e6:.6f} MHz")
    print(f"  Integration      : {integration_sec:.1f} s ({spb:,} samples/block)")
    print(f"  FFT size         : {eff_fft:,}  ({bin_hz:.2f} Hz/bin)")
    print(f"  Welch sub-blocks : {n_welch_sub}  (SNR gain ~{snr_gain:.1f} dB)")
    print(f"  Min SNR          : {min_snr_db:.1f} dB")
    print(f"  Search bandwidth : +/-{search_bw/1e3:.0f} kHz")
    if carrier_hint is not None:
        print(f"  Carrier hint     : {carrier_hint:+.0f} Hz from center")
    print(f"  Excl. sidebands  : {'yes' if excl_sidebands else 'no'}")
    if auto:
        print(f"  Mode             : AUTO (carrier -> OQPSK fallback)")
    elif oqpsk:
        print(f"  Mode             : OQPSK (IQ^4, /4)")
    else:
        print(f"  Mode             : carrier (Welch)")
    print(f"  Adaptive         : yes (auto-increase welch-sub if acceptance < 70%)")
    print(f"  Blocks           : {n_blocks}")
    print(f"{'='*64}")

    measurements = []
    skipped = 0
    rejected_snrs = []   # SNR of rejected blocks (below threshold)
    probe_raw = []       # (t, freq_abs, snr) -- all blocks in probe phase
    n_carrier_mode = 0   # blocks detected via direct carrier (auto mode)
    n_oqpsk_mode   = 0   # blocks detected via OQPSK IQ^4 (auto mode)

    use_tty = interactive and sys.stdout.isatty() and sys.stdin.isatty()
    probe_n = min(20, max(10, n_blocks // 10))
    # Probe phase always runs (adaptive welch-sub tuning works even without TTY)
    probe_done = probe_n >= n_blocks

    t0_proc = time.time()

    for i in range(n_blocks):
        block = iq[i*spb : (i+1)*spb]
        # Timestamp = end of integration window
        t = start_time + timedelta(seconds=(i + 1) * integration_sec)

        try:
            if auto:
                # Attempt 1: direct carrier search
                freq_abs, snr = estimate_carrier(
                    block, sample_rate, center_freq,
                    fft_size=eff_fft, n_sub=n_welch_sub,
                    search_bw=search_bw, carrier_hint=carrier_hint,
                    hint_bw=hint_bw, excl_sidebands=excl_sidebands,
                    oqpsk=False,
                )
                block_mode = 'C'
                if snr < min_snr_db:
                    # Attempt 2: OQPSK IQ^4 -- extra +2 dB margin to avoid false detections
                    freq_q, snr_q = estimate_carrier(
                        block, sample_rate, center_freq,
                        fft_size=eff_fft, n_sub=n_welch_sub,
                        search_bw=search_bw, carrier_hint=carrier_hint,
                        hint_bw=hint_bw, excl_sidebands=False,
                        oqpsk=True,
                    )
                    if snr_q >= min_snr_db + 2.0:   # +2 dB margin for OQPSK
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
                print(f"\r  [ERR] block {i+1:4d}: {e}     ")
            else:
                print(f"  [ERR] block {i+1:4d}: {e}", file=sys.stderr)
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

        # Collect probe phase data (before diagnostics)
        if not probe_done:
            probe_raw.append((t, freq_abs, snr))

        # -- Progress bar (TTY) ---------------------------------------------
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
            mode_indicator = f"[{block_mode}] " if auto else ""
            bw = 28
            filled = int(bw * (i + 1) / n_blocks)
            bar = "█" * filled + "░" * (bw - filled)
            print(f"\r  {status} {mode_indicator}[{bar}] {i+1}/{n_blocks} | "
                  f"ok:{n_ok}({accept_pct:.0f}%) | "
                  f"off:{offset:+.0f}Hz | SNR:{snr:.1f}dB | {eta_str}   ",
                  end='', flush=True)
        else:
            # Non-TTY: print lines periodically
            mode_tag = f'[{block_mode}]' if auto else '[OK]'
            if accepted:
                if len(measurements) <= 5 or (i+1) % 30 == 0 or i == n_blocks-1:
                    print(f"  {mode_tag}  {i+1:4d}/{n_blocks}  {_dt_to_tdm(t)}  "
                          f"offset={offset:+10.2f} Hz  SNR={snr:5.1f} dB")
            else:
                if len(rejected_snrs) <= 3 or (i+1) % 60 == 0:
                    print(f"  [--]  {i+1:4d}/{n_blocks}  "
                          f"offset={offset:+10.2f} Hz  SNR={snr:5.1f} dB  <-- below threshold")

        # -- Diagnostics / adaptation after probe phase ---------------------
        if not probe_done and (i + 1) == probe_n:
            probe_done = True
            if use_tty:
                print()  # newline after progress bar

            n_ok_probe = sum(1 for _, _, s in probe_raw if s >= min_snr_db)
            accept_rate = n_ok_probe / max(probe_n, 1)

            if accept_rate < 0.70:
                # Automatically increase welch sub-blocks until acceptance >= 70%
                # or we hit the cap (500). Re-process probe blocks each time.
                MAX_ADAPTIVE_SUB = 500
                adapted = False
                while accept_rate < 0.70 and n_welch_sub < MAX_ADAPTIVE_SUB:
                    new_sub = min(MAX_ADAPTIVE_SUB, n_welch_sub * 4)
                    snr_gain_new = 10 * math.log10(max(1, new_sub))
                    print(f"  [adaptive] acceptance {accept_rate*100:.0f}% < 70% -- "
                          f"increasing welch-sub: {n_welch_sub} -> {new_sub} "
                          f"(gain ~{snr_gain_new:.1f} dB)")
                    n_welch_sub = new_sub
                    # Re-score probe blocks with new sub count
                    new_probe = []
                    for t2, f2, _ in probe_raw:
                        idx2 = int((t2 - start_time).total_seconds() / integration_sec) - 1
                        blk2 = iq[idx2*spb : (idx2+1)*spb]
                        try:
                            f2new, s2new = estimate_carrier(
                                blk2, sample_rate, center_freq,
                                fft_size=eff_fft, n_sub=n_welch_sub,
                                search_bw=search_bw, carrier_hint=carrier_hint,
                                hint_bw=hint_bw, excl_sidebands=excl_sidebands,
                                oqpsk=oqpsk,
                            )
                            new_probe.append((t2, f2new, s2new))
                        except Exception:
                            new_probe.append((t2, f2, 0.0))
                    probe_raw = new_probe
                    n_ok_probe = sum(1 for _, _, s in probe_raw if s >= min_snr_db)
                    accept_rate = n_ok_probe / max(probe_n, 1)
                    adapted = True

                measurements = [(t2, f2, s2) for t2, f2, s2 in probe_raw if s2 >= min_snr_db]
                skipped = sum(1 for _, _, s in probe_raw if s < min_snr_db)
                rejected_snrs = [s for _, _, s in probe_raw if s < min_snr_db]
                if adapted:
                    print(f"  [adaptive] probe acceptance after adaptation: "
                          f"{n_ok_probe}/{probe_n} ({accept_rate*100:.0f}%)")

                    # If still no signal even at max welch-sub, check if
                    # the probe blocks have weak-but-consistent SNR.
                    # Aggregation doesn't help with Doppler-drifting signals
                    # (carrier smears across bins), so instead detect the
                    # pattern of "consistent weak SNR" and auto-lower threshold.
                    if accept_rate == 0.0:
                        probe_snrs = sorted([s for _, _, s in probe_raw])
                        median_probe_snr = probe_snrs[len(probe_snrs) // 2]

                        if median_probe_snr >= 1.5:
                            # Signal present but below threshold — auto-lower
                            new_thr = max(1.5, median_probe_snr * 0.8)
                            print(f"  [adaptive] weak signal in probe "
                                  f"(median SNR={median_probe_snr:.1f} dB) -- "
                                  f"lowering min-snr: {min_snr_db:.1f} -> "
                                  f"{new_thr:.1f} dB")
                            min_snr_db = new_thr
                            # Re-score probe blocks with new threshold
                            measurements = [(t2, f2, s2) for t2, f2, s2 in probe_raw
                                            if s2 >= min_snr_db]
                            skipped = sum(1 for _, _, s in probe_raw if s < min_snr_db)
                            rejected_snrs = [s for _, _, s in probe_raw if s < min_snr_db]
                            n_ok_probe = len(measurements)
                            accept_rate = n_ok_probe / max(probe_n, 1)
                        else:
                            # Truly no signal — scan forward with lower threshold
                            print(f"  [adaptive] no signal in probe window -- "
                                  f"scanning file for signal onset...")
                            scan_snr_thr = max(1.5, min_snr_db * 0.5)
                            scan_step = max(probe_n, max(1, n_blocks // 30))
                            found_at = None
                            for scan_i in range(probe_n, n_blocks, scan_step):
                                blk = iq[scan_i * spb : (scan_i + 1) * spb]
                                if len(blk) < spb:
                                    break
                                try:
                                    _, snr_scan = estimate_carrier(
                                        blk, sample_rate, center_freq,
                                        fft_size=eff_fft, n_sub=n_welch_sub,
                                        search_bw=search_bw, carrier_hint=carrier_hint,
                                        hint_bw=hint_bw, excl_sidebands=excl_sidebands,
                                        oqpsk=oqpsk,
                                    )
                                    if snr_scan >= scan_snr_thr:
                                        found_at = scan_i
                                        break
                                except Exception:
                                    pass
                            if found_at is not None:
                                onset_sec = found_at * integration_sec
                                onset_t = start_time + timedelta(seconds=onset_sec)
                                print(f"  [adaptive] signal found at block "
                                      f"{found_at+1}/{n_blocks} "
                                      f"(~{onset_sec/60:.1f} min into recording, "
                                      f"~{onset_t.strftime('%H:%M:%S')} UTC)")
                                # Auto-lower threshold based on signal strength
                                test_snrs = []
                                for ti in range(found_at, min(found_at + 10, n_blocks)):
                                    tblk = iq[ti * spb : (ti + 1) * spb]
                                    if len(tblk) < spb:
                                        break
                                    try:
                                        _, ts = estimate_carrier(
                                            tblk, sample_rate, center_freq,
                                            fft_size=eff_fft, n_sub=n_welch_sub,
                                            search_bw=search_bw,
                                            carrier_hint=carrier_hint,
                                            hint_bw=hint_bw,
                                            excl_sidebands=excl_sidebands,
                                            oqpsk=oqpsk,
                                        )
                                        test_snrs.append(ts)
                                    except Exception:
                                        pass
                                if test_snrs:
                                    med_snr = sorted(test_snrs)[len(test_snrs) // 2]
                                    if med_snr < min_snr_db and med_snr >= 1.5:
                                        new_thr = max(1.5, med_snr * 0.8)
                                        print(f"  [adaptive] weak signal "
                                              f"(median SNR={med_snr:.1f} dB) -- "
                                              f"lowering min-snr: "
                                              f"{min_snr_db:.1f} -> {new_thr:.1f} dB")
                                        min_snr_db = new_thr
                            else:
                                print(f"  [adaptive] no signal found anywhere "
                                      f"in recording.")
                                print(f"  Check antenna pointing, center "
                                      f"frequency, or IQ file.")

                    print(f"  Continuing from block {i+2}/{n_blocks} "
                          f"with welch-sub={n_welch_sub}...\n")

            elif interactive:
                new_params = _interactive_probe(
                    probe_raw, rejected_snrs[:], probe_n,
                    min_snr_db, carrier_hint, n_welch_sub, center_freq,
                )
                if new_params:
                    if 'min_snr_db' in new_params:
                        min_snr_db = new_params['min_snr_db']
                        measurements = [(t2, f, s) for t2, f, s in probe_raw if s >= min_snr_db]
                        skipped = sum(1 for _, _, s in probe_raw if s < min_snr_db)
                        rejected_snrs = [s for _, _, s in probe_raw if s < min_snr_db]
                        print(f"  Re-scored {len(probe_raw)} probe blocks -> {len(measurements)} accepted")
                    if 'n_welch_sub' in new_params:
                        n_welch_sub = new_params['n_welch_sub']
                        snr_gain_new = 10 * math.log10(max(1, n_welch_sub))
                        print(f"  New Welch sub-blocks: {n_welch_sub} (gain ~{snr_gain_new:.1f} dB)")
                    if 'carrier_hint' in new_params:
                        carrier_hint = new_params['carrier_hint']
                        print(f"  New carrier hint: {carrier_hint:+.0f} Hz")
                    print(f"  Continuing from block {i+2}/{n_blocks}...\n")

    if use_tty:
        print()  # newline after final progress bar

    print(f"\n  Accepted: {len(measurements)}/{n_blocks}  (skipped: {skipped})")

    if measurements:
        offsets = [m[1]-center_freq for m in measurements]
        snrs    = [m[2] for m in measurements]
        print(f"  Carrier offset : min={min(offsets):+.1f}  max={max(offsets):+.1f}  "
              f"drift={max(offsets)-min(offsets):.1f} Hz")
        print(f"  SNR            : min={min(snrs):.1f}  max={max(snrs):.1f}  "
              f"mean={sum(snrs)/len(snrs):.1f} dB")
        if auto:
            print(f"  Detection modes: carrier={n_carrier_mode}  OQPSK={n_oqpsk_mode}")

    return measurements


# ---------------------------------------------------------------------------
# TDM output
# ---------------------------------------------------------------------------

def write_tdm(measurements, output_path, station_name, center_freq_hz,
              integration_sec, originator=None, dsn_station=None, comment=None,
              participant_1=None):
    """Write CCSDS TDM v2.0 KVN file."""
    if not measurements:
        raise ValueError("No measurements to write")

    freq_offset = round(center_freq_hz)
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
    print(f"  TDM written  : {output_path}")
    print(f"  Measurements : {len(measurements)}")
    print(f"  Duration     : {dur:.0f} s ({dur/60:.1f} min)")
    print(f"  Mode         : {'3-way ('+dsn_station+')' if dsn_station else '1-way'}")
    print(f"{'='*64}")


# ---------------------------------------------------------------------------
# Optional diagnostic spectrum plot
# ---------------------------------------------------------------------------

def plot_spectrum(iq, sample_rate, center_freq, output_png,
                 fft_size=65536, n_sub=50, duration_sec=60.0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [INFO] matplotlib not available -- skipping plot")
        return

    n = min(len(iq), int(sample_rate * duration_sec))
    print(f"  Computing spectrum ({n/sample_rate:.0f} s, FFT={fft_size}, sub={n_sub})...")

    psd   = welch_psd(iq[:n], fft_size, n_sub)
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0/sample_rate)) / 1e3  # kHz
    psd_s = 10 * np.log10(np.fft.fftshift(psd) + 1e-20)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(freqs, psd_s, lw=0.5, color="steelblue")
    ax.set_xlabel("Offset from center [kHz]")
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
                       label=f"+-{dr/1e3:.0f}k sideband" if sign==1 else None)
    ax.axvline(0, color="lime", lw=1.0, ls=":", label="DC/center")
    ax.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    plt.savefig(str(output_png), dpi=120)
    plt.close()
    print(f"  Spectrum saved: {output_png}")



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True,
                   help=".sigmf-meta  OR  .wav (SDR Console/SDR#/HDSDR)  OR  .raw/.bin/.iq (GQRX)")
    p.add_argument("--freq",  type=float, help="Center frequency [Hz]")
    p.add_argument("--rate",  type=float, help="Sample rate [Sps]")
    p.add_argument("--start", type=str,   help="Recording start time (ISO-8601 UTC)")
    p.add_argument("--dtype", type=str,
                   choices=["cf32_le","cf32_be","ci16_le","ci16_be","ci8","cu8","cf64_le"],
                   help="IQ data type (default: cf32_le)")
    p.add_argument("--station",    "-s", required=True,
                   help="Station callsign or name (e.g. SP5LOT)")
    p.add_argument("--originator", type=str)
    p.add_argument("--dsn-station", type=str,
                   help="DSN uplink station name (e.g. DSS-26) -> 3-way mode")
    p.add_argument("--integration", type=float, default=1.0,
                   help="Integration interval [s] (NASA: 1 or 10, default: 1)")
    p.add_argument("--fft-size",    type=int,   default=65536,
                   help="FFT window size (default: 65536)")
    p.add_argument("--welch-sub",   type=int,   default=20,
                   help=(
                       "Number of Welch sub-blocks [default: 20]. "
                       "More = better SNR for weak signals. "
                       "With a small antenna try 50-200."
                   ))
    p.add_argument("--min-snr",     type=float, default=3.0,
                   help="Minimum SNR [dB] to accept a measurement (default: 3.0)")
    p.add_argument("--search-bw",   type=float, default=None,
                   help="Carrier search bandwidth [Hz]")
    p.add_argument("--carrier-hint", type=float, default=None,
                   help=(
                       "Approximate carrier offset [Hz] from center. "
                       "If you can see the signal on a waterfall, enter its offset here "
                       "(e.g. --carrier-hint -15000 = 15 kHz below center). "
                       "Very helpful for weak signals!"
                   ))
    p.add_argument("--hint-bw", type=float, default=50_000,
                   help=(
                       "Half-bandwidth around --carrier-hint [Hz] "
                       "(default 50000 = +/-50 kHz). Narrow this if strong sidebands "
                       "are near the carrier, e.g. --hint-bw 15000."
                   ))
    p.add_argument("--no-excl-sidebands", action="store_true",
                   help="Do not exclude PCM/PM/NRZ sideband regions from carrier search")
    p.add_argument("--oqpsk", action="store_true",
                   help=(
                       "OQPSK suppressed-carrier mode (Artemis II). "
                       "Raises IQ to 4th power before Welch (M-th power carrier recovery), "
                       "removing QPSK modulation and revealing the carrier as CW at 4*delta_f. "
                       "Result divided by 4. Use with --no-excl-sidebands."
                   ))
    p.add_argument("--auto", action="store_true",
                   help=(
                       "Auto-detect modulation per block. Tries direct CW carrier search "
                       "(KPLO, LRO, Artemis I) first; if SNR is too low, falls back to "
                       "OQPSK IQ^4 recovery (Artemis II). "
                       "Useful when the signal type is unknown."
                   ))
    p.add_argument("--max-samples",  type=int,  default=None,
                   help="Load only first N samples (for testing on large files)")
    p.add_argument("--skip-samples", type=int,  default=None,
                   help="Skip first N samples (for testing mid-file segments)")
    p.add_argument("--output",  "-o", default=None, help="Output TDM filename")
    p.add_argument("--spacecraft", "--participant-1", type=str, default=None,
                   dest="spacecraft",
                   help="Spacecraft name for PARTICIPANT_1 (default: ORION)")
    p.add_argument("--location", type=str,
                   help="Station lat,lon,alt (e.g. 52.23,21.01,110) for TDM comment")
    p.add_argument("--comment", type=str)
    p.add_argument("--plot", action="store_true",
                   help="Save Welch spectrum plot to PNG (requires matplotlib)")
    p.add_argument("--no-interactive", action="store_true",
                   help="Disable interactive diagnostics and progress bar (for scripts/cron)")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"ERROR: File not found: {inp}")

    # -- Detect format ------------------------------------------------------
    suffix   = inp.suffix.lower()
    is_sigmf = suffix == ".sigmf-meta"
    is_wav   = suffix == ".wav"
    is_gqrx  = suffix in (".raw", ".bin", ".iq")
    if not is_sigmf and not is_wav and not is_gqrx:
        meta_g = inp.with_suffix(".sigmf-meta")
        if meta_g.exists():
            inp, is_sigmf = meta_g, True
        else:
            is_gqrx = True

    info = {}

    if is_sigmf:
        data_path = inp.with_suffix(".sigmf-data")
        if not data_path.exists():
            sys.exit(f"ERROR: Data file not found: {data_path}")
        print(f"[SigMF] {inp.name}")
        info = read_sigmf_meta(inp)
        print(f"  Datatype  : {info['datatype']}")
        print(f"  Rate      : {info['sample_rate']/1e6:.3f} Msps")
        print(f"  Freq      : {info['center_freq']/1e6:.6f} MHz")
        if info["start_time"]:
            print(f"  Start     : {info['start_time'].isoformat()}")
        print(f"\nLoading IQ from: {data_path}")
        iq = load_iq(data_path, info["datatype"], args.max_samples, args.skip_samples)
    elif is_wav:
        data_path = inp
        print(f"[WAV] {inp.name}")
        info = parse_wav_iq(inp)
        print(f"  Datatype  : {info['datatype']}")
        print(f"  Rate      : {info['sample_rate']/1e6:.3f} Msps")
        if info["center_freq"]:
            print(f"  Freq      : {info['center_freq']/1e6:.6f} MHz")
        if info["start_time"]:
            print(f"  Start     : {info['start_time'].isoformat()}")
        if info.get("hw"):
            print(f"  HW        : {info['hw']}")
        print(f"  Data at   : offset {info['data_offset']} bytes")
        print(f"\nLoading IQ from: {data_path}")
        iq = load_iq(data_path, info["datatype"], args.max_samples, args.skip_samples,
                     data_offset=info["data_offset"])
    else:
        data_path = inp
        fn_info   = parse_gqrx_filename(inp.name)
        info.update(fn_info)
        dt_str = args.dtype or "cf32_le"
        info.setdefault("datatype", dt_str)
        print(f"[GQRX] {inp.name}")
        if fn_info:
            print(f"  From filename: freq={info.get('center_freq',0)/1e6:.3f} MHz  "
                  f"rate={info.get('sample_rate',0)/1e6:.3f} Msps  "
                  f"start={info.get('start_time','?')}")
        print(f"\nLoading IQ from: {data_path}")
        iq = load_iq(data_path, info["datatype"], args.max_samples, args.skip_samples)

    # -- CLI overrides ------------------------------------------------------
    if args.freq:  info["center_freq"] = args.freq
    if args.rate:  info["sample_rate"] = args.rate
    if args.start: info["start_time"]  = _parse_dt(args.start)
    if args.skip_samples and info.get("start_time") and info.get("sample_rate"):
        info["start_time"] += timedelta(seconds=args.skip_samples / info["sample_rate"])

    missing = [k for k in ("center_freq","sample_rate","start_time") if not info.get(k)]
    if missing:
        sys.exit(
            "ERROR -- missing metadata: " + ", ".join(missing) +
            "\nProvide: --freq, --rate, --start"
        )

    cf, sr, t0 = info["center_freq"], info["sample_rate"], info["start_time"]

    # -- Optional spectrum plot ---------------------------------------------
    if args.plot:
        plot_out = Path(args.output or f"{args.station}_spectrum").with_suffix(".png")
        plot_spectrum(iq, sr, cf, plot_out,
                      fft_size=min(65536, 2**int(math.log2(max(1024, int(sr*30))))),
                      n_sub=50,
                      duration_sec=min(120.0, len(iq)/sr))

    # -- Main processing ----------------------------------------------------
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
        print("\nERROR: No measurements -- signal too weak or SNR threshold too high.")
        print("Try: --min-snr 1.5  --welch-sub 100  --carrier-hint <Hz>")
        sys.exit(1)

    # -- Write TDM ----------------------------------------------------------
    sc = args.spacecraft or "ORION"
    out = Path(args.output) if args.output else \
          Path(f"{args.station}_{sc}_{t0.strftime('%Y%m%d_%H%M%S')}.tdm")

    mode_str = "OQPSK (M-th power /4)" if args.oqpsk else "carrier (Welch)"
    loc_line = ""
    if args.location:
        parts = args.location.split(",")
        if len(parts) >= 2:
            lat, lon = float(parts[0]), float(parts[1])
            alt = f", {parts[2]}m" if len(parts) >= 3 else ""
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            loc_line = f"Station: {args.station} ({abs(lat):.2f}{ns}, {abs(lon):.2f}{ew}{alt})\n"
    auto_cmt = (
        f"{args.spacecraft or 'ORION'} one-way Doppler tracking\n"
        f"{loc_line}"
        f"Source: {inp.name}\n"
        f"HW: {info.get('hw','?')} | "
        f"FFT={args.fft_size} Welch={args.welch_sub} int={args.integration}s | "
        f"mode={mode_str}"
    )
    write_tdm(meas, out, args.station, cf, args.integration,
              args.originator, args.dsn_station, args.comment or auto_cmt,
              participant_1=args.spacecraft)



if __name__ == "__main__":
    main()
