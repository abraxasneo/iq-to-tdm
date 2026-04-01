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
import urllib.parse
import urllib.request
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
        seg  = seg - np.mean(seg)   # DC removal per sub-block
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


# ---------------------------------------------------------------------------
# Viterbi ridge tracker for weak / modulated signals
# ---------------------------------------------------------------------------

def _build_spectrogram(iq, sample_rate, fft_size, n_sub, spb, n_blocks,
                       carrier_hint, hint_bw, search_bw, dc_excl=0.0,
                       oqpsk=False):
    """Build spectrogram matrix (n_blocks × n_freq_bins) within search window.

    When oqpsk=True, each block is raised to the 4th power before Welch PSD
    (M-th power carrier recovery). The freq_axis is in the IQ^4 domain (4×).

    Returns:
        psd_matrix : 2D array (n_blocks, n_mask_bins) — power in dB
        freq_axis  : 1D array of offset frequencies [Hz] for masked bins
        mask_idx   : indices into full FFT for the masked bins
    """
    freqs_full = np.fft.fftfreq(fft_size, d=1.0 / sample_rate)

    # Build frequency mask (same logic as estimate_carrier)
    if carrier_hint is not None:
        mask = np.abs(freqs_full - carrier_hint) <= hint_bw
    elif search_bw is not None:
        mask = np.abs(freqs_full) <= search_bw / 2
    else:
        mask = np.abs(freqs_full) <= sample_rate * 0.40

    # Exclude DC spike region if requested
    if dc_excl > 0:
        mask &= np.abs(freqs_full) > dc_excl

    mask_idx = np.where(mask)[0]
    freq_axis = freqs_full[mask_idx]

    # Sort by frequency for clean output
    order = np.argsort(freq_axis)
    mask_idx = mask_idx[order]
    freq_axis = freq_axis[order]

    n_bins = len(mask_idx)
    psd_matrix = np.zeros((n_blocks, n_bins), dtype=np.float64)

    for i in range(n_blocks):
        block = iq[i * spb : (i + 1) * spb]
        if oqpsk:
            block = (block.astype(np.complex128) ** 4).astype(np.complex64)
            block = block - np.mean(block)
        psd = welch_psd(block, fft_size, n_sub)
        psd_matrix[i, :] = psd[mask_idx]

    # Convert to dB
    psd_matrix = 10.0 * np.log10(psd_matrix + 1e-30)

    return psd_matrix, freq_axis, mask_idx


def _smooth_kalman_rts(track_freq, track_snr, poly_order=3, sigma_clip=2.5,
                       q_accel=0.1):
    """Smooth Viterbi frequency track using outlier rejection + Kalman RTS.

    Stage 1: Iterative sigma-clipping polynomial fit to reject outliers.
    Stage 2: Kalman forward-backward (RTS) smoother with SNR-weighted
             measurement noise — optimal batch estimator for smooth signals.

    Args:
        track_freq : raw frequency estimates per frame [Hz]
        track_snr  : per-frame SNR [dB] (used to weight measurements)
        poly_order : polynomial order for outlier rejection (default 3)
        sigma_clip : rejection threshold in sigma units (default 2.5)
        q_accel    : process noise — expected acceleration variance [Hz/step^2]
                     Lower = smoother. For L1 objects ~0.1, lunar ~1.0.

    Returns:
        smoothed frequency track [Hz], same length as input
    """
    nf = len(track_freq)
    t_idx = np.arange(nf, dtype=np.float64)

    # --- Stage 1: outlier rejection via iterative sigma-clipping poly fit ---
    mask = np.ones(nf, dtype=bool)
    for _iteration in range(3):
        if np.sum(mask) < poly_order + 2:
            break
        # Normalize time to [-1, 1] for numerical stability
        t_norm = 2.0 * (t_idx[mask] - t_idx[mask][0]) / max(1, t_idx[mask][-1] - t_idx[mask][0]) - 1.0
        coeffs = np.polyfit(t_norm, track_freq[mask], min(poly_order, np.sum(mask) - 1))
        # Evaluate at all points
        t_all_norm = 2.0 * (t_idx - t_idx[mask][0]) / max(1, t_idx[mask][-1] - t_idx[mask][0]) - 1.0
        poly_fit = np.polyval(coeffs, t_all_norm)
        residuals = np.abs(track_freq - poly_fit)
        sigma = np.std(track_freq[mask] - poly_fit[mask])
        if sigma < 1e-6:
            break
        new_mask = residuals < sigma_clip * sigma
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    n_rejected = nf - np.sum(mask)
    if n_rejected > 0:
        # Replace outliers with polynomial prediction before Kalman
        track_freq = track_freq.copy()
        track_freq[~mask] = poly_fit[~mask]
        # Mark outliers with very low SNR so Kalman trusts them less
        track_snr = track_snr.copy()
        track_snr[~mask] = -10.0

    # --- Stage 2: Kalman RTS smoother, state = [freq, freq_rate] ---
    # Measurement noise per point: inversely proportional to SNR
    snr_linear = np.maximum(10.0 ** (track_snr / 10.0), 0.1)
    # Base noise variance from median SNR
    median_snr_lin = np.median(snr_linear[mask]) if np.any(mask) else 1.0
    base_var = np.var(track_freq[mask] - poly_fit[mask]) if np.any(mask) else 1e6
    base_var = max(base_var, 1.0)  # floor
    R_per = base_var * (median_snr_lin / snr_linear)  # lower SNR → higher variance

    # State vectors and covariance
    x = np.zeros((nf, 2))       # [freq, freq_rate]
    P = np.zeros((nf, 2, 2))
    x_pred = np.zeros((nf, 2))
    P_pred = np.zeros((nf, 2, 2))
    H = np.array([[1.0, 0.0]])

    # Initialize
    x[0] = [track_freq[0], 0.0]
    P[0] = np.diag([base_var, base_var])

    # Forward pass
    for i in range(1, nf):
        dt = 1.0  # uniform frame steps
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = q_accel * np.array([[dt**3 / 3, dt**2 / 2],
                                [dt**2 / 2, dt]])
        # Predict
        x_pred[i] = F @ x[i - 1]
        P_pred[i] = F @ P[i - 1] @ F.T + Q
        # Update
        R_i = np.array([[R_per[i]]])
        y = track_freq[i] - H @ x_pred[i]
        S = H @ P_pred[i] @ H.T + R_i
        K = P_pred[i] @ H.T / S[0, 0]
        x[i] = x_pred[i] + (K * y[0]).flatten()
        P[i] = (np.eye(2) - K @ H) @ P_pred[i]

    # RTS backward smoother
    x_smooth = x.copy()
    P_smooth = P.copy()
    for i in range(nf - 2, -1, -1):
        dt = 1.0
        F = np.array([[1.0, dt], [0.0, 1.0]])
        P_pred_i1 = P_pred[i + 1]
        # Avoid singular inversion
        det = P_pred_i1[0, 0] * P_pred_i1[1, 1] - P_pred_i1[0, 1] * P_pred_i1[1, 0]
        if abs(det) < 1e-30:
            continue
        P_pred_inv = np.array([[P_pred_i1[1, 1], -P_pred_i1[0, 1]],
                               [-P_pred_i1[1, 0], P_pred_i1[0, 0]]]) / det
        C = P[i] @ F.T @ P_pred_inv
        x_smooth[i] = x[i] + C @ (x_smooth[i + 1] - x_pred[i + 1])
        P_smooth[i] = P[i] + C @ (P_smooth[i + 1] - P_pred[i + 1]) @ C.T

    return x_smooth[:, 0]


def _viterbi_ridge(psd_db, freq_axis, max_drift_hz_per_step, stack_k=1):
    """Find optimal frequency track through spectrogram using Viterbi.

    Args:
        psd_db        : 2D array (n_frames, n_bins) — power in dB
        freq_axis     : 1D array of frequencies [Hz]
        max_drift_hz_per_step : max allowed frequency change per time step [Hz]
        stack_k       : stack K consecutive frames before tracking (SNR boost)

    Returns:
        track_freq : 1D array (n_output_frames,) — frequency at each step [Hz]
        track_snr  : 1D array — estimated SNR at each step [dB]
    """
    n_frames, n_bins = psd_db.shape

    # Optional stacking for initial SNR boost
    if stack_k > 1:
        n_stacked = n_frames // stack_k
        stacked = np.zeros((n_stacked, n_bins), dtype=np.float64)
        for i in range(n_stacked):
            # Average in linear domain, then back to dB
            lin = 10.0 ** (psd_db[i*stack_k:(i+1)*stack_k, :] / 10.0)
            stacked[i, :] = 10.0 * np.log10(np.mean(lin, axis=0) + 1e-30)
        psd_work = stacked
    else:
        psd_work = psd_db
        n_stacked = n_frames

    nf, nb = psd_work.shape

    # Noise floor per frame (median across bins)
    noise_floor = np.median(psd_work, axis=1, keepdims=True)
    # Normalized: subtract noise floor so signal shows as positive
    psd_norm = psd_work - noise_floor

    # Max drift in bins per step
    bin_hz = freq_axis[1] - freq_axis[0] if len(freq_axis) > 1 else 1.0
    max_drift_bins = max(1, int(max_drift_hz_per_step / abs(bin_hz) + 0.5))

    # Viterbi forward pass
    # cost[j] = best accumulated score arriving at bin j
    cost = psd_norm[0, :].copy()
    backptr = np.zeros((nf, nb), dtype=np.int32)

    for t in range(1, nf):
        new_cost = np.full(nb, -1e30)
        for j in range(nb):
            # Candidate predecessors: bins within max_drift_bins of j
            lo = max(0, j - max_drift_bins)
            hi = min(nb, j + max_drift_bins + 1)
            best_prev = lo + int(np.argmax(cost[lo:hi]))
            new_cost[j] = cost[best_prev] + psd_norm[t, j]
            backptr[t, j] = best_prev
        cost = new_cost

    # Backtrack
    track_bins = np.zeros(nf, dtype=np.int32)
    track_bins[nf - 1] = int(np.argmax(cost))
    for t in range(nf - 2, -1, -1):
        track_bins[t] = backptr[t + 1, track_bins[t + 1]]

    # Extract frequencies and SNR along track, with sub-bin interpolation
    track_freq = np.zeros(nf)
    track_snr = np.zeros(nf)
    for t in range(nf):
        k = track_bins[t]
        track_snr[t] = psd_norm[t, k]
        # Parabolic interpolation for sub-bin accuracy
        if 0 < k < nb - 1:
            y1 = psd_work[t, k - 1]
            y2 = psd_work[t, k]
            y3 = psd_work[t, k + 1]
            denom = 2 * y2 - y1 - y3
            if denom > 0:
                delta = 0.5 * (y1 - y3) / denom
                track_freq[t] = freq_axis[k] + delta * bin_hz
            else:
                track_freq[t] = freq_axis[k]
        else:
            track_freq[t] = freq_axis[k]

    # Smooth the track — Doppler is physically smooth, remove noise jitter
    # Stage 1: Iterative sigma-clipping with polynomial fit (outlier rejection)
    # Stage 2: Kalman RTS smoother (optimal batch smoothing)
    if nf >= 10:
        # Adaptive q_accel: estimate drift rate from raw track, scale accordingly
        # L1 objects (~0.01 Hz/s): q_accel ~0.1; lunar orbit (~16 Hz/s): q_accel ~50
        diffs = np.diff(track_freq)
        med_rate = np.median(np.abs(diffs))
        # Also estimate acceleration (change in rate)
        if len(diffs) > 1:
            accels = np.abs(np.diff(diffs))
            med_accel = np.median(accels)
        else:
            med_accel = med_rate
        # q_accel = max(0.1, observed_accel^2) — trust data dynamics
        q_accel = max(0.1, float(med_accel ** 2))
        # Cap at reasonable value
        q_accel = min(q_accel, 1e4)
        track_freq = _smooth_kalman_rts(track_freq, track_snr, poly_order=3,
                                        sigma_clip=2.5, q_accel=q_accel)
    elif nf >= 5:
        hw = max(1, nf // 4)
        smoothed = track_freq.copy()
        for i in range(nf):
            lo = max(0, i - hw)
            hi = min(nf, i + hw + 1)
            smoothed[i] = np.median(track_freq[lo:hi])
        track_freq = smoothed

    # Expand back to original frame count if stacked
    if stack_k > 1:
        track_freq_full = np.repeat(track_freq, stack_k)[:n_frames]
        track_snr_full = np.repeat(track_snr, stack_k)[:n_frames]
        return track_freq_full, track_snr_full

    return track_freq, track_snr


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
    centroid    = False,
    dc_excl     = 0.0,
):
    """
    Find residual PCM/PM/NRZ carrier using Welch method.

    --oqpsk mode (M-th power carrier recovery):
      For OQPSK (suppressed carrier) raise IQ to 4th power.
      QPSK modulation cancels (phases 0/90/180/270 deg * 4 = 0 deg),
      leaving a pure CW line at 4*delta_f. Result divided by 4.

    --centroid mode (modulated signal tracking):
      Instead of finding a single peak bin, computes the power-weighted
      spectral centroid within the search window. Works for BPSK/QPSK
      telemetry signals where the carrier is spread across a bandwidth.
      Requires --carrier-hint to define the search region.

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

    # DC removal: subtract mean to eliminate SDR DC spike before PSD.
    # Always applied — DC component carries no Doppler information.
    iq_proc = iq_proc - np.mean(iq_proc)

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
    if excl_sidebands and not centroid:
        for dr in ORION_DATA_RATES_HZ:
            guard = SIDEBAND_GUARD_HZ + dr * 0.05
            for sign in (+1, -1):
                mask[np.abs(freqs - sign * dr) < guard] = False

    # -- DC exclusion zone (avoid DC spur from SDR) --------------------------
    if dc_excl > 0:
        mask &= np.abs(freqs) > dc_excl

    if not np.any(mask):
        # Fallback: drop sideband mask (may not apply to this band)
        mask = np.abs(freqs) <= sample_rate * 0.40
        if dc_excl > 0:
            mask &= np.abs(freqs) > dc_excl

    # -- Centroid mode ------------------------------------------------------
    if centroid:
        psd_m = psd[mask]
        freqs_m = freqs[mask]

        # Noise floor: measured OUTSIDE the hint window (robust estimate)
        if carrier_hint is not None:
            noise_region = np.abs(freqs) <= sample_rate * 0.40
            noise_region &= ~mask   # exclude signal window
            if np.sum(noise_region) > 100:
                noise_p = float(np.median(psd[noise_region]))
            else:
                noise_p = float(np.median(np.sort(psd_m)[:max(1, len(psd_m)//5)]))
        else:
            noise_p = float(np.median(np.sort(psd_m)[:max(1, len(psd_m)//5)]))

        # Subtract noise baseline from signal window
        sig_psd = np.maximum(psd_m - noise_p, 0.0)
        total_sig = float(np.sum(sig_psd))

        if total_sig <= 0:
            return center_freq + (carrier_hint or 0.0), 0.0

        # Power-weighted centroid
        raw_offset = float(np.sum(freqs_m * sig_psd) / total_sig)

        # SNR: mean power in signal window vs noise
        sig_mean = float(np.mean(psd_m))
        if noise_p > 0 and sig_mean > noise_p:
            snr_db = 10.0 * math.log10(sig_mean / noise_p)
        else:
            snr_db = 0.0

        freq_abs = center_freq + raw_offset
        return freq_abs, snr_db

    # -- Peak mode (CW carrier) --------------------------------------------
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
    centroid        = False,
    weak            = False,
    max_drift       = 10.0,
    weak_stack      = 1,
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
    if weak and oqpsk:
        print(f"  Mode             : OQPSK+WEAK (IQ^4 + Viterbi ridge tracker)")
        print(f"  Max drift        : {max_drift:.1f} Hz/s ({max_drift*integration_sec:.1f} Hz/step)")
        if weak_stack > 1:
            print(f"  Frame stacking   : {weak_stack}x (SNR boost ~{5*math.log10(weak_stack):.1f} dB)")
    elif weak:
        print(f"  Mode             : WEAK (Viterbi ridge tracker)")
        print(f"  Max drift        : {max_drift:.1f} Hz/s ({max_drift*integration_sec:.1f} Hz/step)")
        if weak_stack > 1:
            print(f"  Frame stacking   : {weak_stack}x (SNR boost ~{5*math.log10(weak_stack):.1f} dB)")
    elif auto:
        print(f"  Mode             : AUTO (carrier -> OQPSK fallback)")
    elif centroid:
        print(f"  Mode             : centroid (power-weighted spectral center)")
    elif oqpsk:
        print(f"  Mode             : OQPSK (IQ^4, /4)")
    else:
        print(f"  Mode             : carrier (Welch)")
    if not weak:
        print(f"  Adaptive         : yes (auto-increase welch-sub if acceptance < 70%)")
    print(f"  Blocks           : {n_blocks}")
    print(f"{'='*64}")

    # ------------------------------------------------------------------
    # AUTO-DETECT mode: probe CW → OQPSK → weak Viterbi (if no manual mode set)
    # ------------------------------------------------------------------
    auto_detected_mode = None
    original_carrier_hint = carrier_hint   # save before auto-detect modifies it
    probe_dc_excl = 0.0   # measured DC spike exclusion (set by auto-detect probe)
    narrow_dc_notch = False  # True when carrier near DC — notch only 1 bin
    # ------------------------------------------------------------------
    # Measure DC spike from first block (always, regardless of mode)
    # ------------------------------------------------------------------
    blk0 = iq[0:spb]
    psd0 = welch_psd(blk0, eff_fft, n_welch_sub)
    freqs0 = np.fft.fftfreq(eff_fft, d=1.0 / sample_rate)
    dc_bin = np.argmin(np.abs(freqs0))
    dc_power = 10.0 * np.log10(psd0[dc_bin] + 1e-30)
    noise_floor = 10.0 * np.log10(np.median(psd0) + 1e-30)
    dc_snr = dc_power - noise_floor
    bin_hz = sample_rate / eff_fft
    if dc_snr > 10.0:
        probe_dc_excl = 3.0 * bin_hz
        print(f"\n  [DC spike] {dc_snr:.0f} dB above noise, "
              f"excluding ±{probe_dc_excl:.0f} Hz")
    elif dc_snr > 6.0:
        probe_dc_excl = 2.0 * bin_hz
        print(f"\n  [DC spike] Mild: {dc_snr:.0f} dB, "
              f"excluding ±{probe_dc_excl:.0f} Hz")
    else:
        print(f"\n  [DC spike] None detected ({dc_snr:.1f} dB)")

    if not weak and not oqpsk and not centroid and carrier_hint is None:
        # ---------------------------------------------------------------
        # Parallel probe: test first 10 blocks with BOTH CW and OQPSK
        # ---------------------------------------------------------------
        n_probe = min(10, n_blocks)
        probe_ok = 0       # CW detections
        probe_offsets = []  # CW offsets [Hz]
        probe_snrs = []     # CW SNRs [dB]
        probe_ok_q = 0       # OQPSK detections
        probe_offsets_q = [] # OQPSK offsets [Hz]
        probe_snrs_q = []    # OQPSK SNRs [dB]
        for pi in range(n_probe):
            blk = iq[pi * spb : (pi + 1) * spb]
            # CW attempt
            try:
                f_det, s = estimate_carrier(
                    blk, sample_rate, center_freq,
                    fft_size=eff_fft, n_sub=n_welch_sub,
                    search_bw=search_bw, carrier_hint=carrier_hint,
                    hint_bw=hint_bw, excl_sidebands=excl_sidebands,
                    oqpsk=False, dc_excl=probe_dc_excl)
                if s >= min_snr_db:
                    probe_ok += 1
                    probe_offsets.append(f_det - center_freq)
                    probe_snrs.append(s)
            except Exception:
                pass
            # OQPSK attempt (IQ^4 carrier recovery)
            try:
                f_det_q, s_q = estimate_carrier(
                    blk, sample_rate, center_freq,
                    fft_size=eff_fft, n_sub=n_welch_sub,
                    search_bw=search_bw, carrier_hint=carrier_hint,
                    hint_bw=hint_bw, excl_sidebands=False,
                    oqpsk=True, dc_excl=probe_dc_excl)
                if s_q >= min_snr_db + 2.0:
                    probe_ok_q += 1
                    probe_offsets_q.append(f_det_q - center_freq)
                    probe_snrs_q.append(s_q)
            except Exception:
                pass

        # --- CW quality metrics ---
        if len(probe_offsets) >= 3:
            med = float(np.median(probe_offsets))
            offset_scatter = float(np.median(np.abs(np.array(probe_offsets) - med)))
        elif probe_offsets:
            med = float(np.median(probe_offsets))
            offset_scatter = 0.0
        else:
            med = 0.0
            offset_scatter = 1e9
        cw_stable = offset_scatter < 2000.0
        cw_rate = probe_ok / n_probe

        # --- OQPSK quality metrics ---
        if len(probe_offsets_q) >= 3:
            med_q = float(np.median(probe_offsets_q))
            scatter_q = float(np.median(np.abs(np.array(probe_offsets_q) - med_q)))
        elif probe_offsets_q:
            med_q = float(np.median(probe_offsets_q))
            scatter_q = 0.0
        else:
            med_q = 0.0
            scatter_q = 1e9
        oqpsk_stable = scatter_q < 2000.0
        oqpsk_rate = probe_ok_q / n_probe

        # --- Bi-modal DC spike detection ---
        bimodal = False
        sign_changes = 0
        if len(probe_offsets) >= 4:
            offs_arr = np.array(probe_offsets)
            signs = np.sign(offs_arr)
            sign_changes = int(np.sum(signs[1:] != signs[:-1]))
            if sign_changes >= len(offs_arr) * 0.4 and abs(med) < 500:
                bimodal = True

        # --- Compare CW vs OQPSK quality ---
        # Score = acceptance_rate × consistency (lower scatter = better)
        cw_score = cw_rate * max(0.0, 1.0 - offset_scatter / 5000.0) if cw_rate > 0 else 0
        oqpsk_score = oqpsk_rate * max(0.0, 1.0 - scatter_q / 5000.0) if oqpsk_rate > 0 else 0
        # OQPSK needs to be clearly better to override CW (avoid false positives
        # from frequency comb — comb under IQ^4 gives inconsistent peaks)
        oqpsk_wins = (oqpsk_score > cw_score * 1.2
                      and oqpsk_rate >= 0.3
                      and oqpsk_stable)

        print(f"\n  [auto-detect] Probe results: "
              f"CW {probe_ok}/{n_probe} (scatter {offset_scatter:.0f} Hz), "
              f"OQPSK {probe_ok_q}/{n_probe} (scatter {scatter_q:.0f} Hz)")

        # Helper: coarse scan excluding DC to find real signal
        def _coarse_dc_scan():
            """Returns (carrier_hint, hint_bw, accum_db, range) or None."""
            _cfft = min(4096, eff_fft)
            _cn = min(200, n_blocks)
            _max_sub = spb // (_cfft // 2) - 1  # 50% overlap
            _cns = min(max(n_welch_sub, 20), _max_sub)
            _dc_ex = max(probe_dc_excl * 15, 500.0)
            _freqs = np.fft.fftfreq(_cfft, d=1.0 / sample_rate)
            _bw_mask = np.abs(_freqs) <= search_bw / 2
            _dc_mask = np.abs(_freqs) <= _dc_ex
            _idx = np.where(_bw_mask)[0]
            _order = np.argsort(_freqs[_idx])
            _idx = _idx[_order]
            _fax = _freqs[_idx]
            _dc_suppress = np.where(_dc_mask)[0]
            _psd = np.zeros((_cn, len(_idx)), dtype=np.float64)
            for _ci in range(_cn):
                _blk = iq[_ci * spb : (_ci + 1) * spb]
                _p = welch_psd(_blk, _cfft, _cns)
                _nf = np.median(_p)
                _p[_dc_suppress] = _nf
                _psd[_ci, :] = _p[_idx]
            _psd = 10.0 * np.log10(_psd + 1e-30)
            _med_spec = np.median(_psd, axis=0)
            _psd = _psd - _med_spec
            _drift = max_drift * integration_sec
            _track, _snr = _viterbi_ridge(_psd, _fax, _drift)
            _acc = np.sum(np.maximum(10.0 ** (_snr / 10.0), 0.0))
            _acc_db = 10.0 * math.log10(max(1.0, _acc))
            _med = float(np.median(_track))
            if _acc_db >= 3.0 and abs(_med) > _dc_ex:
                _rng = float(np.max(_track) - np.min(_track))
                return _med, max(hint_bw, _rng * 2 + 2000), _acc_db, _rng
            return None

        # --- Decision tree ---
        if oqpsk_wins:
            # OQPSK is clearly better than CW — use auto CW/OQPSK mode
            auto_detected_mode = "OQPSK"
            auto = True
            print(f"  [auto-detect] OQPSK signal detected "
                  f"(score {oqpsk_score:.2f} vs CW {cw_score:.2f}), "
                  f"using auto CW/OQPSK mode.")
        elif cw_rate >= 0.7 and cw_stable and not bimodal:
            if abs(med) < 500 and probe_dc_excl > 0:
                if dc_snr > 10.0:
                    print(f"  [auto-detect] CW probe near DC "
                          f"(median {med:+.0f} Hz) "
                          f"+ strong DC spike ({dc_snr:.0f} dB)")
                    print(f"  [auto-detect] Scanning for signal away from DC...")
                    cs = _coarse_dc_scan()
                    if cs is not None:
                        auto_detected_mode = "WEAK"
                        weak = True
                        carrier_hint = cs[0]
                        hint_bw = cs[1]
                        print(f"  [auto-detect] Found signal at "
                              f"{carrier_hint:+.0f} Hz (accum SNR "
                              f"{cs[2]:.1f} dB, range {cs[3]:.0f} Hz)")
                        print(f"  [auto-detect] Search window: "
                              f"{carrier_hint:+.0f} ± {hint_bw:.0f} Hz")
                    else:
                        bin_hz = sample_rate / eff_fft
                        probe_dc_excl = bin_hz
                        auto_detected_mode = "CW"
                        print(f"  [auto-detect] No signal away from DC — "
                              f"CW carrier near DC, narrowing exclusion "
                              f"to ±{bin_hz:.0f} Hz.")
                else:
                    bin_hz = sample_rate / eff_fft
                    probe_dc_excl = bin_hz
                    auto_detected_mode = "CW"
                    print(f"  [auto-detect] CW carrier near DC "
                          f"(median {med:+.0f} Hz)")
                    print(f"  [auto-detect] Narrowing DC exclusion to "
                          f"±{bin_hz:.0f} Hz (1 bin) to preserve carrier.")
            else:
                auto_detected_mode = "CW"
                print(f"  [auto-detect] CW carrier detected "
                      f"(scatter {offset_scatter:.0f} Hz), "
                      f"using standard mode.")
        elif bimodal and cw_rate >= 0.3:
            median_off = float(np.median(probe_offsets)) if probe_offsets else 0.0
            auto_detected_mode = "WEAK"
            weak = True
            print(f"  [auto-detect] DC spike interference detected "
                  f"({sign_changes} sign flips, "
                  f"median {median_off:+.0f} Hz)")
            print(f"  [auto-detect] Scanning for signal away from DC...")
            cs = _coarse_dc_scan()
            if cs is not None:
                carrier_hint = cs[0]
                hint_bw = cs[1]
                print(f"  [auto-detect] Found signal at {carrier_hint:+.0f} Hz "
                      f"(accum SNR {cs[2]:.1f} dB, range {cs[3]:.0f} Hz)")
                print(f"  [auto-detect] Search window: "
                      f"{carrier_hint:+.0f} ± {hint_bw:.0f} Hz")
            elif abs(median_off) < 500:
                carrier_hint = median_off
                hint_bw = max(hint_bw, 50000)
                narrow_dc_notch = True
                print(f"  [auto-detect] Signal near DC — "
                      f"using Viterbi with narrow DC notch.")
            else:
                carrier_hint = median_off
                hint_bw = max(hint_bw, 50000)
                print(f"  [auto-detect] Using Viterbi to track "
                      f"through DC region...")
        elif oqpsk_rate >= 0.3 and oqpsk_stable:
            # OQPSK is viable even if CW also had marginal detections
            auto_detected_mode = "OQPSK"
            auto = True
            print(f"  [auto-detect] OQPSK signal detected "
                  f"({probe_ok_q}/{n_probe} blocks), "
                  f"using auto CW/OQPSK mode.")
        elif cw_rate >= 0.3:
            median_off = float(np.median(probe_offsets)) if probe_offsets else 0.0
            auto_detected_mode = "WEAK"
            weak = True
            carrier_hint = median_off
            hint_bw = max(hint_bw, 50000)
            stable_msg = "stable" if cw_stable else "scattered"
            print(f"  [auto-detect] Marginal CW signal "
                  f"({probe_ok}/{n_probe} blocks, {stable_msg}, "
                  f"median {median_off:+.0f} Hz)")
            print(f"  [auto-detect] Using Viterbi + CW refinement...")
        else:
            # Both failed → switch to weak mode with auto carrier search
            auto_detected_mode = "WEAK"
            weak = True
            print(f"  [auto-detect] No strong signal "
                  f"(CW: {probe_ok}/{n_probe}, OQPSK: {probe_ok_q}/{n_probe})")
            print(f"  [auto-detect] Switching to weak signal mode "
                  f"(Viterbi ridge tracker)...")

            if carrier_hint is None:
                print(f"  [auto-detect] Scanning for signal...")
                cs = _coarse_dc_scan()
                if cs is not None:
                    carrier_hint = cs[0]
                    hint_bw = cs[1]
                    print(f"  [auto-detect] Found signal at "
                          f"{carrier_hint:+.0f} Hz (accum SNR "
                          f"{cs[2]:.1f} dB)")
                    print(f"  [auto-detect] Search window: "
                          f"{carrier_hint:+.0f} ± {hint_bw:.0f} Hz")
                else:
                    print(f"  [auto-detect] No signal found in "
                          f"coarse scan, using full band...")

    # ------------------------------------------------------------------
    # WEAK mode: spectrogram + Viterbi ridge tracker
    # ------------------------------------------------------------------
    if weak:
        # Exclude DC region in Viterbi — DC spike from SDR can hijack tracking
        # DC handling for Viterbi: use spectral subtraction when DC spike
        # is strong (>10 dB) — it removes the stationary pedestal while
        # preserving the drifting carrier. When DC spike is mild or absent,
        # use a fixed exclusion zone instead.
        use_spectral_sub = dc_snr > 10.0
        if use_spectral_sub:
            weak_dc_excl = 0.0  # spectral subtraction handles DC
        else:
            weak_dc_excl = max(probe_dc_excl * 15, 500.0)

        # OQPSK+WEAK: scale carrier hint and search window by 4×
        # IQ^4 moves carrier from f to 4*f, so search at 4× offset
        vit_carrier_hint = carrier_hint
        vit_hint_bw = hint_bw
        vit_search_bw = search_bw
        vit_max_drift = max_drift
        if oqpsk:
            if vit_carrier_hint is not None:
                vit_carrier_hint = vit_carrier_hint * 4.0
                vit_hint_bw = vit_hint_bw * 4.0
            if vit_search_bw is not None:
                vit_search_bw = min(vit_search_bw * 4.0, sample_rate * 0.80)
            vit_max_drift = vit_max_drift * 4.0
            weak_dc_excl = max(weak_dc_excl, 500.0)  # IQ^4 creates DC from modulation
            print(f"\n  [OQPSK+WEAK] IQ^4 carrier recovery + Viterbi tracking")
            print(f"  [OQPSK+WEAK] Frequencies in IQ^4 domain (÷4 after tracking)")

        print(f"\n  Building spectrogram ({n_blocks} frames)...", flush=True)
        psd_matrix, freq_axis, _ = _build_spectrogram(
            iq, sample_rate, eff_fft, n_welch_sub, spb, n_blocks,
            vit_carrier_hint, vit_hint_bw, vit_search_bw,
            dc_excl=weak_dc_excl, oqpsk=oqpsk)

        if use_spectral_sub:
            # Spectral subtraction: remove stationary features (DC spike,
            # LO phase noise pedestal). Drifting carrier survives because
            # it appears at different bins in each frame.
            median_spectrum = np.median(psd_matrix, axis=0)
            psd_matrix = psd_matrix - median_spectrum

        max_drift_per_step = vit_max_drift * integration_sec
        print(f"  Running Viterbi ridge tracker "
              f"(max drift {max_drift_per_step:.1f} Hz/step, "
              f"{len(freq_axis)} freq bins)...", flush=True)

        track_freq, track_snr = _viterbi_ridge(
            psd_matrix, freq_axis, max_drift_per_step, stack_k=weak_stack)

        # OQPSK+WEAK: divide tracked frequencies by 4 to get actual carrier
        if oqpsk:
            track_freq = track_freq / 4.0
            print(f"  [OQPSK+WEAK] Divided {len(track_freq)} tracked "
                  f"frequencies by 4")

        # -- CW refinement: use Viterbi track as per-block carrier hint ------
        # Viterbi gives robust detection (100% coverage), CW Welch gives better
        # frequency precision (sub-Hz vs ~10 Hz). Hybrid: try CW at each Viterbi
        # position, fall back to Viterbi result if CW fails.
        # In OQPSK+WEAK mode, CW refinement also uses IQ^4 (oqpsk=True).
        accum_snr_linear = np.sum(np.maximum(10.0 ** (track_snr / 10.0), 0.0))
        accum_snr_db = 10.0 * math.log10(max(1.0, accum_snr_linear))
        weak_accept_all = accum_snr_db >= 6.0

        # CW refinement bandwidth: ±500 Hz around Viterbi position
        # (in actual freq domain, not IQ^4 domain — estimate_carrier handles
        #  the ×4 internally when oqpsk=True)
        refine_bw = 500.0
        n_refined = 0
        measurements = []
        skipped = 0
        for i in range(min(len(track_freq), n_blocks)):
            t = start_time + timedelta(seconds=(i + 1) * integration_sec)
            viterbi_freq = center_freq + track_freq[i]
            viterbi_snr = float(track_snr[i])
            viterbi_offset = track_freq[i]

            # Try CW Welch refinement at Viterbi position
            freq_abs = viterbi_freq
            snr = viterbi_snr
            refined = False
            if weak_accept_all or viterbi_snr >= min_snr_db:
                try:
                    block = iq[i * spb : (i + 1) * spb]
                    cw_freq, cw_snr = estimate_carrier(
                        block, sample_rate, center_freq,
                        fft_size=eff_fft, n_sub=n_welch_sub,
                        carrier_hint=viterbi_offset, hint_bw=refine_bw,
                        excl_sidebands=False, dc_excl=probe_dc_excl,
                        oqpsk=oqpsk,
                    )
                    if cw_snr >= min_snr_db:
                        freq_abs = cw_freq
                        snr = cw_snr
                        refined = True
                        n_refined += 1
                except Exception:
                    pass

            offset = freq_abs - center_freq
            if weak_accept_all or snr >= min_snr_db:
                measurements.append((t, freq_abs, snr))
                tag = "OK"
            else:
                skipped += 1
                tag = "--"

            if i < 5 or (i + 1) % max(1, n_blocks // 20) == 0 or i == n_blocks - 1:
                r_tag = "R" if refined else " "
                print(f"  [{tag:>2s}]{r_tag}{i+1:5d}/{n_blocks}  "
                      f"{t.strftime('%Y-%jT%H:%M:%S.') + f'{t.microsecond//1000:03d}':26s}"
                      f"  offset={offset:+10.2f} Hz  SNR={snr:5.1f} dB")

        n_ok = len(measurements)
        print(f"\n  Accepted: {n_ok}/{n_blocks}  (skipped: {skipped})")
        if n_refined:
            print(f"  CW refined     : {n_refined}/{n_ok} "
                  f"({n_refined*100//max(1,n_ok)}%)")
        if measurements:
            offsets = [m[1] - center_freq for m in measurements]
            snrs = [m[2] for m in measurements]
            print(f"  Carrier offset : min={min(offsets):+.1f}  "
                  f"max={max(offsets):+.1f}  "
                  f"drift={max(offsets)-min(offsets):.1f} Hz")
            print(f"  SNR            : min={min(snrs):.1f}  "
                  f"max={max(snrs):.1f}  mean={sum(snrs)/len(snrs):.1f} dB")
            accum_snr = 10.0 * math.log10(max(1, sum(10**(s/10) for s in snrs)))
            print(f"  Accumulated SNR : {accum_snr:.1f} dB "
                  f"(track total across {n_ok} frames)")

        # Detect mode transitions: sustained drift rate reversal over 60-pt window
        # Uses longer window for robustness against Viterbi tracker oscillation
        weak_transitions = []
        if len(measurements) >= 200:
            offsets_arr = [m[1] - center_freq for m in measurements]
            win = 60
            for mi in range(win, len(offsets_arr) - win):
                # Drift rate before and after this point
                rate_before = (offsets_arr[mi] - offsets_arr[mi - win]) / (win * integration_sec)
                rate_after = (offsets_arr[mi + win] - offsets_arr[mi]) / (win * integration_sec)
                # Sign reversal with significant magnitude on both sides
                if (rate_before * rate_after < 0
                        and abs(rate_before) > 3.0 and abs(rate_after) > 3.0
                        and abs(rate_before - rate_after) > 8.0):
                    # Debounce: skip if too close to previous transition (5 min)
                    if (not weak_transitions
                            or (measurements[mi][0] - weak_transitions[-1][0]).total_seconds() > 300):
                        weak_transitions.append((
                            measurements[mi][0],
                            measurements[mi][1],
                            measurements[mi][1],
                        ))
        if weak_transitions:
            print(f"  Mode transitions: {len(weak_transitions)}")
            for mt_t, mt_from, mt_to in weak_transitions:
                print(f"    {mt_t.strftime('%Y-%jT%H:%M:%S')}: "
                      f"{mt_from/1e6:.6f} MHz "
                      f"(drift rate reversal)")

        return measurements, weak_transitions

    measurements = []
    skipped = 0
    rejected_snrs = []   # SNR of rejected blocks (below threshold)
    probe_raw = []       # (t, freq_abs, snr) -- all blocks in probe phase
    n_carrier_mode = 0   # blocks detected via direct carrier (auto mode)
    n_oqpsk_mode   = 0   # blocks detected via OQPSK IQ^4 (auto mode)
    mode_transitions = []  # list of (datetime, from_freq, to_freq) for TDM COMMENT

    # DC exclusion: apply in CW mode when no explicit carrier hint was given
    # (user-supplied carrier_hint means DC is far from the signal)
    user_carrier_hint = original_carrier_hint  # original CLI value (before auto-detect)
    # Use DC exclusion measured during auto-detect probe, or 0 if no spike found
    dc_excl = probe_dc_excl if (not oqpsk and not centroid) else 0.0

    # Carrier tracking: after probe phase, track carrier to prevent DC spur jumps
    tracking_offset = None   # last accepted carrier offset from center [Hz]
    tracking_bw = None       # adaptive tracking bandwidth [Hz]
    TRACKING_MAX_JUMP = 500  # max Hz jump per block before flagging transition

    use_tty = interactive and sys.stdout.isatty() and sys.stdin.isatty()
    probe_n = min(20, max(10, n_blocks // 10))
    # Probe phase always runs (adaptive welch-sub tuning works even without TTY)
    probe_done = probe_n >= n_blocks

    t0_proc = time.time()

    for i in range(n_blocks):
        block = iq[i*spb : (i+1)*spb]
        # Timestamp = end of integration window
        t = start_time + timedelta(seconds=(i + 1) * integration_sec)

        # Carrier tracking: after probe phase, use tracking hint to follow signal
        eff_hint = carrier_hint
        eff_hint_bw = hint_bw
        if probe_done and tracking_offset is not None and user_carrier_hint is None:
            eff_hint = tracking_offset
            eff_hint_bw = tracking_bw

        try:
            if auto:
                # Attempt 1: direct carrier search
                freq_abs, snr = estimate_carrier(
                    block, sample_rate, center_freq,
                    fft_size=eff_fft, n_sub=n_welch_sub,
                    search_bw=search_bw, carrier_hint=eff_hint,
                    hint_bw=eff_hint_bw, excl_sidebands=excl_sidebands,
                    oqpsk=False, dc_excl=dc_excl,
                )
                block_mode = 'C'
                if snr < min_snr_db:
                    # Attempt 2: OQPSK IQ^4 -- extra +2 dB margin to avoid false detections
                    freq_q, snr_q = estimate_carrier(
                        block, sample_rate, center_freq,
                        fft_size=eff_fft, n_sub=n_welch_sub,
                        search_bw=search_bw, carrier_hint=eff_hint,
                        hint_bw=eff_hint_bw, excl_sidebands=False,
                        oqpsk=True, dc_excl=dc_excl,
                    )
                    if snr_q >= min_snr_db + 2.0:   # +2 dB margin for OQPSK
                        freq_abs, snr, block_mode = freq_q, snr_q, 'Q'
            else:
                freq_abs, snr = estimate_carrier(
                    block, sample_rate, center_freq,
                    fft_size=eff_fft, n_sub=n_welch_sub,
                    search_bw=search_bw, carrier_hint=eff_hint,
                    hint_bw=eff_hint_bw, excl_sidebands=excl_sidebands,
                    oqpsk=oqpsk, centroid=centroid, dc_excl=dc_excl,
                )
                block_mode = 'M' if centroid else ('Q' if oqpsk else 'C')
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
            # -- Mode transition detection ------------------------------------
            if tracking_offset is not None and probe_done:
                jump = abs(offset - tracking_offset)
                if jump > TRACKING_MAX_JUMP:
                    mode_transitions.append((t, tracking_offset + center_freq, freq_abs))
                    if use_tty:
                        print(f"\n  [TRANSITION] block {i+1}: "
                              f"jump {jump:+.0f} Hz "
                              f"({tracking_offset:+.0f} -> {offset:+.0f} Hz)")
                    else:
                        print(f"  [TRANSITION] block {i+1}: "
                              f"jump {jump:+.0f} Hz "
                              f"({tracking_offset:+.0f} -> {offset:+.0f} Hz)")

            # -- Update carrier tracking --------------------------------------
            tracking_offset = offset
            # Tracking bandwidth: ±5000 Hz initially, narrows to ±2000 Hz
            # after we have enough measurements for drift estimation
            if len(measurements) < 10:
                tracking_bw = 5000.0
            else:
                # Estimate drift rate from last 10 measurements
                recent = measurements[-10:]
                dt_sec = (recent[-1][0] - recent[0][0]).total_seconds()
                if dt_sec > 0:
                    drift_rate = abs(recent[-1][1] - recent[0][1]) / dt_sec
                    # bw = max(2000, 3× expected drift per block + margin)
                    tracking_bw = max(2000.0, 3.0 * drift_rate * integration_sec + 500.0)
                else:
                    tracking_bw = 5000.0

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
                                oqpsk=oqpsk, centroid=centroid,
                                dc_excl=dc_excl,
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
                                        oqpsk=oqpsk, centroid=centroid,
                                        dc_excl=dc_excl,
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
                                            oqpsk=oqpsk, centroid=centroid,
                                            dc_excl=dc_excl,
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

            # Initialize carrier tracking from probe results
            if measurements and user_carrier_hint is None:
                tracking_offset = measurements[-1][1] - center_freq
                tracking_bw = 5000.0

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
        if mode_transitions:
            print(f"  Mode transitions: {len(mode_transitions)}")
            for mt_t, mt_from, mt_to in mode_transitions:
                print(f"    {_dt_to_tdm(mt_t)}: "
                      f"{mt_from/1e6:.6f} -> {mt_to/1e6:.6f} MHz")

    return measurements, mode_transitions


# ---------------------------------------------------------------------------
# Post-processing: Horizons validation, 2-way detection, LO drift correction
# ---------------------------------------------------------------------------

# Spacecraft name → JPL Horizons SPK ID mapping
_SPACECRAFT_SPK = {
    'ORION': '-1024', 'ARTEMIS': '-1024',
    'LRO': '-85',
    'KPLO': '-155', 'DANURI': '-155',
    'CAPSTONE': '-1176',
    'LADEE': '-12',
    'SLIM': '-240',
    'WIND': '-8',
    'ACE': '-92',
    'SOHO': '-21',
    'DSCOVR': '-78',
    'STEREO-A': '-234', 'STEREOA': '-234',
    'JWST': '-170',
}

C_KMS = 299_792.458  # speed of light [km/s]


def _query_horizons(spk_id, site_coord, t_start_str, t_stop_str):
    """Query JPL Horizons for range-rate and elevation.
    Returns list of (datetime, deldot_km/s, elevation_deg) or None.
    """
    params = {
        'format':      'json',
        'COMMAND':     f"'{spk_id}'",
        'OBJ_DATA':    "'NO'",
        'MAKE_EPHEM':  "'YES'",
        'TABLE_TYPE':  "'OBSERVER'",
        'CENTER':      "'coord@399'",
        'COORD_TYPE':  "'GEODETIC'",
        'SITE_COORD':  f"'{site_coord}'",
        'START_TIME':  f"'{t_start_str}'",
        'STOP_TIME':   f"'{t_stop_str}'",
        'STEP_SIZE':   "'1m'",
        'QUANTITIES':  "'4,20'",
        'CAL_FORMAT':  "'CAL'",
        'TIME_DIGITS': "'MINUTES'",
    }
    url = ('https://ssd.jpl.nasa.gov/api/horizons.api?'
           + urllib.parse.urlencode(params))
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read().decode())
    except Exception:
        return None

    result = data.get('result', '')
    if 'No ephemeris' in result or 'Cannot' in result:
        return None

    rows = []
    in_data = False
    for line in result.splitlines():
        if '$$SOE' in line:
            in_data = True
            continue
        if '$$EOE' in line:
            break
        if not in_data:
            continue
        m = re.match(
            r'\s*(\d{4}-\w{3}-\d{2}\s+\d{2}:\d{2})'
            r'\s+\S+\s+([-\d.]+)\s+([-\d.]+)\s+[-\d.]+\s+([-\d.]+)',
            line
        )
        if m:
            dt = datetime.strptime(m.group(1), '%Y-%b-%d %H:%M').replace(
                tzinfo=timezone.utc)
            elev = float(m.group(3))
            deldot = float(m.group(4))
            rows.append((dt, deldot, elev))
    return rows if rows else None


def validate_with_horizons(measurements, center_freq_hz, spacecraft_name,
                           location_str, mode_transitions=None):
    """Post-process: query Horizons, detect 1-way vs 2-way, estimate LO drift.

    When classified coherent/non-coherent transitions are present, validates
    each segment with its proper Doppler model (2-way for coherent, 1-way
    for non-coherent).

    Returns dict with: doppler_mode, rms, dc_offset, lo_drift_hz_per_s,
    corrected_measurements (or None if correction applied).
    """
    spk_id = _SPACECRAFT_SPK.get(spacecraft_name.upper())
    if not spk_id:
        print(f"\n  [validate] Unknown spacecraft '{spacecraft_name}' "
              f"— skipping Horizons check.")
        return None

    # Convert location from lat,lon,alt to lon,lat,alt (Horizons format)
    parts = location_str.split(',')
    if len(parts) < 2:
        return None
    lat, lon = parts[0], parts[1]
    alt = parts[2] if len(parts) >= 3 else '0'
    # Horizons wants lon,lat,alt_km
    try:
        alt_km = float(alt) / 1000.0 if float(alt) > 10 else float(alt)
    except ValueError:
        alt_km = 0.0
    site_coord = f"{lon},{lat},{alt_km}"

    # Ensure query window is at least 2 minutes for Horizons step_size='1m'
    t0_q = measurements[0][0] - timedelta(minutes=1)
    t1_q = measurements[-1][0] + timedelta(minutes=1)
    t_start = t0_q.strftime('%Y-%m-%d %H:%M')
    t_stop = t1_q.strftime('%Y-%m-%d %H:%M')

    print(f"\n  [validate] Querying JPL Horizons for {spacecraft_name} "
          f"(SPK {spk_id})...", end='', flush=True)
    hor = _query_horizons(spk_id, site_coord, t_start, t_stop)
    if not hor:
        print(" no ephemeris available.")
        return None
    print(f" {len(hor)} points.")

    # Check for transponder transitions → per-segment validation
    # Dispatch for ANY transition (including 'unknown' from drift-rate classifier)
    # — _validate_segments will self-classify using Horizons 1-way vs 2-way RMS
    if mode_transitions:
        return _validate_segments(measurements, center_freq_hz, hor,
                                  mode_transitions)

    # --- Single-segment: test 1-way and 2-way ---
    best = None
    for mode, scale in [('1-way', 1.0), ('2-way', 2.0)]:
        hor_dop = [(t, scale * (-dd * center_freq_hz / C_KMS))
                   for t, dd, _ in hor]

        pairs = []
        for t_m, f_m, _ in measurements:
            f_offset = f_m - center_freq_hz  # offset from SDR center
            best_dt, best_hf = None, None
            for t_h, d_h in hor_dop:
                dt = abs((t_m - t_h).total_seconds())
                if dt <= 90 and (best_dt is None or dt < best_dt):
                    best_dt = dt
                    best_hf = d_h
            if best_hf is not None:
                pairs.append((t_m, f_offset, best_hf, f_offset - best_hf))

        if not pairs:
            continue

        diffs = [p[3] for p in pairs]
        dc = sum(diffs) / len(diffs)
        res = [d - dc for d in diffs]
        rms_c = math.sqrt(sum(r**2 for r in res) / len(res))

        # Try linear LO drift compensation
        rms_best = rms_c
        drift_rate = 0.0
        drift_coeffs = None
        if len(pairs) >= 20 and rms_c > 100.0:
            try:
                t0 = pairs[0][0]
                t_sec = [(p[0] - t0).total_seconds() for p in pairs]
                poly = np.polyfit(t_sec, diffs, 1)  # [slope, intercept]
                coeffs = [poly[1], poly[0]]  # [intercept, slope]
                rate = abs(coeffs[1])
                if rate > 0.1:
                    vals = [coeffs[0] + coeffs[1]*t for t in t_sec]
                    r_lin = [d - v for d, v in zip(diffs, vals)]
                    rms_l = math.sqrt(sum(r**2 for r in r_lin) / len(r_lin))
                    if rms_l < rms_c * 0.7:
                        rms_best = rms_l
                        drift_rate = coeffs[1]
                        drift_coeffs = coeffs
                        dc = coeffs[0]
            except Exception:
                pass

        if best is None or rms_best < best['rms']:
            best = {
                'mode': mode,
                'rms': rms_best,
                'rms_const': rms_c,
                'dc_offset': dc,
                'drift_rate': drift_rate,
                'drift_coeffs': drift_coeffs,
                'n_pairs': len(pairs),
                'pairs': pairs,
            }

    if not best:
        print("  [validate] No time overlap with Horizons.")
        return None

    # --- Auto-segment: if RMS is high, try to find a breakpoint ---
    # This detects transponder mode changes (1-way ↔ 2-way) that produce
    # a frequency step spread over ~1 min, missed by the block-level jump detector.
    rms_check = best.get('rms_const', best['rms'])  # use pre-drift-fit RMS
    if rms_check > 50.0 and len(best['pairs']) >= 40:
        print(f"  [validate] RMS {rms_check:.1f} Hz (pre-fit) — "
              f"searching for mode transition...")
        diffs = [p[3] for p in best['pairs']]
        n = len(diffs)
        best_split_rms = rms_check
        best_split_idx = None
        # Try split points from 20% to 80%
        for split in range(n // 5, 4 * n // 5):
            seg1 = diffs[:split]
            seg2 = diffs[split:]
            if len(seg1) < 10 or len(seg2) < 10:
                continue
            dc1 = sum(seg1) / len(seg1)
            dc2 = sum(seg2) / len(seg2)
            # DC offset must differ by > 50 Hz (real transition)
            if abs(dc1 - dc2) < 50:
                continue
            r1 = [d - dc1 for d in seg1]
            r2 = [d - dc2 for d in seg2]
            rms_combined = math.sqrt(
                (sum(r**2 for r in r1) + sum(r**2 for r in r2)) / n
            )
            if rms_combined < best_split_rms * 0.5:
                best_split_rms = rms_combined
                best_split_idx = split
        if best_split_idx is not None:
            t_split = best['pairs'][best_split_idx][0]
            # Create synthetic transition and dispatch to _validate_segments
            seg1_dc = sum(diffs[:best_split_idx]) / best_split_idx
            seg2_dc = sum(diffs[best_split_idx:]) / (n - best_split_idx)
            print(f"  [validate] Found transition at {t_split.strftime('%H:%M:%S')}: "
                  f"DC {seg1_dc:+.0f} → {seg2_dc:+.0f} Hz "
                  f"(split RMS {best_split_rms:.1f} Hz)")
            auto_transitions = [(
                t_split,
                center_freq_hz + seg1_dc,
                center_freq_hz + seg2_dc,
            )]
            return _validate_segments(measurements, center_freq_hz, hor,
                                      auto_transitions)

    # Report
    print(f"  [validate] Doppler mode    : {best['mode']}")
    print(f"  [validate] DC offset       : {best['dc_offset']:+.1f} Hz")
    tx_freq = center_freq_hz + best['dc_offset']
    print(f"  [validate] TX frequency    : {tx_freq/1e6:.6f} MHz")
    print(f"  [validate] RMS residual    : {best['rms']:.1f} Hz")
    if best['drift_coeffs']:
        print(f"  [validate] LO drift        : "
              f"{best['drift_rate']:+.2f} Hz/s "
              f"({best['drift_rate']*60:+.1f} Hz/min)")
        print(f"  [validate] RMS (no drift)  : {best['rms_const']:.1f} Hz")

    # Apply LO drift correction to measurements
    corrected = None
    if best['drift_coeffs']:
        t0 = measurements[0][0]
        corrected = []
        c = best['drift_coeffs']
        for t_m, f_m, snr in measurements:
            dt = (t_m - t0).total_seconds()
            lo_correction = c[1] * dt
            corrected.append((t_m, f_m - lo_correction, snr))
        print(f"  [validate] Applied LO drift correction to {len(corrected)} "
              f"measurements.")

    best['corrected'] = corrected
    return best


def _validate_segments(measurements, center_freq_hz, hor, transitions):
    """Per-segment validation: try both 1-way and 2-way for each segment,
    pick the mode that gives the lowest RMS per segment."""
    C_KMS = 299792.458

    # Split measurements at transition points
    seg_boundaries = [0]
    for ct in transitions:
        t_trans = ct[0]
        for mi, (mt, _, _) in enumerate(measurements):
            if mt > t_trans and mi > seg_boundaries[-1]:
                seg_boundaries.append(mi)
                break
    seg_boundaries.append(len(measurements))

    raw_segments = []
    for si in range(len(seg_boundaries) - 1):
        seg_meas = measurements[seg_boundaries[si]:seg_boundaries[si+1]]
        if seg_meas:
            raw_segments.append(seg_meas)

    def _eval_segment(seg_meas, scale):
        """Evaluate segment with given Doppler scale, return (rms, dc, pairs)."""
        hor_dop = [(t, scale * (-dd * center_freq_hz / C_KMS))
                   for t, dd, _ in hor]
        pairs = []
        for t_m, f_m, _ in seg_meas:
            f_offset = f_m - center_freq_hz
            best_dt, best_hf = None, None
            for t_h, d_h in hor_dop:
                dt = abs((t_m - t_h).total_seconds())
                if dt <= 90 and (best_dt is None or dt < best_dt):
                    best_dt = dt
                    best_hf = d_h
            if best_hf is not None:
                pairs.append((t_m, f_offset, best_hf, f_offset - best_hf))
        if not pairs:
            return 1e12, 0.0, []
        diffs = [p[3] for p in pairs]
        dc = sum(diffs) / len(diffs)
        res = [d - dc for d in diffs]
        rms = math.sqrt(sum(r**2 for r in res) / len(res))
        return rms, dc, pairs

    # For each segment, pick the mode (1-way or 2-way) with lowest RMS
    seg_results = []
    total_rms_num = 0.0
    total_n = 0
    classified_transitions = []

    for si, seg_meas in enumerate(raw_segments):
        rms_1, dc_1, pairs_1 = _eval_segment(seg_meas, 1.0)
        rms_2, dc_2, pairs_2 = _eval_segment(seg_meas, 2.0)

        if rms_2 < rms_1 * 0.7:
            smode, doppler, rms, dc, pairs = ('coherent', '2-way',
                                               rms_2, dc_2, pairs_2)
        else:
            smode, doppler, rms, dc, pairs = ('non-coherent', '1-way',
                                               rms_1, dc_1, pairs_1)

        tx_freq = center_freq_hz + dc
        t0_seg = seg_meas[0][0].strftime('%H:%M:%S')
        t1_seg = seg_meas[-1][0].strftime('%H:%M:%S')
        print(f"  [validate] Segment {si+1}: {smode} ({doppler}) "
              f"{t0_seg}-{t1_seg}: "
              f"{len(pairs)} pts, RMS={rms:.1f} Hz, "
              f"DC={dc:+.1f} Hz, TX={tx_freq/1e6:.6f} MHz"
              f"  (1-way RMS={rms_1:.1f}, 2-way RMS={rms_2:.1f})")

        res = [p[3] - dc for p in pairs]
        total_rms_num += sum(r**2 for r in res)
        total_n += len(res)

        seg_results.append({
            'mode': doppler,
            'transponder': smode,
            'rms': rms,
            'dc_offset': dc,
            'n_pairs': len(pairs),
        })

        # Build classified transition labels
        if si > 0:
            prev_mode = seg_results[si - 1]['transponder']
            if prev_mode == 'coherent' and smode == 'non-coherent':
                label = 'coh_to_noncoh'
            elif prev_mode == 'non-coherent' and smode == 'coherent':
                label = 'noncoh_to_coh'
            else:
                label = 'unknown'
            if si - 1 < len(transitions):
                t = transitions[si - 1]
                classified_transitions.append(
                    (t[0], t[1], t[2], label))

    if total_n > 0:
        combined_rms = math.sqrt(total_rms_num / total_n)
        print(f"  [validate] Combined RMS    : {combined_rms:.1f} Hz "
              f"({total_n} pts)")
    else:
        combined_rms = 0.0

    return {
        'mode': 'per-segment',
        'rms': combined_rms,
        'rms_const': combined_rms,
        'dc_offset': 0.0,
        'drift_rate': 0.0,
        'drift_coeffs': None,
        'n_pairs': total_n,
        'pairs': [],
        'corrected': None,
        'segments': seg_results,
        'classified_transitions': classified_transitions,
    }


# ---------------------------------------------------------------------------
# Coherent / non-coherent transponder classification
# ---------------------------------------------------------------------------

def _classify_transponder_transitions(measurements, transitions, integration_sec):
    """Classify mode transitions as coherent→non-coherent or vice versa.

    In coherent mode (DSN uplink locked), observed Doppler ≈ 2× one-way.
    At a coherent→non-coherent transition the drift rate approximately halves.
    """
    if not transitions or len(measurements) < 100:
        return transitions

    freqs = [m[1] for m in measurements]
    times_s = [(m[0] - measurements[0][0]).total_seconds() for m in measurements]

    def _drift_rate(i_start, i_end):
        """Linear regression slope (Hz/s) over measurement indices."""
        n = i_end - i_start
        if n < 10:
            return None
        dt = times_s[i_end - 1] - times_s[i_start]
        if dt < 5:
            return None
        t = [times_s[j] - times_s[i_start] for j in range(i_start, i_end)]
        f = [freqs[j] for j in range(i_start, i_end)]
        st = sum(t); sf = sum(f)
        stt = sum(ti * ti for ti in t)
        stf = sum(ti * fi for ti, fi in zip(t, f))
        det = n * stt - st * st
        if abs(det) < 1e-12:
            return None
        return (n * stf - st * sf) / det

    classified = []
    for trans in transitions:
        t_trans = trans[0]
        t_s = (t_trans - measurements[0][0]).total_seconds()
        idx = min(range(len(times_s)), key=lambda i: abs(times_s[i] - t_s))

        # Windows: before = 60 pts ending 5 before transition
        #          after  = 60 pts starting 90 after (skip settling)
        win = min(60, max(10, idx // 2))
        skip = 90

        ib_end = max(0, idx - 5)
        ib_start = max(0, ib_end - win)
        ia_start = min(len(measurements), idx + skip)
        ia_end = min(len(measurements), ia_start + win)

        rate_before = _drift_rate(ib_start, ib_end)
        rate_after = _drift_rate(ia_start, ia_end)

        label = 'unknown'
        if (rate_before is not None and rate_after is not None
                and abs(rate_after) > 0.3):
            ratio = abs(rate_before) / abs(rate_after)
            if 1.3 <= ratio <= 3.5:
                label = 'coh_to_noncoh'
            elif 0.3 <= ratio <= 0.75:
                label = 'noncoh_to_coh'
            print(f"  [transponder] Transition at {_dt_to_tdm(t_trans)}: "
                  f"rate before={rate_before:+.2f} Hz/s, "
                  f"after={rate_after:+.2f} Hz/s, "
                  f"ratio={ratio:.2f} → {label}")
        else:
            reason = ("rates too small" if rate_before is not None
                      else "insufficient data")
            print(f"  [transponder] Transition at {_dt_to_tdm(t_trans)}: "
                  f"{reason} → {label}")

        classified.append((trans[0], trans[1], trans[2], label))

    return classified


# ---------------------------------------------------------------------------
# TDM output
# ---------------------------------------------------------------------------

def write_tdm(measurements, output_path, station_name, center_freq_hz,
              integration_sec, originator=None, dsn_station=None, comment=None,
              participant_1=None, mode_transitions=None):
    """Write CCSDS TDM v2.0 KVN file.

    When classified coherent/non-coherent transitions are present, writes
    multiple observation data segments (separate META+DATA blocks per CCSDS
    503.0-B-2 section 3.1).
    """
    if not measurements:
        raise ValueError("No measurements to write")

    freq_offset = round(center_freq_hz)
    now_utc = datetime.now(timezone.utc)
    orig    = originator or station_name

    p1 = participant_1 or "ORION"
    if dsn_station:
        part_lines = [f"PARTICIPANT_1          = {dsn_station}",
                      f"PARTICIPANT_2          = {p1}",
                      f"PARTICIPANT_3          = {station_name}"]
        path_str   = "1,2,3"
        data_kw    = "RECEIVE_FREQ_3"
    else:
        part_lines = [f"PARTICIPANT_1          = {p1}",
                      f"PARTICIPANT_2          = {station_name}"]
        path_str   = "1,2"
        data_kw    = "RECEIVE_FREQ_2"

    # Check for classified transponder transitions
    classified = [t for t in (mode_transitions or [])
                  if len(t) >= 4 and t[3] in ('coh_to_noncoh', 'noncoh_to_coh')]

    # --- Header (CCSDS 503.0-B-2 section 3.2, table 3-2) ---
    lines = [
        "CCSDS_TDM_VERS = 2.0",
    ]
    if comment:
        for ln in comment.strip().splitlines():
            lines.append(f"COMMENT {ln}")
    lines += [
        f"CREATION_DATE  = {_dt_to_tdm(now_utc)}Z",
        f"ORIGINATOR     = {orig}",
        "",
    ]

    if classified:
        # --- Multi-segment output (coherent / non-coherent) ---
        # Determine mode of first segment from the first transition
        first_label = classified[0][3]
        first_mode = ('coherent' if first_label == 'coh_to_noncoh'
                      else 'non-coherent')

        # Build list of (start_idx, end_idx, mode) segments
        segments = []
        seg_start = 0
        current_mode = first_mode
        for ct in classified:
            t_trans = ct[0]
            # Find split index: first measurement AFTER transition time
            split_idx = None
            for mi, (mt, _, _) in enumerate(measurements):
                if mt > t_trans and mi > seg_start:
                    split_idx = mi
                    break
            if split_idx is None:
                continue
            segments.append((seg_start, split_idx, current_mode))
            seg_start = split_idx
            current_mode = ('non-coherent' if current_mode == 'coherent'
                            else 'coherent')
        # Last segment
        segments.append((seg_start, len(measurements), current_mode))

        seg_info = []
        for si, se, smode in segments:
            seg_meas = measurements[si:se]
            if not seg_meas:
                continue
            t_start = seg_meas[0][0]
            t_stop  = seg_meas[-1][0]
            is_coh  = (smode == 'coherent')

            mode_comment = (
                "Coherent transponder (2-way Doppler)"
                if is_coh else
                "Non-coherent transponder (1-way Doppler)"
            )

            lines += [
                "META_START",
                f"COMMENT {mode_comment}",
                "TIME_SYSTEM            = UTC",
                f"START_TIME             = {_dt_to_tdm(t_start)}",
                f"STOP_TIME              = {_dt_to_tdm(t_stop)}",
                *part_lines,
                "MODE                   = SEQUENTIAL",
                f"PATH                   = {path_str}",
            ]
            if is_coh:
                lines += [
                    f"TURNAROUND_NUMERATOR   = {TURNAROUND_NUMERATOR}",
                    f"TURNAROUND_DENOMINATOR = {TURNAROUND_DENOMINATOR}",
                ]
            lines += [
                f"INTEGRATION_INTERVAL   = {integration_sec:.1f}",
                "INTEGRATION_REF        = END",
                f"FREQ_OFFSET            = {freq_offset:.1f}",
                "META_STOP",
                "",
                "DATA_START",
            ]
            for (t, fa, snr) in seg_meas:
                lines.append(
                    f"{data_kw} = {_dt_to_tdm(t)}  {fa - freq_offset:+.3f}")
            lines += ["DATA_STOP", ""]

            seg_info.append((smode, len(seg_meas),
                             (t_stop - t_start).total_seconds()))
    else:
        # --- Single-segment output (original behavior) ---
        t_start = measurements[0][0]
        t_stop  = measurements[-1][0]

        lines += [
            "META_START",
            "TIME_SYSTEM            = UTC",
            f"START_TIME             = {_dt_to_tdm(t_start)}",
            f"STOP_TIME              = {_dt_to_tdm(t_stop)}",
            *part_lines,
            "MODE                   = SEQUENTIAL",
            f"PATH                   = {path_str}",
            f"TURNAROUND_NUMERATOR   = {TURNAROUND_NUMERATOR}",
            f"TURNAROUND_DENOMINATOR = {TURNAROUND_DENOMINATOR}",
            f"INTEGRATION_INTERVAL   = {integration_sec:.1f}",
            "INTEGRATION_REF        = END",
            f"FREQ_OFFSET            = {freq_offset:.1f}",
            "META_STOP",
            "",
        ]

        lines.append("DATA_START")
        if mode_transitions:
            for mt in mode_transitions:
                mt_t, mt_freq = mt[0], mt[1]
                mt_info = mt[2] if len(mt) >= 3 else mt_freq
                lines.append(f"COMMENT Mode transition at {_dt_to_tdm(mt_t)} "
                             f"near {mt_freq/1e6:.6f} MHz ({mt_info})")
        for (t, fa, snr) in measurements:
            lines.append(
                f"{data_kw} = {_dt_to_tdm(t)}  {fa - freq_offset:+.3f}")
        lines += ["DATA_STOP", ""]

        seg_info = None

    with open(output_path, "w", encoding="ascii") as f:
        f.write("\n".join(lines))

    dur = (measurements[-1][0] - measurements[0][0]).total_seconds()
    print(f"\n{'='*64}")
    print(f"  TDM written  : {output_path}")
    print(f"  Measurements : {len(measurements)}")
    print(f"  Duration     : {dur:.0f} s ({dur/60:.1f} min)")
    if seg_info:
        for smode, sn, sdur in seg_info:
            print(f"  Segment      : {smode} — {sn} pts, "
                  f"{sdur:.0f} s ({sdur/60:.1f} min)")
    else:
        print(f"  Mode         : "
              f"{'3-way ('+dsn_station+')' if dsn_station else '1-way'}")
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
    p.add_argument("--centroid", action="store_true",
                   help=(
                       "Spectral centroid mode for modulated signals (BPSK/QPSK telemetry). "
                       "Instead of finding a single CW peak, tracks the power-weighted "
                       "center frequency of the signal band. "
                       "Use with --carrier-hint and --hint-bw to define the search window. "
                       "Works for SOHO, spacecraft with suppressed carrier, etc."
                   ))
    p.add_argument("--weak", action="store_true",
                   help=(
                       "Weak signal mode (Viterbi ridge tracker). "
                       "Builds a spectrogram and finds the optimal frequency track "
                       "using dynamic programming, exploiting signal continuity across "
                       "time frames. Can detect signals below per-frame noise floor "
                       "(visible on waterfall but not in single-frame PSD). "
                       "Use with --carrier-hint and --hint-bw to limit search region. "
                       "Use --max-drift to set max Doppler rate [Hz/s] (default: 10)."
                   ))
    p.add_argument("--max-drift", type=float, default=10.0,
                   help=(
                       "Maximum Doppler drift rate [Hz/s] for --weak mode (default: 10). "
                       "Spacecraft at L1/L2: ~1-5 Hz/s, lunar: ~5-50 Hz/s."
                   ))
    p.add_argument("--weak-stack", type=int, default=1,
                   help=(
                       "Stack K consecutive frames before ridge tracking in --weak mode "
                       "(default: 1, no stacking). Use 3-10 for extra SNR boost."
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
    meas, transitions = process_iq(
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
        centroid        = args.centroid,
        weak            = args.weak,
        max_drift       = args.max_drift,
        weak_stack      = args.weak_stack,
    )

    if not meas:
        print("\nERROR: No measurements -- signal too weak or SNR threshold too high.")
        print("Try: --min-snr 1.5  --welch-sub 100  --carrier-hint <Hz>")
        sys.exit(1)

    # -- Classify transponder transitions -----------------------------------
    if transitions:
        transitions = _classify_transponder_transitions(
            meas, transitions, args.integration)

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
        f"{args.spacecraft or 'ORION'} Doppler tracking\n"
        f"{loc_line}"
        f"Source: {inp.name}\n"
        f"HW: {info.get('hw','?')} | "
        f"FFT={args.fft_size} Welch={args.welch_sub} int={args.integration}s | "
        f"mode={mode_str}"
    )
    write_tdm(meas, out, args.station, cf, args.integration,
              args.originator, args.dsn_station, args.comment or auto_cmt,
              participant_1=args.spacecraft, mode_transitions=transitions)

    # -- Post-processing: Horizons validation ----------------------------------
    if args.spacecraft and args.location:
        val = validate_with_horizons(meas, cf, args.spacecraft, args.location,
                                     mode_transitions=transitions)
        if val and val.get('classified_transitions'):
            # Horizons self-classified segments → rewrite TDM with proper labels
            ct = val['classified_transitions']
            print(f"\n  [rewrite] Rewriting TDM with Horizons-classified "
                  f"transponder segments...")
            write_tdm(meas, out, args.station, cf,
                      args.integration, args.originator, args.dsn_station,
                      args.comment or auto_cmt, participant_1=args.spacecraft,
                      mode_transitions=ct)
        elif val and val.get('corrected'):
            # Rewrite TDM with LO-drift-corrected measurements
            drift_cmt = (
                f"{auto_cmt}\n"
                f"LO drift corrected: {val['drift_rate']:+.2f} Hz/s "
                f"({val['mode']} Doppler, RMS {val['rms']:.1f} Hz)"
            )
            write_tdm(val['corrected'], out, args.station, cf,
                      args.integration, args.originator, args.dsn_station,
                      drift_cmt, participant_1=args.spacecraft,
                      mode_transitions=transitions)


if __name__ == "__main__":
    main()
