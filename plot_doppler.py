#!/usr/bin/env python3
"""Doppler comparison: our TDM (Welch) vs CAMRAS single-FFT, 2022-12-01 21:42 UTC."""

import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta

CENTER = 2_216_500_000.0  # Hz


def parse_dt(s):
    s = s.strip().rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def parse_doy(s):
    s = s.strip()
    s = re.sub(r'T(\d{2}:\d{2}:\d{2}):(\d+)', r'T\1.\2', s)
    m = re.match(r'(\d{4})-(\d{3})T(\d{2}):(\d{2}):(\d{2})(?:[.,](\d+))?', s)
    if not m:
        return None
    year, doy, hh, mm, ss, frac = m.groups()
    base = datetime(int(year), 1, 1, tzinfo=timezone.utc) + timedelta(days=int(doy) - 1)
    sec = int(ss) + (float('0.' + frac) if frac else 0.0)
    return base.replace(hour=int(hh), minute=int(mm), second=0) + timedelta(seconds=sec)


# -- Load our TDM (Welch) -----------------------------------------------------
our_times, our_freqs = [], []
with open('examples/generated_small.tdm') as f:
    for line in f:
        m = re.match(r'RECEIVE_FREQ_2\s*=\s*(\S+)\s+([\d.eE+-]+)', line.strip())
        if m:
            t = parse_doy(m.group(1))
            if t:
                our_times.append(t)
                our_freqs.append(float(m.group(2)))   # already relative to FREQ_OFFSET

# -- Load CAMRAS single-FFT doppler_20221201.txt ------------------------------
# Format: unix_ts, absolute_freq_Hz, power
sfft_times, sfft_freqs = [], []
clip_start = datetime(2022, 12, 1, 21, 40, 0, tzinfo=timezone.utc)
clip_end   = datetime(2022, 12, 1, 21, 46, 0, tzinfo=timezone.utc)

with open('camras_test/doppler_20221201.txt') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        if len(parts) < 2:
            continue
        try:
            ts  = float(parts[0])
            hz  = float(parts[1])
            t   = datetime.fromtimestamp(ts, tz=timezone.utc)
            if clip_start <= t <= clip_end:
                sfft_times.append(t)
                sfft_freqs.append(hz - CENTER)   # offset from center
        except ValueError:
            pass

print(f"Our TDM (Welch):      {len(our_times)} measurements -> {our_freqs}")
print(f"CAMRAS single-FFT:    {len(sfft_times)} measurements in window 21:40-21:46 UTC")
if sfft_freqs:
    valid = [f for f in sfft_freqs if abs(f) < 200_000]
    print(f"  range: {min(valid):+.0f} ... {max(valid):+.0f} Hz  (after discarding outliers)")

# -- Plot ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle('Artemis I Doppler — 2022-12-01 ~21:42 UTC, CAMRAS Dwingeloo\n'
             'Comparison: Welch (our TDM) vs single-FFT (CAMRAS reference)',
             fontsize=12, fontweight='bold')

# Single-FFT — grey dots (raw, noisy)
ax.scatter(sfft_times, sfft_freqs, s=6, color='silver', alpha=0.5,
           label='CAMRAS single-FFT (raw, 1 s)', zorder=2)

# Single-FFT — 5s rolling median for readability
if len(sfft_times) >= 5:
    import statistics
    med_times, med_freqs = [], []
    for i in range(2, len(sfft_times) - 2):
        window = [sfft_freqs[j] for j in range(i-2, i+3) if abs(sfft_freqs[j]) < 200_000]
        if window:
            med_times.append(sfft_times[i])
            med_freqs.append(statistics.median(window))
    ax.plot(med_times, med_freqs, color='darkorange', linewidth=1.5,
            label='CAMRAS single-FFT (5 s rolling median)', zorder=3)

# Our Welch TDM
ax.scatter(our_times, our_freqs, s=120, color='steelblue', zorder=5,
           label=f'SP5LOT Welch TDM  ({our_freqs[0]:+.1f} Hz)', marker='D')
for t, f in zip(our_times, our_freqs):
    ax.annotate(f'{f:+.1f} Hz', (t, f), textcoords='offset points',
                xytext=(8, -12), fontsize=9, color='steelblue', fontweight='bold')

ax.set_xlabel('UTC')
ax.set_ylabel('Doppler offset from 2216.500 MHz [Hz]')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax.tick_params(axis='x', rotation=25)
ax.grid(True, alpha=0.35)
ax.legend(fontsize=9)
ax.set_xlim(clip_start, clip_end)
ax.set_ylim(-60000, -30000)

# Mark IQ start time
iq_start = datetime(2022, 12, 1, 21, 42, 38, tzinfo=timezone.utc)
ax.axvline(iq_start, color='steelblue', linestyle='--', alpha=0.5, linewidth=1)
ax.text(iq_start, ax.get_ylim()[0] if ax.get_ylim()[0] > -1e6 else -60000,
        '  IQ start\n  21:42:38', fontsize=8, color='steelblue', va='bottom')

plt.tight_layout()
out = 'doppler_camras_dec1_comparison.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out}")
