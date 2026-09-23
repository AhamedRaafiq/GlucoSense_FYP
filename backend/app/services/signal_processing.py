import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, medfilt, find_peaks
from scipy.interpolate import interp1d
from typing import Dict, Any, List, Tuple
from ..config import settings

def remove_spikes(signal: np.ndarray) -> np.ndarray:
    """Step 3: Spike removal using median filter with kernel size 3."""
    return medfilt(signal, kernel_size=3)

def invert_signal(signal: np.ndarray) -> np.ndarray:
    """Step 4: Invert the PPG signal (MAX30102 outputs inverted)."""
    return signal * -1.0

def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Step 5: 4th-order lowpass Butterworth filter."""
    sos = butter(order, cutoff, btype='low', fs=fs, output='sos')
    return sosfiltfilt(sos, data)

def butter_highpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Step 7: 4th-order highpass Butterworth filter."""
    sos = butter(order, cutoff, btype='high', fs=fs, output='sos')
    return sosfiltfilt(sos, data)

def min_max_normalize(data: np.ndarray) -> np.ndarray:
    """Step 8: Min-max normalization (0 to 1)."""
    dmin, dmax = np.min(data), np.max(data)
    if dmax - dmin == 0:
        return np.zeros_like(data)
    return (data - dmin) / (dmax - dmin)

def detect_beats(signal: np.ndarray, fs: float) -> List[Tuple[int, int]]:
    """
    Detect beats in the PPG signal.
    Returns list of (start_idx, end_idx) tuples for each beat.
    """
    # Find systolic peaks
    min_dist = int(0.40 * fs)  # 0.40s minimum distance
    
    # Calculate prominence threshold (0.20 of peak-to-peak amplitude)
    sig_range = np.max(signal) - np.min(signal)
    prominence = 0.20 * sig_range if sig_range > 0 else 0.05
    
    peaks, _ = find_peaks(signal, distance=min_dist, prominence=prominence)
    
    # If not enough peaks, return empty
    if len(peaks) < 3:
        return []
        
    beats = []
    # Segments between successive peaks are considered beats (or valley to valley, peak to peak is standard)
    # Using peak-to-peak segmentation
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i+1]
        duration_sec = (end_idx - start_idx) / fs
        
        # Validate beat duration (0.35s to 1.50s)
        if 0.35 <= duration_sec <= 1.50:
            beats.append((start_idx, end_idx))
            
    return beats

def resample_beat(beat_signal: np.ndarray, target_len: int = 220) -> np.ndarray:
    """Resample a single beat to target length using linear interpolation."""
    original_indices = np.linspace(0, len(beat_signal) - 1, num=len(beat_signal))
    target_indices = np.linspace(0, len(beat_signal) - 1, num=target_len)
    interpolator = interp1d(original_indices, beat_signal, kind='linear')
    return interpolator(target_indices)

def calculate_ensemble_average(signal: np.ndarray, beats: List[Tuple[int, int]], target_len: int = 220) -> np.ndarray:
    """Step 11: Calculate ensemble average of resampled beats."""
    if not beats:
        return np.zeros(target_len)
        
    resampled_beats = []
    for start, end in beats:
        beat_segment = signal[start:end]
        resampled = resample_beat(beat_segment, target_len)
        resampled_beats.append(resampled)
        
    return np.mean(resampled_beats, axis=0)

def process_single_window(ir_raw: np.ndarray, red_raw: np.ndarray, fs: float = 400.0) -> Dict[str, Any]:
    """
    Process a single 15-second window of IR and RED signals.
    """
    log_messages = []
    
    # Step 3: Spike removal
    ir_no_spikes = remove_spikes(ir_raw)
    red_no_spikes = remove_spikes(red_raw)
    
    # Step 4: Invert signals
    ir_inverted = invert_signal(ir_no_spikes)
    red_inverted = invert_signal(red_no_spikes)
    
    # Step 5: Low-pass filter (16Hz)
    ir_lp = butter_lowpass_filter(ir_inverted, cutoff=16.0, fs=fs, order=4)
    red_lp = butter_lowpass_filter(red_inverted, cutoff=16.0, fs=fs, order=4)
    
    # Step 7: High-pass filter (0.5Hz)
    ir_hp = butter_highpass_filter(ir_lp, cutoff=0.5, fs=fs, order=4)
    red_hp = butter_highpass_filter(red_lp, cutoff=0.5, fs=fs, order=4)
    
    # Step 8: Normalize (MinMax)
    ir_normalized = min_max_normalize(ir_hp)
    red_normalized = min_max_normalize(red_hp)
    
    # Step 11: Beat detection & Ensemble averaging
    ir_beats = detect_beats(ir_normalized, fs)
    red_beats = detect_beats(red_normalized, fs)
    
    log_messages.append(f"Detected {len(ir_beats)} valid IR beats and {len(red_beats)} valid RED beats.")
    
    ir_ensemble = calculate_ensemble_average(ir_normalized, ir_beats, target_len=220)
    red_ensemble = calculate_ensemble_average(red_normalized, red_beats, target_len=220)
    
    # Calculate derivatives on ensemble averages
    # VPG (Velocity PPG - 1st derivative)
    ir_vpg = np.gradient(ir_ensemble)
    red_vpg = np.gradient(red_ensemble)
    
    # SDPPG (Acceleration PPG - 2nd derivative)
    ir_sdppg = np.gradient(ir_vpg)
    red_sdppg = np.gradient(red_vpg)
    
    return {
        "ir_filtered_full": ir_normalized.tolist(),
        "red_filtered_full": red_normalized.tolist(),
        "ir_ensemble": ir_ensemble.tolist(),
        "red_ensemble": red_ensemble.tolist(),
        "ir_vpg": ir_vpg.tolist(),
        "red_vpg": red_vpg.tolist(),
        "ir_sdppg": ir_sdppg.tolist(),
        "red_sdppg": red_sdppg.tolist(),
        "ir_beats": ir_beats,
        "red_beats": red_beats,
        "log": "\n".join(log_messages)
    }

def slice_raw_data_into_windows(df: pd.DataFrame, window_size_samples: int = 6000) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Slices raw PPG dataframe into non-overlapping windows of window_size_samples.
    df must contain 'IR' and 'RED' (or 'ir_value' and 'red_value') columns.
    """
    # Normalize column names
    cols = {c.upper(): c for c in df.columns}
    ir_col = cols.get('IR') or cols.get('IR_VALUE')
    red_col = cols.get('RED') or cols.get('RED_VALUE')
    
    if not ir_col or not red_col:
        raise ValueError("Dataframe must contain IR and RED columns.")
        
    ir_data = df[ir_col].values
    red_data = df[red_col].values
    
    num_windows = len(df) // window_size_samples
    windows = []
    
    for i in range(num_windows):
        start = i * window_size_samples
        end = start + window_size_samples
        windows.append((ir_data[start:end], red_data[start:end]))
        
    # If there's leftover data and we have zero windows, check if we have enough for at least one
    if num_windows == 0 and len(df) >= window_size_samples // 2:
        # Pad with zeros or duplicate last elements to make up window_size_samples
        padded_ir = np.pad(ir_data, (0, window_size_samples - len(ir_data)), 'edge')
        padded_red = np.pad(red_data, (0, window_size_samples - len(red_data)), 'edge')
        windows.append((padded_ir, padded_red))
        
    return windows
