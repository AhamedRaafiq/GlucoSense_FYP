import numpy as np
import scipy.stats as stats
from scipy.signal import welch, find_peaks
from typing import Dict, Any, List

def calculate_shannon_entropy(signal: np.ndarray, bins: int = 64) -> float:
    """Calculate Shannon Entropy of the signal."""
    # Ensure no NaN
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return 0.0
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

def calculate_spectral_entropy(signal: np.ndarray, fs: float) -> float:
    """Calculate Spectral Entropy using Welch PSD."""
    signal = signal[~np.isnan(signal)]
    if len(signal) < 16:
        return 0.0
    f, psd = welch(signal, fs, nperseg=min(len(signal), 256))
    psd_sum = np.sum(psd)
    if psd_sum == 0:
        return 0.0
    psd_norm = psd / psd_sum
    psd_norm = psd_norm[psd_norm > 0]
    return float(-np.sum(psd_norm * np.log2(psd_norm)))

def teager_energy_operator(signal: np.ndarray) -> np.ndarray:
    """Compute Teager Energy Operator: x[n]^2 - x[n-1]*x[n+1]."""
    if len(signal) < 3:
        return np.zeros_like(signal)
    teo = np.zeros_like(signal)
    teo[1:-1] = signal[1:-1]**2 - signal[:-2] * signal[2:]
    # Fill edge cases
    teo[0] = teo[1]
    teo[-1] = teo[-2]
    return teo

def get_harmonic_ratio(signal: np.ndarray) -> float:
    """Approximate harmonic ratio: ratio of fundamental peak to total energy."""
    fft_vals = np.abs(np.fft.rfft(signal))
    if len(fft_vals) < 3:
        return 0.0
    # Find peaks in FFT (ignoring DC component)
    peaks, _ = find_peaks(fft_vals[1:])
    if len(peaks) == 0:
        return 0.0
    fundamental_idx = peaks[0] + 1
    fundamental_energy = fft_vals[fundamental_idx]
    total_energy = np.sum(fft_vals)
    return float(fundamental_energy / total_energy) if total_energy > 0 else 0.0

def find_dicrotic_notch_pos(ensemble: np.ndarray, sdppg: np.ndarray) -> float:
    """
    Estimate dicrotic notch position as index where the second derivative (SDPPG)
    has a local maximum after the systolic peak (which is usually around 15-30% of beat).
    """
    peak_idx = np.argmax(ensemble)
    if peak_idx >= len(ensemble) - 5:
        return float(len(ensemble) // 2)
        
    # Search in the region after systolic peak
    search_region = sdppg[peak_idx:]
    if len(search_region) == 0:
        return float(peak_idx)
        
    # Find local maxima in SDPPG in the search region
    peaks, _ = find_peaks(search_region)
    if len(peaks) > 0:
        notch_idx = peak_idx + peaks[0]
    else:
        notch_idx = peak_idx + np.argmax(search_region)
        
    return float(notch_idx)

def calculate_time_features(ensemble: np.ndarray) -> Dict[str, float]:
    """Calculate rise time, decay time, and pulse width from ensemble average."""
    peak_idx = np.argmax(ensemble)
    valleys = find_peaks(-ensemble)[0]
    
    # Valley before peak
    start_idx = 0
    if len(valleys) > 0:
        prev_valleys = valleys[valleys < peak_idx]
        if len(prev_valleys) > 0:
            start_idx = prev_valleys[-1]
            
    # Valley after peak
    end_idx = len(ensemble) - 1
    if len(valleys) > 0:
        next_valleys = valleys[valleys > peak_idx]
        if len(next_valleys) > 0:
            end_idx = next_valleys[0]
            
    rise_time = float(peak_idx - start_idx) / 400.0  # At 400Hz
    decay_time = float(end_idx - peak_idx) / 400.0
    
    # Pulse width at half-max
    max_val = ensemble[peak_idx]
    min_val = ensemble[start_idx]
    half_max = min_val + 0.5 * (max_val - min_val)
    
    above_half = np.where(ensemble >= half_max)[0]
    pulse_width = float(len(above_half)) / 400.0 if len(above_half) > 0 else 0.1
    
    return {
        "rise_time": rise_time,
        "decay_time": decay_time,
        "pulse_width": pulse_width
    }

def extract_channel_features(filtered_full: np.ndarray, ensemble: np.ndarray, vpg: np.ndarray, sdppg: np.ndarray, beats: List[Any], fs: float = 400.0) -> Dict[str, float]:
    """Extract 18 base features for a single channel (RED or IR)."""
    # 1. Skewness
    skew = float(stats.skew(filtered_full)) if len(filtered_full) > 0 else 0.0
    
    # 2. Kurtosis
    kurt = float(stats.kurtosis(filtered_full)) if len(filtered_full) > 0 else 0.0
    
    # 3. Shannon Entropy
    shannon_ent = calculate_shannon_entropy(filtered_full)
    
    # 4. Spectral Entropy
    spectral_ent = calculate_spectral_entropy(filtered_full, fs)
    
    # Time features (rise, decay, pulse width) from ensemble
    time_feats = calculate_time_features(ensemble)
    
    # 5. Pulse width
    p_width = time_feats["pulse_width"]
    
    # Beat interval features
    ppi_list = []
    if beats:
        for start, end in beats:
            ppi_list.append((end - start) / fs)
            
    # 6. PPI (Peak-to-Peak Interval)
    ppi = float(np.mean(ppi_list)) if ppi_list else 0.8
    
    # 7. Systolic Amplitude (max - min of ensemble)
    sys_amp = float(np.max(ensemble) - np.min(ensemble)) if len(ensemble) > 0 else 1.0
    
    # 8. BPM
    bpm = 60.0 / ppi if ppi > 0 else 75.0
    
    # 9. HRV (SDNN in ms)
    hrv = float(np.std(ppi_list) * 1000.0) if len(ppi_list) > 1 else 30.0
    
    # TEO features
    teo = teager_energy_operator(filtered_full)
    
    # 10. TEO Mean
    teo_mean = float(np.mean(teo))
    
    # 11. TEO Std Dev
    teo_std = float(np.std(teo))
    
    # 12. 1st Derivative Mean (VPG)
    vpg_mean = float(np.mean(vpg))
    
    # 13. 2nd Derivative Mean (SDPPG)
    sdppg_mean = float(np.mean(sdppg))
    
    # 14. 2nd Derivative Skewness
    sdppg_skew = float(stats.skew(sdppg)) if len(sdppg) > 0 else 0.0
    
    # 15. Harmonic Ratio
    harmonic_ratio = get_harmonic_ratio(ensemble)
    
    # 17. Rise Time
    rise_time = time_feats["rise_time"]
    
    # 18. Decay Time
    decay_time = time_feats["decay_time"]
    
    # 19. Dicrotic Notch position
    dicrotic_notch = find_dicrotic_notch_pos(ensemble, sdppg)
    
    return {
        "Skewness": skew,
        "Kurtosis": kurt,
        "Shannon Entropy": shannon_ent,
        "Spectral Entropy": spectral_ent,
        "pulse width": p_width,
        "PPI": ppi,
        "systolic amplitude": sys_amp,
        "BPM": bpm,
        "HRV": hrv,
        "TEO Mean": teo_mean,
        "TEO std dev": teo_std,
        "1st_Derivative_Mean": vpg_mean,
        "2nd_Derivative_Mean": sdppg_mean,
        "2nd_Derivative_Skewness": sdppg_skew,
        "Harmonic ratio": harmonic_ratio,
        "Rise time": rise_time,
        "Decay time": decay_time,
        "Dicrotic notch": dicrotic_notch
    }

def calculate_ensemble_ratio(red_raw: np.ndarray, ir_raw: np.ndarray, red_hp: np.ndarray, ir_hp: np.ndarray) -> float:
    """
    Step 16: Ensemble Ratio (Red AC/DC / IR AC/DC).
    AC = standard deviation of the highpass signal (AC component)
    DC = mean of the raw signal (DC component)
    """
    red_ac = np.std(red_hp)
    red_dc = np.mean(red_raw)
    ir_ac = np.std(ir_hp)
    ir_dc = np.mean(ir_raw)
    
    if ir_ac == 0 or ir_dc == 0 or red_dc == 0:
        return 0.5  # Fallback
        
    red_ac_dc = red_ac / red_dc
    ir_ac_dc = ir_ac / ir_dc
    
    return float(red_ac_dc / ir_ac_dc)

def extract_features_from_window(
    ir_raw: np.ndarray, red_raw: np.ndarray,
    processed_results: Dict[str, Any], fs: float = 400.0
) -> Dict[str, float]:
    """
    Extract all features from a single processed window and engineer the 24 final features.
    """
    ir_full = np.array(processed_results["ir_filtered_full"])
    red_full = np.array(processed_results["red_filtered_full"])
    ir_ens = np.array(processed_results["ir_ensemble"])
    red_ens = np.array(processed_results["red_ensemble"])
    ir_vpg = np.array(processed_results["ir_vpg"])
    red_vpg = np.array(processed_results["red_vpg"])
    ir_sdp = np.array(processed_results["ir_sdppg"])
    red_sdp = np.array(processed_results["red_sdppg"])
    ir_beats = processed_results["ir_beats"]
    red_beats = processed_results["red_beats"]
    
    # Extract IR base features (18 features)
    ir_features = extract_channel_features(ir_full, ir_ens, ir_vpg, ir_sdp, ir_beats, fs)
    
    # Extract RED base features (needed for engineered ratios/differences)
    red_features = extract_channel_features(red_full, red_ens, red_vpg, red_sdp, red_beats, fs)
    
    # Step 16: Ensemble Ratio
    ens_ratio = calculate_ensemble_ratio(red_raw, ir_raw, red_full, ir_full)
    
    # Construct the 24 features
    engineered_features = {}
    
    # IR Base Features
    for name, val in ir_features.items():
        engineered_features[f"IR_{name}"] = val
        
    # Engineered Ratios/Differences
    # 1. Ratio_systolic_amplitude = Red_systolic_amplitude / IR_systolic_amplitude
    ir_sys = ir_features["systolic amplitude"]
    red_sys = red_features["systolic amplitude"]
    engineered_features["Ratio_systolic_amplitude"] = red_sys / ir_sys if ir_sys > 0 else 1.0
    
    # 2. Ratio_TEO_Mean = Red_TEO_Mean / IR_TEO_Mean
    ir_teo = ir_features["TEO Mean"]
    red_teo = red_features["TEO Mean"]
    engineered_features["Ratio_TEO_Mean"] = red_teo / ir_teo if ir_teo > 0 else 1.0
    
    # 3. Diff_2nd_Derivative_Mean = Red_2nd_Derivative_Mean - IR_2nd_Derivative_Mean
    engineered_features["Diff_2nd_Derivative_Mean"] = red_features["2nd_Derivative_Mean"] - ir_features["2nd_Derivative_Mean"]
    
    # 4. Diff_Spectral_Entropy = Red_Spectral Entropy - IR_Spectral Entropy
    engineered_features["Diff_Spectral_Entropy"] = red_features["Spectral Entropy"] - ir_features["Spectral Entropy"]
    
    # 5. Diff_Dicrotic_notch = Red_Dicrotic notch - IR_Dicrotic notch
    engineered_features["Diff_Dicrotic_notch"] = red_features["Dicrotic notch"] - ir_features["Dicrotic notch"]
    
    # 6. Ensemble ratio
    engineered_features["Ensemble ratio"] = ens_ratio
    
    return engineered_features

def average_features_across_windows(window_features_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Average features across all sliced windows for a subject."""
    if not window_features_list:
        return {}
        
    avg_features = {}
    keys = window_features_list[0].keys()
    
    for key in keys:
        vals = [win[key] for win in window_features_list if key in win]
        avg_features[key] = float(np.mean(vals)) if vals else 0.0
        
    return avg_features
