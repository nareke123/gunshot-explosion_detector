import numpy as np


def estimate_noise_floor_dbfs(
    acoustic_summaries: list[dict[str, float]],
    percentile: float = 20.0,
) -> float:
    """Estimate ambient loudness from a batch of windows."""
    if not acoustic_summaries:
        return -90.0
    rms_values = [summary["rms_dbfs"] for summary in acoustic_summaries]
    return float(np.percentile(rms_values, percentile))


def passes_event_gate(
    acoustic_summary: dict[str, float],
    config: dict,
    noise_floor_dbfs: float,
) -> tuple[bool, dict[str, float | bool | str]]:
    """Decide whether a window is salient enough to classify as an event."""
    min_rms_dbfs = float(config.get("event_gate_min_rms_dbfs", -35.0))
    min_peak_dbfs = float(config.get("event_gate_min_peak_dbfs", -18.0))
    min_rms_above_noise_floor_db = float(
        config.get("event_gate_min_rms_above_noise_floor_db", 10.0)
    )
    min_crest_factor = float(config.get("event_gate_min_crest_factor", 2.5))

    adaptive_rms_threshold_dbfs = noise_floor_dbfs + min_rms_above_noise_floor_db
    loud_enough = acoustic_summary["rms_dbfs"] >= max(min_rms_dbfs, adaptive_rms_threshold_dbfs)
    transient_enough = (
        acoustic_summary["peak_dbfs"] >= min_peak_dbfs
        and acoustic_summary["crest_factor"] >= min_crest_factor
    )

    passed = bool(loud_enough or transient_enough)
    reason = "loud" if loud_enough else "transient" if transient_enough else "background"
    details = {
        "passed": passed,
        "reason": reason,
        "noise_floor_dbfs": float(noise_floor_dbfs),
        "adaptive_rms_threshold_dbfs": float(adaptive_rms_threshold_dbfs),
        "loud_enough": bool(loud_enough),
        "transient_enough": bool(transient_enough),
    }
    return passed, details
