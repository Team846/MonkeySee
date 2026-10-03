import glob
import time
from threading import Lock
from typing import List, Optional, Tuple

MIN_SAMPLE_S = 1.0

_lock = Lock()
_prev_times: Optional[List[Tuple[int, int]]] = None
_sampled_at = 0.0
_stats: Optional[dict] = None


def _cpu_thermal_zones() -> List[str]:
    zones = []
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            with open(f"{zone}/type") as f:
                kind = f.read().strip()
        except OSError:
            continue
        if any(k in kind for k in ("soc", "core", "cpu")):
            zones.append(zone)
    return zones or sorted(glob.glob("/sys/class/thermal/thermal_zone*"))


_CPU_ZONES = _cpu_thermal_zones()


def _read_cpu_times() -> List[Tuple[int, int]]:
    times = []
    with open("/proc/stat") as f:
        for line in f:
            if not (line.startswith("cpu") and line[3].isdigit()):
                continue
            fields = [int(v) for v in line.split()[1:9]]
            total = sum(fields)
            times.append((total - fields[3] - fields[4], total))
    return times


def _read_temp_c() -> Optional[float]:
    temps = []
    for zone in _CPU_ZONES:
        try:
            with open(f"{zone}/temp") as f:
                temps.append(int(f.read()) / 1000.0)
        except (OSError, ValueError):
            pass
    return max(temps) if temps else None


def get_system_stats() -> Optional[dict]:
    global _prev_times, _sampled_at, _stats
    with _lock:
        now = time.monotonic()
        if _stats is not None and now - _sampled_at < MIN_SAMPLE_S:
            return _stats
        try:
            if _prev_times is None:
                _prev_times = _read_cpu_times()
                time.sleep(0.1)
            times = _read_cpu_times()
        except OSError:
            return None

        cores = [
            100.0 * (busy - prev_busy) / (total - prev_total) if total > prev_total else 0.0
            for (busy, total), (prev_busy, prev_total) in zip(times, _prev_times)
        ]
        _prev_times = times
        _sampled_at = now
        _stats = {
            "cpu": sum(cores) / len(cores) if cores else 0.0,
            "cores": cores,
            "temp_c": _read_temp_c(),
        }
        return _stats
