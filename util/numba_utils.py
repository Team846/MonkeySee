import os
from util.config import ConfigCategory


_numba_category = ConfigCategory("Numba")
_cache_config = _numba_category.getIntConfig("cache", 1)


def _parse_env_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    return True


def numba_cache_enabled() -> bool:
    env = os.getenv("MONKEYSEE_NUMBA_CACHE")
    if env is not None:
        return _parse_env_bool(env)
    return _cache_config.valueInt() != 0


def njit_cached(*args, **kwargs):
    from numba import njit

    kwargs.setdefault("cache", numba_cache_enabled())
    return njit(*args, **kwargs)

