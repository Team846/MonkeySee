import socket
import struct
import threading
from collections import deque
from time import monotonic_ns, sleep
from typing import Optional

_MAGIC = 0x8461
_PING = struct.Struct(">HQ")
_PONG = struct.Struct(">HQQ")


class TimeSync:
    def __init__(
        self,
        host: str,
        port: int = 5809,
        period_s: float = 0.1,
        window_s: float = 5.0,
        max_age_s: float = 2.0,
        max_rtt_ms: float = 50.0,
        jump_reset_ms: float = 100.0,
    ):
        self._addr = (host, port)
        self._period_ns = int(period_s * 1e9)
        self._window_ns = int(window_s * 1e9)
        self._max_age_ns = int(max_age_s * 1e9)
        self._max_rtt_ns = int(max_rtt_ms * 1e6)
        self._jump_reset_us = jump_reset_ms * 1e3
        self._samples = deque()
        self._best = None
        self._lock = threading.Lock()
        threading.Thread(target=self._run, name="TimeSync", daemon=True).start()

    def _run(self):
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    while True:
                        start = monotonic_ns()
                        self._ping(sock)
                        elapsed = monotonic_ns() - start
                        sleep(max(0.0, (self._period_ns - elapsed) / 1e9))
            except OSError:
                sleep(1.0)

    def _ping(self, sock):
        t1 = monotonic_ns()
        sock.sendto(_PING.pack(_MAGIC, t1), self._addr)
        deadline = t1 + 50_000_000
        while True:
            remaining_s = (deadline - monotonic_ns()) / 1e9
            if remaining_s <= 0:
                return
            sock.settimeout(remaining_s)
            try:
                data, _ = sock.recvfrom(64)
            except socket.timeout:
                return
            t4 = monotonic_ns()
            if len(data) < _PONG.size:
                continue
            magic, fpga_us, echo = _PONG.unpack_from(data)
            if magic != _MAGIC or echo != t1:
                continue
            self._add(t4, t4 - t1, fpga_us - (t1 + t4) / 2000.0)
            return

    def _add(self, t4_ns: int, rtt_ns: int, offset_us: float):
        with self._lock:
            if self._best is not None and abs(offset_us - self._best[2]) > self._jump_reset_us:
                self._samples.clear()
            self._samples.append((t4_ns, rtt_ns, offset_us))
            while self._samples and t4_ns - self._samples[0][0] > self._window_ns:
                self._samples.popleft()
            self._best = min(self._samples, key=lambda s: s[1])

    def _current(self):
        with self._lock:
            best = self._best
            newest = self._samples[-1][0] if self._samples else None
        if best is None or newest is None:
            return None
        if monotonic_ns() - newest > self._max_age_ns or best[1] > self._max_rtt_ns:
            return None
        return best

    def to_robot_time_s(self, pi_ns: int) -> Optional[float]:
        best = self._current()
        if best is None:
            return None
        return (pi_ns / 1000.0 + best[2]) / 1e6

    def rtt_ms(self) -> float:
        best = self._current()
        return -1.0 if best is None else best[1] / 1e6
