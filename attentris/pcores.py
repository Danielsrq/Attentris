"""Pin this Python process to the performance cores of a hybrid CPU, as the Rust bot does
(bot/src/cpu.rs): affinity mask = the logical processors `bot cpus` reports as
'performance', and Windows power throttling (EcoQoS) off. No-op elsewhere or on failure.

    import pcores; n = pcores.pin()   # returns the number of P logical processors (or None)
"""
import ctypes
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.join(HERE, "..", "bot", "target", "release", "bot.exe")


def performance_mask():
    out = subprocess.run([BOT, "cpus"], capture_output=True, text=True, check=True).stdout
    mask = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit() and parts[2] == "performance":
            mask |= 1 << int(parts[0])
    return mask


class _Throttle(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


def pin(verbose=True):
    if sys.platform != "win32":
        return None
    try:
        mask = performance_mask()
        if not mask:
            return None
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        h = ctypes.c_void_p(k32.GetCurrentProcess())
        if not k32.SetProcessAffinityMask(h, ctypes.c_size_t(mask)):
            raise OSError(ctypes.get_last_error())
        # ProcessPowerThrottling = 4; control EXECUTION_SPEED (1), state 0 = never throttle
        st = _Throttle(1, 1, 0)
        k32.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st))
        n = bin(mask).count("1")
        if verbose:
            print("pinned to %d performance logical processors (mask %#x), power throttling off" % (n, mask), flush=True)
        return n
    except Exception as e:  # never block a run on this
        if verbose:
            print("P-core pinning failed (%s); running unpinned" % e, flush=True)
        return None
