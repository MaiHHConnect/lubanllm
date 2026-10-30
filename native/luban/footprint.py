# SPDX-License-Identifier: Apache-2.0
"""Darwin physical memory measurements shared by the isolated HF probe."""
import ctypes


def footprint(pid):
    """Darwin SDK rusage_info_v4: 16-byte UUID followed by 35 uint64 fields."""
    names = (
        "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size "
        "resident_size phys_footprint proc_start_abstime proc_exit_abstime "
        "child_user_time child_system_time child_pkg_idle_wkups child_interrupt_wkups "
        "child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten "
        "cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background "
        "cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated "
        "cpu_time_qos_user_interactive billed_system_time serviced_system_time "
        "logical_writes lifetime_max_phys_footprint instructions cycles billed_energy "
        "serviced_energy interval_max_phys_footprint runnable_time"
    ).split()

    class Usage(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in names]

    assert ctypes.sizeof(Usage) == 296
    assert Usage.phys_footprint.offset == 72
    assert Usage.lifetime_max_phys_footprint.offset == 240
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.proc_pid_rusage.restype = ctypes.c_int
    data = Usage()
    if lib.proc_pid_rusage(pid, 4, ctypes.byref(data)) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    return {"physical_footprint_bytes": data.phys_footprint,
            "lifetime_peak_physical_footprint_bytes": data.lifetime_max_phys_footprint}

