//! Keep the bot on the performance cores of a hybrid CPU (e.g. Intel 12th gen: P-cores
//! with hyper-threading plus E-cores).
//!
//! Windows' scheduler moves work it considers background -- a process started from a
//! background shell, or one under power throttling (EcoQoS) -- onto the efficiency
//! cores. Two fixes, both applied to this process at startup:
//!   1. opt out of power throttling, so Windows does not treat it as background work;
//!   2. restrict the process to the logical processors with the highest efficiency
//!      class reported by Windows (the P-cores), read from the OS rather than assumed.
//! `--any-cores` skips both.

/// (logical processor index, efficiency class) for every logical processor.
#[cfg(windows)]
pub fn cpu_sets() -> Vec<(u32, u8)> {
    use windows_sys::Win32::System::SystemInformation::{GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut out = Vec::new();
    unsafe {
        let mut len: u32 = 0;
        GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut len, GetCurrentProcess(), 0);
        if len == 0 {
            return out;
        }
        let mut buf = vec![0u8; len as usize];
        if GetSystemCpuSetInformation(buf.as_mut_ptr() as *mut SYSTEM_CPU_SET_INFORMATION, len, &mut len, GetCurrentProcess(), 0) == 0 {
            return out;
        }
        // Entries are variable-sized; walk them by their own Size field.
        let mut off = 0usize;
        while off + std::mem::size_of::<u32>() <= len as usize {
            let info = &*(buf.as_ptr().add(off) as *const SYSTEM_CPU_SET_INFORMATION);
            if info.Size == 0 {
                break;
            }
            let cs = info.Anonymous.CpuSet;
            out.push((cs.LogicalProcessorIndex as u32 + 64 * cs.Group as u32, cs.EfficiencyClass));
            off += info.Size as usize;
        }
    }
    out
}

/// Linux: a hybrid Intel CPU lists its P-cores in /sys/devices/cpu_core/cpus and its E-cores in
/// /sys/devices/cpu_atom/cpus (class 1 / 0). Anything else (AMD, older Intel, a VM) has neither:
/// every online CPU gets class 0, so `performance_mask` sees no hybrid CPU and restricts nothing.
#[cfg(target_os = "linux")]
pub fn cpu_sets() -> Vec<(u32, u8)> {
    let list = |path: &str| parse_cpu_list(&std::fs::read_to_string(path).unwrap_or_default());
    let (p, e) = (list("/sys/devices/cpu_core/cpus"), list("/sys/devices/cpu_atom/cpus"));
    let mut out: Vec<(u32, u8)> = if !p.is_empty() && !e.is_empty() {
        p.iter().map(|&i| (i, 1)).chain(e.iter().map(|&i| (i, 0))).collect()
    } else {
        list("/sys/devices/system/cpu/online").into_iter().map(|i| (i, 0)).collect()
    };
    out.sort();
    out
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn cpu_sets() -> Vec<(u32, u8)> {
    Vec::new()
}

/// A Linux sysfs CPU list ("0-15,20,22-23"); empty on a missing file or an unexpected format.
#[allow(dead_code)] // used on Linux only; tested everywhere
fn parse_cpu_list(s: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for part in s.trim().split(',').filter(|p| !p.is_empty()) {
        let mut ends = part.split('-').map(|v| v.trim().parse::<u32>());
        match (ends.next(), ends.next(), ends.next()) {
            (Some(Ok(a)), Some(Ok(b)), None) if a <= b => out.extend(a..=b),
            (Some(Ok(a)), None, None) => out.push(a),
            _ => return Vec::new(),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::parse_cpu_list;

    #[test]
    fn sysfs_cpu_lists() {
        assert_eq!(parse_cpu_list("0-3\n"), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_list("0-1,4,6-7"), vec![0, 1, 4, 6, 7]);
        assert_eq!(parse_cpu_list(""), Vec::<u32>::new()); // no such file: not hybrid
        assert_eq!(parse_cpu_list("garbage"), Vec::<u32>::new());
        assert_eq!(parse_cpu_list("5-2"), Vec::<u32>::new());
    }
}

/// Logical processors in the highest efficiency class (the P-cores), as an affinity mask.
pub fn performance_mask() -> Option<usize> {
    let sets = cpu_sets();
    let top = sets.iter().map(|&(_, c)| c).max()?;
    if sets.iter().all(|&(_, c)| c == top) {
        return None; // not a hybrid CPU: nothing to restrict
    }
    let mut mask = 0usize;
    for &(i, c) in &sets {
        if c == top && i < usize::BITS {
            mask |= 1 << i;
        }
    }
    Some(mask)
}

/// Opt out of power throttling and pin to the P-cores. Returns a one-line description.
#[cfg(windows)]
pub fn prefer_performance_cores() -> String {
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, ProcessPowerThrottling, SetProcessAffinityMask, SetProcessInformation,
        PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_STATE,
    };
    let mut notes = Vec::new();
    unsafe {
        // ControlMask = EXECUTION_SPEED with StateMask = 0: "never throttle me".
        let state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            StateMask: 0,
        };
        let ok = SetProcessInformation(
            GetCurrentProcess(),
            ProcessPowerThrottling,
            &state as *const _ as *const core::ffi::c_void,
            std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
        );
        notes.push(if ok != 0 { "power throttling off" } else { "power throttling unchanged" }.to_string());
        if let Some(mask) = performance_mask() {
            let ok = SetProcessAffinityMask(GetCurrentProcess(), mask);
            notes.push(if ok != 0 {
                format!("pinned to {} performance logical processors (mask {:#x})", mask.count_ones(), mask)
            } else {
                "affinity unchanged".to_string()
            });
        }
    }
    notes.join(", ")
}

/// Linux: pin to the P-cores of a hybrid CPU (sched_setaffinity; threads started later inherit
/// it). No power-throttling equivalent is needed. Not hybrid: nothing to do.
#[cfg(target_os = "linux")]
pub fn prefer_performance_cores() -> String {
    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
    }
    let Some(mask) = performance_mask() else { return String::new() };
    let mut set = [0u64; 16]; // cpu_set_t: 1024 bits
    set[0] = mask as u64;
    let ok = unsafe { sched_setaffinity(0, std::mem::size_of_val(&set), set.as_ptr()) } == 0;
    if ok {
        format!("pinned to {} performance logical processors (mask {:#x})", mask.count_ones(), mask)
    } else {
        "affinity unchanged".to_string()
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn prefer_performance_cores() -> String {
    String::new()
}
