# Stops the B2B cap sweep immediately: the sweep script and the bot run it started.
# Finished configurations are kept; the one in progress reruns when sweep.sh resumes.
$procs = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like '*b2b_sweep*sweep.sh*' -or
    ($_.Name -eq 'bot.exe' -and ($_.CommandLine -like '*--label allspin_cap_*' -or $_.CommandLine -like '*--label standard_cap_*'))
}
# Scripts first, so the loop cannot start the next configuration, then the bot.
$procs | Sort-Object { $_.Name -eq 'bot.exe' } | ForEach-Object {
    try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop; "stopped $($_.ProcessId) $($_.Name)" } catch {}
}
