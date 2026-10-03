# Stops the normalised-vs-raw comparison immediately: the run.sh loop first, then the tune.
# Each tune resumes from its last finished generation when run.sh is started again.
$procs = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like '*cem_norm_vs_raw*run.sh*' -or
    ($_.Name -eq 'bot.exe' -and $_.CommandLine -like '* tune *' -and $_.CommandLine -like '*cem_norm_vs_raw*')
}
$procs | Sort-Object { $_.Name -eq 'bot.exe' } | ForEach-Object {
    try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop; "stopped $($_.ProcessId) $($_.Name)" } catch {}
}
