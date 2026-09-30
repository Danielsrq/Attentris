# Stops the ablation immediately: the run.sh loop first, then its bot run.
# Finished configurations are kept; the one in progress reruns when run.sh resumes.
$procs = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like '*results/ablation/run.sh*' -or $_.CommandLine -like '*results\ablation\run.sh*' -or
    ($_.Name -eq 'bot.exe' -and $_.CommandLine -match '--label (zero|survival_only|hand_set|tuned)_d\d')
}
$procs | Sort-Object { $_.Name -eq 'bot.exe' } | ForEach-Object {
    try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop; "stopped $($_.ProcessId) $($_.Name)" } catch {}
}
