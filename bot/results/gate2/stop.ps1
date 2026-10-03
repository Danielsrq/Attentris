# Stops results/gate2/run.sh immediately: the script first, then its bot process.
# Every step resumes from its state / cache file when run.sh is started again.
$procs = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like '*results/gate2/run.sh*' -or $_.CommandLine -like '*results\gate2\run.sh*' -or
    ($_.Name -eq 'bot.exe' -and $_.CommandLine -like '*results/gate2*')
}
$procs | Sort-Object { $_.Name -eq 'bot.exe' } | ForEach-Object {
    try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop; "stopped $($_.ProcessId) $($_.Name)" } catch {}
}
