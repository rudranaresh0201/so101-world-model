#!/usr/bin/env bash
# Memory guard: pause training jobs before the OOM killer can kill them.
#   scripts/mem_guard.sh unitA unitB unitC     (pause order = argument order; run under systemd-run)
# - MemAvailable < PAUSE_MB  -> SIGSTOP the next running unit in the list (one at a time, 30 s apart).
#                               A paused job loses nothing: it freezes and the kernel can swap it out.
# - MemAvailable > RESUME_MB -> SIGCONT the most recently paused unit (one at a time, 30 s apart).
# - The last running unit is never paused (something must keep making progress).
# - Exits once every listed unit has finished. Log: outputs/mem_guard.log
PAUSE_MB=${PAUSE_MB:-600}
RESUME_MB=${RESUME_MB:-1500}
UNITS=("$@")
LOG="$(dirname "$0")/../outputs/mem_guard.log"
PAUSED=()
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
swap_mb()  { awk '/SwapFree/{print int($2/1024)}' /proc/meminfo; }
active() { systemctl --user is-active --quiet "$1"; }
is_paused() { [[ " ${PAUSED[*]} " == *" $1 "* ]]; }

log "started: units ${UNITS[*]}  pause <${PAUSE_MB}MB  resume >${RESUME_MB}MB"
n=0; cooldown=0
while :; do
  running=(); for u in "${UNITS[@]}"; do active "$u" && ! is_paused "$u" && running+=("$u"); done
  alive=0;    for u in "${UNITS[@]}"; do active "$u" && alive=1; done
  [ "$alive" = 0 ] && { log "all units finished; guard stopping"; exit 0; }
  a=$(avail_mb); s=$(swap_mb)
  if [ "$cooldown" -le 0 ]; then
    if [ "$a" -lt "$PAUSE_MB" ] && [ "${#running[@]}" -gt 1 ]; then
      u=${running[0]}
      systemctl --user kill -s STOP "$u" && PAUSED+=("$u") && log "PAUSED $u (avail ${a}MB, swap free ${s}MB)"
      cooldown=3
    elif [ "${#PAUSED[@]}" -gt 0 ] && { [ "$a" -gt "$RESUME_MB" ] || [ "${#running[@]}" = 0 ]; }; then
      u=${PAUSED[-1]}; unset 'PAUSED[-1]'
      systemctl --user kill -s CONT "$u" && log "RESUMED $u (avail ${a}MB)"
      cooldown=3
    fi
  fi
  cooldown=$((cooldown - 1))
  n=$((n + 1)); [ $((n % 60)) = 0 ] && log "ok: avail ${a}MB swap free ${s}MB paused=[${PAUSED[*]}] running=[${running[*]}]"
  sleep 10
done
