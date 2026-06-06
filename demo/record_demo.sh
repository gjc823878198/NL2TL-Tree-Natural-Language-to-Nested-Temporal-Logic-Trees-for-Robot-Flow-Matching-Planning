#!/usr/bin/env bash
# =============================================================================
#  record_demo.sh -- one-click, ~zero-CPU screen recorder for the demo (MP4).
#
#  Records the WHOLE screen to an H.264 .mp4 using OBS with:
#     capture : Screen Capture (PipeWire)  -- the ONLY thing that works on this
#               GNOME-Wayland box (x11grab = black; ffmpeg kmsgrab is blocked
#               because the iGPU VAAPI driver is too old AND the Intel->NVIDIA
#               frame bridge is unimplemented in this ffmpeg build).
#     encode  : NVIDIA NVENC (RTX 5080) -- a dedicated encode chip, NOT the CPU
#               and NOT CUDA. So recording costs ~0% CPU: your CPU-bound TeLoGraF
#               planner and Gazebo are untouched.
#
#  FIRST TIME ONLY (≈30 s, needed once because the Wayland screen-share grant
#  must be interactive):
#       ./record_demo.sh setup
#     -> OBS opens on the "NL2TLDemo" profile. In OBS:
#        1. Sources panel -> "+" -> "Screen Capture (PipeWire)" -> OK
#        2. In the dialog, pick your monitor -> "Share"  (OBS remembers it)
#        3. (verify) Settings -> Output -> Recording -> Video Encoder shows
#           "NVIDIA NVENC H.264"  and Recording Format = mp4   (already preset)
#        4. Close OBS.
#
#  THEN, every time:
#       ./record_demo.sh            # one click: OBS starts recording, minimized
#       ...run your demo...
#       ./record_demo.sh stop       # finalize the .mp4 (or click Stop in OBS tray)
#
#  Output: ~/Videos/  (newest file printed on stop).
# =============================================================================
set -euo pipefail

PROFILE="NL2TLDemo"
OBS_CFG="$HOME/.config/obs-studio"
PROFILE_INI="$OBS_CFG/basic/profiles/$PROFILE/basic.ini"
RECDIR="$HOME/Videos"
mode="${1:-start}"

have_obs() { command -v obs >/dev/null || { echo "OBS not installed (apt install obs-studio)"; exit 1; }; }
mkdir -p "$RECDIR"

# Recreate the NVENC+mp4 profile if it is missing, so the script is self-contained
# (resolution auto-detected from the primary panel; OBS scales other monitors).
ensure_profile() {
  [ -f "$PROFILE_INI" ] && return 0
  local res w h
  res=$(cat /sys/class/drm/card*-eDP-1/modes 2>/dev/null | head -1)
  [ -z "$res" ] && res=$(cat /sys/class/drm/card*/modes 2>/dev/null | head -1)
  [ -z "$res" ] && res="1920x1080"
  w=${res%x*}; h=${res#*x}
  mkdir -p "$(dirname "$PROFILE_INI")"
  cat > "$PROFILE_INI" <<EOF
[General]
Name=$PROFILE

[Output]
Mode=Advanced

[AdvOut]
RecType=Standard
RecEncoder=ffmpeg_nvenc
RecFilePath=$RECDIR
RecFormat=mp4
RecRescale=false
RecTracks=1
FFOutputToFile=true

[Video]
BaseCX=$w
BaseCY=$h
OutputCX=$w
OutputCY=$h
FPSType=0
FPSCommon=30
ScaleType=bicubic
ColorFormat=NV12

[Audio]
SampleRate=48000
ChannelSetup=Stereo
EOF
  cat > "$(dirname "$PROFILE_INI")/recordEncoder.json" <<'EOF'
{
    "rate_control": "CQP",
    "cqp": 23,
    "preset": "hq",
    "profile": "high",
    "bf": 2,
    "psycho_aq": true,
    "lookahead": false,
    "keyint_sec": 0
}
EOF
  echo ">> created OBS profile '$PROFILE' (${w}x${h}, NVENC, mp4)."
}

# Guard: the profile MUST encode with NVENC, never x264 (which would load the CPU).
assert_nvenc() {
  if ! grep -q '^RecEncoder=ffmpeg_nvenc' "$PROFILE_INI" 2>/dev/null; then
    echo "!! profile '$PROFILE' is not set to NVENC -- refusing to record on the CPU."
    echo "   expected RecEncoder=ffmpeg_nvenc in $PROFILE_INI"
    exit 1
  fi
}

# Heuristic: has a PipeWire screen-capture source been added to any collection?
pipewire_source_ready() {
  grep -rqiE "pipewire" "$OBS_CFG/basic/scenes/" 2>/dev/null
}

case "$mode" in
  setup)
    have_obs; ensure_profile; assert_nvenc
    echo ">> Opening OBS on profile '$PROFILE'. Do the 4 one-time steps in the header, then close OBS."
    obs --profile "$PROFILE" >/dev/null 2>&1 &
    echo ">> (OBS launched, PID $!).  After you add the PipeWire source + share your monitor, you're done."
    ;;

  start|"")
    have_obs; ensure_profile; assert_nvenc
    if pgrep -x obs >/dev/null; then
      echo "!! OBS is already running -- stop it first ('./record_demo.sh stop') to avoid double sessions."; exit 1
    fi
    if ! pipewire_source_ready; then
      echo "!! No PipeWire screen-capture source found yet. Run './record_demo.sh setup' once first."; exit 1
    fi
    echo "============================================================"
    echo " Recording WHOLE screen -> $RECDIR/  (NVENC / RTX 5080, mp4)"
    echo "   CPU cost ~0%  --  TeLoGraF/Gazebo planner untouched."
    echo " Stop with:  ./record_demo.sh stop   (or click Stop in the OBS tray)"
    echo "============================================================"
    obs --profile "$PROFILE" --startrecording --minimize-to-tray >/dev/null 2>&1 &
    echo ">> OBS recording (PID $!)."
    ;;

  stop)
    if ! pgrep -x obs >/dev/null; then echo "(OBS not running)"; exit 0; fi
    echo ">> stopping + finalizing the mp4..."
    pkill -INT -x obs || true            # OBS finalizes the recording on graceful exit
    for _ in $(seq 1 20); do pgrep -x obs >/dev/null || break; sleep 0.3; done
    newest=$(ls -t "$RECDIR"/*.mp4 2>/dev/null | head -1 || true)
    [ -n "$newest" ] && echo ">> saved: $(du -h "$newest" | cut -f1)  $newest" || echo ">> (no mp4 found in $RECDIR)"
    ;;

  status)
    if pgrep -x obs >/dev/null; then echo "OBS running (PID $(pgrep -x obs|tr '\n' ' '))"; else echo "OBS not running"; fi
    grep -q '^RecEncoder=ffmpeg_nvenc' "$PROFILE_INI" 2>/dev/null && echo "encoder: NVENC (GPU) ✔" || echo "encoder: NOT nvenc ✘"
    pipewire_source_ready && echo "pipewire source: configured ✔" || echo "pipewire source: NOT set up -> run './record_demo.sh setup'"
    ;;

  -h|--help) sed -n '2,40p' "$0" ;;
  *) echo "usage: $0 [setup|start|stop|status]"; exit 2 ;;
esac
