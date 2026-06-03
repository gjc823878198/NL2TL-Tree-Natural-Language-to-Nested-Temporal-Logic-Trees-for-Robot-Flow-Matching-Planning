"""
Mobile phone UI: talk to your space, get a verified spatiotemporal robot command.

A non-expert opens this on a phone (Streamlit is responsive; serve it on the LAN
with `streamlit run mobile_app.py --server.address 0.0.0.0` and browse to
http://<host-ip>:8501 from the phone -- the phone keyboard's mic gives voice
input for free).  They type a task in plain language; the app runs the full
pipeline (mobile_command.nl_to_spatiotemporal_command) and shows the verified
command, then dispatches it to the robot follower.

Layout is single-column ('centered') for phones.
"""
import json
from pathlib import Path

import streamlit as st

from mobile_command import (DEFAULT_SCENE, nl_to_spatiotemporal_command,
                            dispatch_to_robot)

st.set_page_config(page_title="Talk to the Robot", layout="centered")
st.title("🤖 Talk to the Robot")
st.caption("Say what you want in plain language. We turn it into a verified, "
           "spatiotemporal command — no model training, checked before it runs.")

scene = DEFAULT_SCENE
places = ", ".join(f"{k} ({g['kind']})" for k, g in scene["grounding"].items())
st.info(f"**Scene `{scene['name']}`** — known places: {places}.  "
        f"Robot starts at {tuple(scene['start'])}.")

examples = [
    "Eventually reach the kitchen within the time bound, and always avoid the bedroom.",
    "Visit the living room, then the kitchen, then go to the charger, always avoiding the bedroom.",
    "Reach the charger within 40 steps while never entering the bedroom.",
]
pick = st.selectbox("Example tasks", ["(type your own)"] + examples)
nl = st.text_area("Your task", "" if pick == "(type your own)" else pick,
                  height=90, placeholder="e.g. go to the kitchen but never enter the bedroom")

import os
if not os.getenv("GROQ_API_KEY"):
    st.warning("Set GROQ_API_KEY to enable NL parsing (free key: console.groq.com/keys).")

if st.button("▶︎ Plan command", disabled=not nl.strip()):
    with st.spinner("Parsing → verifying → planning…"):
        cmd = nl_to_spatiotemporal_command(nl, scene, backend="auto")
    st.session_state["cmd"] = cmd

cmd = st.session_state.get("cmd")
if cmd:
    # parse + self-correction
    badge = "✅ accepted" if cmd.get("parse_accepted") else "⚠️ not fully validated"
    st.subheader(f"1. Understood task  ({badge}, {cmd.get('parse_rounds',0)} "
                 f"self-correction round(s))")
    if cmd.get("error"):
        st.error(cmd["error"])
    else:
        # spatiotemporal command
        st.subheader("2. Spatiotemporal command")
        for i, r in enumerate(cmd.get("reach_sequence", []), 1):
            w = r["window"]
            when = f"within t∈{w}" if w else "eventually"
            st.write(f"**{i}. reach `{r['zone']}`** at ({r['x']},{r['y']}) — {when}")
        for z in cmd.get("avoid_zones", []):
            st.write(f"**avoid `{z['zone']}`** at ({z['x']},{z['y']}) — always")

        # feasibility (Tier-3) + robustness
        fz = cmd.get("feasibility")
        if fz:
            tag = "in-distribution → flow matching" if fz["feasible"] \
                else "out-of-distribution → A* backend"
            st.caption(f"feasibility self-check: prior satisfy-rate "
                       f"{fz['satisfy_rate']:.0%} → {tag}")
        rob = cmd.get("robustness")
        if rob is not None:
            (st.success if rob > 0 else st.error)(
                f"STL robustness of the planned path: ρ = {rob:+.3f} "
                f"({'satisfies' if rob > 0 else 'VIOLATES'} the command)")
        ks = cmd.get("keep_safe_rho")
        if ks is not None:
            st.caption(f"keep-safe predicate  ρ(G ¬unsafe) = {ks:+.3f} m "
                       f"(clearance to nearest obstacle; >0 = always safe)")

        # trajectory plot
        traj = cmd.get("trajectory", [])
        if traj:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(4, 4))
                xs, ys = zip(*traj)
                ax.plot(xs, ys, "-o", ms=2, color="tab:green", label="plan")
                ax.scatter([scene["start"][0]], [scene["start"][1]],
                           c="black", marker="s", s=60, label="start")
                for k, g in scene["grounding"].items():
                    c = "tab:blue" if g["kind"] == "reach" else "tab:red"
                    ax.add_patch(plt.Circle((g["x"], g["y"]), g["r"],
                                            color=c, alpha=0.3))
                    ax.text(g["x"], g["y"], k, fontsize=7, ha="center")
                b = scene["bounds"]
                ax.set_xlim(b[0], b[1]); ax.set_ylim(b[2], b[3])
                ax.set_aspect("equal"); ax.legend(fontsize=7)
                st.pyplot(fig)
            except Exception as e:
                st.caption(f"(plot skipped: {e})")

        # dispatch
        st.subheader("3. Send to robot")
        if st.button("📡 Dispatch command to robot"):
            path = dispatch_to_robot(cmd)
            st.success(f"Command written to {path}\n\n"
                       "Run it in Gazebo (two terminals) with:\n"
                       "`ros2 launch sim_ros2/launch/tb3_sim.launch.py "
                       "case:=mobile rviz:=true`\n"
                       "`python3 sim_ros2/tb3_follower.py --case mobile`")
        with st.expander("Raw command JSON"):
            st.json({k: v for k, v in cmd.items() if k != "parse_transcript"})
