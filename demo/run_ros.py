"""
ROS demo GUI node: the natural-language front-end for the Gazebo closed loop.

This is the process the demo launch file (launch/demo.launch.py) starts once the
Gazebo world + TurtleBot3 are up. It shows a text box; when you submit a task it

  1. parses the sentence with the paper's frozen-LLM pipeline
     (nl_grounding.nl_to_case -> nested STL tree -> grounded planner case),
  2. writes the case to /tmp/nl_demo_case.json,
  3. opens RViz (markers config) so you can watch, and
  4. starts the TurtleBot3 follower (code/sim_ros2/tb3_follower.py --case-file ...)
     which plans with TeLoGraF and drives the robot through the task.

So: type a task -> STL tree -> plan -> the robot executes, RViz opens with it.
Run standalone (after `demo.launch.py` is up) with:

    python3 run_ros.py
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nl_grounding as G                                            # noqa: E402

CODE = G.CODE
FOLLOWER = CODE / "sim_ros2" / "tb3_follower.py"
RVIZ_CFG = CODE / "sim_ros2" / "gui" / "markers.rviz"
CASE_JSON = Path(tempfile.gettempdir()) / "nl_demo_case.json"

from nl_input import EXAMPLES                                       # noqa: E402


class DemoGUI:
    def __init__(self, model: str, use_llm: bool, launch_rviz: bool = True):
        import tkinter as tk
        self.tk = tk
        self.model = model
        self.use_llm = use_llm
        self.launch_rviz = launch_rviz        # False when the launch file opened RViz
        self.procs = []                       # spawned follower / rviz
        self.root = tk.Tk()
        self.root.title("NL2TL-Tree demo — natural-language task")
        self.root.geometry("760x540")
        tk.Label(self.root,
                 text="Type a task for the TurtleBot3 (regions A / B / C):",
                 font=("DejaVu Sans", 13, "bold")).pack(anchor="w", padx=12,
                                                        pady=(12, 4))
        self.entry = tk.Text(self.root, height=3, width=80,
                             font=("DejaVu Sans", 12), wrap="word")
        self.entry.pack(padx=12, pady=4)
        self.entry.insert("1.0", EXAMPLES[1])

        exf = tk.Frame(self.root)
        exf.pack(fill="x", padx=12)
        for i, ex in enumerate(EXAMPLES):
            tk.Button(exf, text=f"Example {i + 1}",
                      command=lambda e=ex: self._fill(e)
                      ).grid(row=0, column=i, padx=3, pady=4)

        # environment-complexity hint for the pre-flight completability gauge
        self.env_var = tk.StringVar(value="normal")
        envf = tk.Frame(self.root)
        envf.pack(fill="x", padx=12, pady=(6, 0))
        tk.Label(envf, text="Environment (pre-flight feasibility):").grid(
            row=0, column=0, sticky="w")
        for j, lvl in enumerate(("open", "normal", "complex")):
            tk.Radiobutton(envf, text=lvl, variable=self.env_var, value=lvl).grid(
                row=0, column=j + 1, padx=4)

        bf = tk.Frame(self.root)
        bf.pack(pady=8)
        tk.Button(bf, text="Run task", font=("DejaVu Sans", 13, "bold"),
                  bg="#2ecc71", command=self._run).grid(row=0, column=0, padx=6)
        tk.Button(bf, text="Stop robot", command=self._stop).grid(row=0, column=1,
                                                                  padx=6)
        tk.Button(bf, text="Quit", command=self._quit).grid(row=0, column=2, padx=6)

        tk.Label(self.root, text="Parsed STL tree / status:",
                 font=("DejaVu Sans", 10, "bold")).pack(anchor="w", padx=12,
                                                        pady=(8, 0))
        self.out = tk.Text(self.root, height=10, width=88,
                           font=("DejaVu Sans Mono", 10), wrap="word",
                           bg="#0f1419", fg="#d8e0e8")
        self.out.pack(padx=12, pady=4, fill="both", expand=True)
        self._log("Gazebo world is up. Enter a task and press Run.\n"
                  f"LLM parse: {'ON (' + model + ')' if use_llm else 'OFF (offline keyword parser)'}\n")

    def _fill(self, text):
        self.entry.delete("1.0", "end")
        self.entry.insert("1.0", text)

    def _log(self, msg):
        self.out.insert("end", msg + "\n")
        self.out.see("end")
        self.root.update_idletasks()

    def _run(self):
        nl = self.entry.get("1.0", "end").strip()
        if not nl:
            self._log("(empty task)")
            return
        self._log(f"\n>>> NL: {nl}")
        case, tree, info, used_llm = G.nl_to_case(nl, model=self.model,
                                                 use_llm=self.use_llm)
        self._log(f"parse : {'frozen LLM' if used_llm else 'keyword fallback (offline)'}")
        self._log(f"STL   : {G.stl_string(tree)}")
        self._log(f"plan  : visit {' -> '.join(info['goal_names'])}  "
                  f"| keep_safe={info['keep_safe']} "
                  f"| deadline={info['deadline_s']:.0f}s")
        # pre-flight completability gauge (BEFORE launching the robot)
        try:
            import feasibility as Feas
            from sim_ros2.scenario import CYLINDERS
            obs = [{"kind": "circle", "x": x, "y": y, "r": r}
                   for (x, y, r) in CYLINDERS]
            fe = Feas.estimate(G.START, info["goals"], info["deadline_s"],
                               self.env_var.get(), obs)
            self._log("check : " + Feas.summary(fe))
        except Exception as e:
            self._log(f"check : (gauge unavailable: {e})")
        CASE_JSON.write_text(json.dumps(case, indent=1))
        self._log(f"wrote {CASE_JSON}")
        self._stop()                          # stop only the previous follower
        env = dict(os.environ, QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM",
                                                              "xcb"))
        # ONE persistent RViz (the SAME markers.rviz as the closed-loop sim):
        # it shows /ubicomp/markers -- start, all goal regions, the planned path,
        # the TRAVELLED history trajectory, the sense ring and the sim clock --
        # and stays up across tasks (re-opened only if the attendee closed it), so
        # the demo's RViz matches the closed-loop simulation exactly.
        if self.launch_rviz and (getattr(self, "rviz", None) is None
                                 or self.rviz.poll() is not None):
            self.rviz = subprocess.Popen(["rviz2", "-d", str(RVIZ_CFG)], env=env)
            self._log("opened RViz (persistent) — shows the live + history "
                      "trajectory, same as the closed-loop sim")
        # start the follower: plans with TeLoGraF, drives the TurtleBot3
        self.procs.append(subprocess.Popen(
            [sys.executable, str(FOLLOWER), "--case-file", str(CASE_JSON),
             "--multi-goal", "--scene-external"], cwd=str(CODE), env=env))
        self._log("started TurtleBot3 follower -> the robot will plan and move "
                  "(first TeLoGraF call can take ~10-30 s).")
        # the best-of-N OOD self-check batch-samples the flow prior (~15-30 s); run
        # it in the BACKGROUND so it NEVER blocks the GUI or delays the robot.
        import threading
        threading.Thread(target=self._run_selfcheck, args=(info,),
                         daemon=True).start()

    def _run_selfcheck(self, info):
        """Best-of-N OOD self-check, off the GUI thread (logs its verdict + saves
        the figure when done; failures are non-fatal)."""
        try:
            import matplotlib; matplotlib.use("Agg")
            import selfcheck as SC
            from sim_ros2.scenario import CYLINDERS as _CYL
            obs = [{"kind": "circle", "x": x, "y": y, "r": r} for (x, y, r) in _CYL]
            sc = SC.run(info, obstacles=obs, world_bounds=G.WORLD_BOUNDS,
                        start=G.START)
            msg = "self  : " + SC.summary(sc)
            if sc.get("available"):
                import run_2d
                p = run_2d.render_selfcheck(info, sc)
                if p:
                    msg += f"\nself  : self-check figure -> {p}"
        except Exception as e:
            msg = f"self  : (self-check unavailable: {e})"
        try:                                  # log back on the GUI (main) thread
            self.root.after(0, lambda: self._log(msg))
        except Exception:
            print(msg)

    def _stop(self):
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
        self.procs = []

    def _quit(self):
        self._stop()
        if getattr(self, "rviz", None) is not None and self.rviz.poll() is None:
            self.rviz.terminate()                # close the persistent RViz too
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true",
                    help="use the offline keyword parser instead of Groq")
    ap.add_argument("--model", default=G.DEFAULT_MODEL)
    ap.add_argument("--no-rviz", action="store_true",
                    help="don't open RViz from the GUI (the launch file already did)")
    args = ap.parse_args()
    DemoGUI(model=args.model, use_llm=not args.no_llm,
            launch_rviz=not args.no_rviz).run()


if __name__ == "__main__":
    main()
