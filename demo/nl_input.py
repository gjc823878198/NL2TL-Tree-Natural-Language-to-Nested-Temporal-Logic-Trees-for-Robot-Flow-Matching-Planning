"""
Natural-language input box (tkinter) for the demo.

`ask_nl()` pops up a small window with a text field, a few one-click example
tasks, and a Run button, and returns the typed sentence (or "" if cancelled).
Both front-ends use it: the 2D demo (run_2d.py) and the ROS demo GUI
(run_ros.py, which embeds the richer panel below).

The examples all refer to the demo map's three regions A / B / C.
"""
from __future__ import annotations

EXAMPLES = [
    "Go to region A while always staying clear of every obstacle.",
    "Visit B, then A, then C in order, always avoiding obstacles, "
    "and finish within 180 seconds.",
    "Reach region C within 90 seconds, keeping safe the whole time.",
    "Visit A and C while never hitting an obstacle.",
]


def ask_nl(title: str = "NL2TL-Tree demo — enter a task") -> str:
    """Blocking pop-up; returns the entered sentence (empty string if closed)."""
    import tkinter as tk

    result = {"nl": ""}
    root = tk.Tk()
    root.title(title)
    root.geometry("680x300")

    tk.Label(root, text="Natural-language task (regions A / B / C on the map):",
             font=("DejaVu Sans", 12, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
    entry = tk.Text(root, height=3, width=72, font=("DejaVu Sans", 12), wrap="word")
    entry.pack(padx=12, pady=4)
    entry.insert("1.0", EXAMPLES[1])
    entry.focus_set()

    tk.Label(root, text="Examples (click to fill):",
             font=("DejaVu Sans", 10)).pack(anchor="w", padx=12, pady=(8, 0))
    frame = tk.Frame(root)
    frame.pack(fill="x", padx=12)

    def fill(text):
        entry.delete("1.0", "end")
        entry.insert("1.0", text)

    for i, ex in enumerate(EXAMPLES):
        tk.Button(frame, text=f"Example {i + 1}", command=lambda e=ex: fill(e)
                  ).grid(row=0, column=i, padx=3, pady=4, sticky="w")

    def submit():
        result["nl"] = entry.get("1.0", "end").strip()
        root.destroy()

    btns = tk.Frame(root)
    btns.pack(pady=12)
    tk.Button(btns, text="Run task", font=("DejaVu Sans", 12, "bold"),
              command=submit, bg="#2ecc71").grid(row=0, column=0, padx=6)
    tk.Button(btns, text="Cancel", command=root.destroy).grid(row=0, column=1, padx=6)
    root.bind("<Control-Return>", lambda _e: submit())

    root.mainloop()
    return result["nl"]


if __name__ == "__main__":
    print("entered:", ask_nl())
