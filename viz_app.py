"""
Streamlit visualization tool for STL parse trees.

Modes:
  - Single formula : paste any STL formula, see parse tree + JSON AST + round-trip
  - Browse dataset : open a JSONL produced by convert_dataset.py, flip through
                     samples, and mark incorrect ones (saved to *.feedback.jsonl)

Run:
  streamlit run viz_app.py
"""
import json
from pathlib import Path

import streamlit as st

from stl_parser import parse_stl, ast_to_dot, ast_to_stl

st.set_page_config(page_title="STL → AST Visualizer", layout="wide")
st.title("STL → AST 可视化检查工具")
st.caption("Verify that the parser correctly turns STL formulas into the tree representation we'll feed to the LLM.")

mode = st.radio(
    "Input source",
    ["Single formula", "Browse converted dataset", "Groq: NL → tree"],
    horizontal=True,
)

EXAMPLES = [
    "prop_1",
    "G[0,10] prop_1",
    "F[2,5] (prop_1 & prop_2)",
    "prop_1 U[0,5] prop_2",
    "!prop_1",
    "( ( ( prop_2 until [176,415] prop_1 ) and prop_3 ) equal prop_4 )",
    "globally [10,50] ( prop_1 imply finally [0,5] prop_2 )",
    "G[0,100] (prop_1 -> F[0,5] prop_2)",
]


def _render(ast, nl=None, source_formula=None):
    """Two-column rendering: tree on the left, JSON + round-trip on the right."""
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Parse tree")
        st.graphviz_chart(ast_to_dot(ast), use_container_width=True)
    with c2:
        if nl:
            st.subheader("Natural language")
            st.write(nl)
        if source_formula is not None:
            st.subheader("Source formula")
            st.code(source_formula, language="text")
        st.subheader("JSON AST")
        st.json(ast, expanded=True)
        st.subheader("Round-trip (AST → STL string)")
        try:
            st.code(ast_to_stl(ast), language="text")
        except Exception as e:
            st.error(f"Round-trip failed: {e}")


# ----------------------------- single-formula mode -----------------------------
if mode == "Single formula":
    pick = st.selectbox("Examples", ["(type your own)"] + EXAMPLES, index=0)
    default = "" if pick == "(type your own)" else pick
    formula = st.text_area("STL formula", default, height=80,
                           placeholder="e.g. G[0,10] (prop_1 & F[2,5] prop_2)")

    if formula.strip():
        try:
            ast = parse_stl(formula)
            _render(ast, source_formula=formula)
        except Exception as e:
            st.error(f"Parse error: {type(e).__name__}: {e}")
            with st.expander("Traceback"):
                st.exception(e)

# ----------------------------- dataset browsing mode -----------------------------
elif mode == "Browse converted dataset":
    default_path = str(Path(__file__).parent / "data" / "nl2tl_converted.jsonl")
    path_str = st.text_input("Path to converted JSONL", default_path)
    path = Path(path_str)

    if not path.exists():
        st.info("Run `python3 convert_dataset.py <input> data/nl2tl_converted.jsonl --limit 100` first.")
        st.stop()

    @st.cache_data(show_spinner=False)
    def load(p: str):
        return Path(p).read_text(encoding="utf-8").splitlines()

    lines = load(str(path))
    if not lines:
        st.warning("File is empty.")
        st.stop()

    st.caption(f"{len(lines)} samples in {path}")

    # filter switch for failed/ok
    show_only_ok = st.checkbox("Show only successfully parsed samples", value=True)

    idxs = []
    for i, ln in enumerate(lines):
        try:
            obj = json.loads(ln)
            if show_only_ok and obj.get("ok") is False:
                continue
            idxs.append(i)
        except json.JSONDecodeError:
            continue
    if not idxs:
        st.warning("No samples match the current filter.")
        st.stop()

    pos = st.slider("Sample position", 0, len(idxs) - 1, 0)
    abs_idx = idxs[pos]
    obj = json.loads(lines[abs_idx])
    st.caption(f"line {abs_idx + 1} / {len(lines)}")

    if obj.get("ok") is False:
        st.error(f"Parse FAILED for this row:  {obj.get('error', '')}")
        st.markdown(f"**Natural language:** {obj.get('natural', '(missing)')}")
        st.code(obj.get("formula", ""), language="text")
    else:
        ast = obj.get("tree")
        _render(ast,
                nl=obj.get("natural"),
                source_formula=obj.get("formula"))

    # ---- correction interface (human-in-the-loop) ----
    st.divider()
    st.subheader("Mark this sample as incorrect (for active-learning feedback)")
    note = st.text_input("Note: what's wrong / suggested fix", "")
    if st.button("Save feedback"):
        fb_path = path.with_suffix(".feedback.jsonl")
        with fb_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "line": abs_idx,
                "note": note,
                "sample": obj,
            }, ensure_ascii=False) + "\n")
        st.success(f"Saved feedback to {fb_path}")

# ----------------------------- Groq NL→tree mode -----------------------------
else:  # "Groq: NL → tree"
    import os
    st.markdown(
        "Type a natural-language task. Groq's free Llama-3.3-70b endpoint will "
        "be called with few-shot examples drawn from the converted NL2TL dataset. "
        "Free tier is generous: **30 RPM / 14,400 RPD**, no card required."
    )

    api_key_present = bool(os.getenv("GROQ_API_KEY"))
    if not api_key_present:
        st.warning("`GROQ_API_KEY` not set. Get a free key at "
                   "https://console.groq.com/keys, then `export GROQ_API_KEY=gsk_...`")

    fs_path = st.text_input("Few-shot source (converted JSONL)",
                            str(Path(__file__).parent / "data" / "nl2tl_converted.jsonl"),
                            key="groq_fs")
    n_shots = st.slider("Number of few-shot examples", 0, 16, 8, key="groq_n")
    model = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen2.5-32b"],
        index=0,
        key="groq_model",
        help="70b = best quality; 8b-instant = fastest; qwen-32b = good balance",
    )

    nl = st.text_area("Natural-language task", height=80, key="groq_nl",
                      placeholder="e.g. Always, whenever prop_1 happens, prop_2 must follow within 5 time units.")

    if st.button("Run Groq →  tree",
                 disabled=not (api_key_present and nl.strip())):
        try:
            from nl_to_tree_groq import load_few_shot, nl_to_tree  # noqa: F401
            import groq as _groq
            fs = load_few_shot(Path(fs_path), n=n_shots) if Path(fs_path).exists() else []
            with st.spinner(f"Calling {model} with {len(fs)} few-shot examples..."):
                pred = nl_to_tree(nl, model=model, few_shot=fs,
                                  client=_groq.Groq())
            _render(pred, nl=nl)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
            with st.expander("Traceback"):
                st.exception(e)
