"""
Real-Time Priority Inversion Analyzer
=====================================
Streamlit dashboard that simulates a single-CPU real-time scheduler with and
without Priority Inheritance (PI), measures the difference, and runs a small
ML classifier as an early-warning layer.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import copy
import re
from dataclasses import replace

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from metrics import calculate_all_metrics
from risk_model import load_model, predict_risk
from scenarios import ResourceConfig, TaskConfig, create_priority_inversion_scenario
from scheduler_engine import (
    simulate_with_priority_inheritance,
    simulate_without_priority_inheritance,
)

# =========================================================
# PAGE CONFIG (must be the first Streamlit call)
# =========================================================

st.set_page_config(
    page_title="Priority Inversion Analyzer",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CONSTANTS
# =========================================================

TASK_ORDER = ["H", "M", "L"]
PRIORITY_CLASS = {"H": "High", "M": "Medium", "L": "Low"}
POLICIES = {"baseline": "Without PI", "pi": "With PI"}
RESULT_FIELDS = ("history", "state", "metrics", "log")

STATE_COLORS = {"RUNNING": "#65D6A2", "BLOCKED": "#FF6B6B", "READY": "#3B4B46"}
STATE_LABELS = {
    "RUNNING": "Running",
    "BLOCKED": "Blocked on R1",
    "READY": "Ready (waiting for CPU)",
}
STATE_ALIASES = {"WAITING": "READY", "PREEMPTED": "READY"}

POLICY_COLORS = {"Without PI": "#FF6B6B", "With PI": "#65D6A2"}
RISK_ORDER = ["LOW", "MEDIUM", "HIGH"]
RISK_CLASS = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}

DEFAULT_STATE = {
    f"{prefix}_{field}": None for prefix in POLICIES for field in RESULT_FIELDS
}
DEFAULT_STATE["sweep_df"] = None


# =========================================================
# SMALL HELPERS
# =========================================================

def _version_tuple(text: str) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", text)[:2])


_NEW_WIDTH_API = _version_tuple(st.__version__) >= (1, 50)


def stretch(fn, *args, **kwargs):
    """Call a Streamlit element full-width on both old and new Streamlit.

    Newer releases replaced ``use_container_width=True`` with ``width="stretch"``
    (and are removing the old argument), so try the new form first.
    """
    if _NEW_WIDTH_API:
        try:
            return fn(*args, width="stretch", **kwargs)
        except Exception:
            pass
    return fn(*args, use_container_width=True, **kwargs)


def html(markup: str) -> None:
    """Render HTML safely through st.markdown.

    Markdown ends an HTML block at the first blank line and treats lines
    indented by 4+ spaces as code, so collapse the markup onto one line.
    """
    flat = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def fmt(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else f"{number:.1f}"


def as_percent(value) -> float:
    """CPU utilisation may be stored as a 0-1 fraction or as a percentage."""
    number = float(value)
    return number * 100 if number <= 1.0 else number


def compare_tone(before, after) -> str:
    try:
        before, after = float(before), float(after)
    except (TypeError, ValueError):
        return "same"
    if after < before:
        return "good"
    if after > before:
        return "bad"
    return "same"


def style_fig(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=20, r=20, t=60 if title else 30, b=20),
        title=title,
        font=dict(color="#D6E2DE"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def active_map() -> dict:
    return {task.name: task for task in st.session_state["active_tasks"]}


# =========================================================
# SESSION STATE
# =========================================================

for _key, _default in DEFAULT_STATE.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

if "active_tasks" not in st.session_state:
    _tasks, _resources = create_priority_inversion_scenario()
    st.session_state["active_tasks"] = _tasks
    st.session_state["active_resources"] = _resources

if "scenario_mode" not in st.session_state:
    st.session_state["scenario_mode"] = "Classic Demo"


def clear_results() -> None:
    for key, default in DEFAULT_STATE.items():
        st.session_state[key] = default


def set_active_tasks(tasks, resources) -> None:
    st.session_state["active_tasks"] = tasks
    st.session_state["active_resources"] = resources
    clear_results()


def load_classic_scenario() -> None:
    tasks, resources = create_priority_inversion_scenario()
    set_active_tasks(tasks, resources)


def on_mode_change() -> None:
    """Switching back to Classic must actually restore the classic workload.

    (Previously the radio said "Classic" while the custom tasks stayed active.)
    """
    if st.session_state["scenario_mode"] == "Classic Demo":
        load_classic_scenario()


def reset_to_classic_scenario() -> None:
    load_classic_scenario()
    st.session_state["scenario_mode"] = "Classic Demo"


def store_result(prefix: str, history, state, log) -> None:
    st.session_state[f"{prefix}_history"] = history
    st.session_state[f"{prefix}_state"] = state
    st.session_state[f"{prefix}_metrics"] = calculate_all_metrics(history, state)
    st.session_state[f"{prefix}_log"] = log


def run_experiment(which: str) -> None:
    tasks = st.session_state["active_tasks"]
    resources = st.session_state["active_resources"]
    try:
        with st.spinner("Running simulation…"):
            if which in ("both", "baseline"):
                store_result("baseline", *simulate_without_priority_inheritance(tasks, resources))
            if which in ("both", "pi"):
                store_result("pi", *simulate_with_priority_inheritance(tasks, resources))
        st.toast("Simulation finished", icon="✅")
    except Exception as error:  # surface engine errors instead of a blank page
        st.error(f"Simulation failed: {type(error).__name__}: {error}")


def both_results_ready() -> bool:
    return all(
        st.session_state[f"{prefix}_{field}"] is not None
        for prefix in POLICIES
        for field in ("history", "state", "metrics")
    )


# =========================================================
# SCENARIO VALIDATION
# =========================================================

def validate_custom(values: dict) -> list[str]:
    errors = []
    priorities = [values[name]["priority"] for name in TASK_ORDER]
    if len(set(priorities)) < 3:
        errors.append("Use distinct priorities for H, M, and L so the scheduler has a clear priority order.")
    return errors


def scenario_notes(active: dict) -> list[tuple[str, str]]:
    """Heads-up messages for custom workloads that won't show an inversion."""
    h, m, l = active["H"], active["M"], active["L"]
    notes = []
    if not (h.priority > m.priority > l.priority):
        notes.append(("warning", "Priorities are not ordered H > M > L (this app treats a higher number as higher priority), so the classic inversion pattern may not appear."))
    if h.arrival_time <= l.arrival_time:
        notes.append(("warning", "H arrives no later than L, so L never holds R1 when H needs it and there is nothing to invert. Make L arrive first."))
    elif h.arrival_time >= l.arrival_time + l.execution_time:
        notes.append(("warning", "H arrives after L would already have finished, so H never waits for R1. Move H's arrival earlier or lengthen L."))
    if h.deadline < h.arrival_time + h.execution_time:
        notes.append(("info", "H's deadline is earlier than arrival + execution, so H cannot meet it under any policy. That is a deliberate stress case."))
    return notes


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    :root {
        --bg: #050908;
        --panel: rgba(255,255,255,0.055);
        --border: rgba(255,255,255,0.11);
        --text: #F2F7F5;
        --muted: #9FB5AD;
        --green: #65D6A2;
        --amber: #FFCC66;
        --red: #FF6B6B;
        --blue: #73B7FF;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(34, 96, 78, 0.24), transparent 30%),
            radial-gradient(circle at 100% 0%, rgba(24, 59, 53, 0.55), transparent 34%),
            linear-gradient(180deg, #07110F 0%, var(--bg) 52%, #030505 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.8rem;
        padding-left: 2.4rem;
        padding-right: 2.4rem;
        padding-bottom: 4rem;
    }

    footer { visibility: hidden; }

    [data-testid="stSidebar"] {
        background: rgba(5, 10, 9, 0.92);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 16px;
    }

    button[data-baseweb="tab"] { font-weight: 700; }

    .hero { padding: 8px 0; }

    .eyebrow {
        color: var(--green);
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .main-title {
        font-size: 44px;
        font-weight: 850;
        letter-spacing: -1.4px;
        line-height: 1.06;
        margin-bottom: 8px;
    }

    .subtitle {
        color: var(--muted);
        font-size: 17px;
        line-height: 1.55;
        max-width: 920px;
        margin-bottom: 22px;
    }

    .glass-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.075), rgba(255,255,255,0.035));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 16px 50px rgba(0,0,0,0.24);
        backdrop-filter: blur(14px);
        height: 100%;
    }

    .card-title { font-size: 17px; font-weight: 800; margin-bottom: 10px; }

    .section-label { color: var(--green); font-size: 13px; font-weight: 700; margin-bottom: 4px; }
    .section-title { font-size: 25px; font-weight: 800; margin-bottom: 14px; }

    .status-title { color: #C6D5D0; font-size: 13px; }
    .status-value { font-size: 22px; font-weight: 800; margin-top: 5px; }
    .status-dot { color: var(--green); margin-right: 6px; }
    .status-dot.wait { color: var(--amber); }

    .kpi-label { color: #92A79F; font-size: 13px; font-weight: 700; }
    .kpi-value { font-size: 30px; line-height: 1.2; font-weight: 850; margin-top: 6px; }
    .kpi-before { color: #B8C9C3; }
    .kpi-arrow { color: var(--muted); margin: 0 10px; font-weight: 600; }
    .kpi-after.good { color: var(--green); }
    .kpi-after.bad { color: var(--red); }
    .kpi-after.same { color: var(--text); }
    .kpi-note { color: #8EA39B; font-size: 12px; margin-top: 5px; }

    .danger-box, .success-box, .info-box {
        border-radius: 18px;
        padding: 20px;
        height: 100%;
    }
    .danger-box { background: rgba(255,80,80,0.085); border: 1px solid rgba(255,80,80,0.22); }
    .success-box { background: rgba(60,200,140,0.085); border: 1px solid rgba(60,200,140,0.22); }
    .info-box { background: rgba(115,183,255,0.075); border: 1px solid rgba(115,183,255,0.20); }

    .verdict { border-radius: 18px; padding: 18px 22px; margin: 6px 0 18px 0; }
    .verdict.success { background: rgba(60,200,140,0.10); border: 1px solid rgba(60,200,140,0.35); }
    .verdict.warning { background: rgba(255,204,102,0.09); border: 1px solid rgba(255,204,102,0.35); }
    .verdict.info { background: rgba(115,183,255,0.08); border: 1px solid rgba(115,183,255,0.30); }
    .verdict-title { font-size: 19px; font-weight: 800; margin-bottom: 4px; }
    .verdict-text { color: #C9D8D3; font-size: 15px; line-height: 1.55; }

    .flow-row { display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px; margin-bottom: 14px; }
    .flow-node {
        flex: 1 1 110px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 12px 10px;
        text-align: center;
        font-weight: 700;
        font-size: 14px;
    }
    .flow-node.bad { background: rgba(255,107,107,0.10); border-color: rgba(255,107,107,0.40); }
    .flow-node.good { background: rgba(101,214,162,0.10); border-color: rgba(101,214,162,0.40); }
    .flow-node.warn { background: rgba(255,204,102,0.10); border-color: rgba(255,204,102,0.40); }
    .flow-arrow { align-self: center; color: var(--green); font-weight: 900; }

    .risk-high { color: var(--red); font-size: 34px; font-weight: 900; }
    .risk-medium { color: var(--amber); font-size: 34px; font-weight: 900; }
    .risk-low { color: var(--green); font-size: 34px; font-weight: 900; }

    .small-muted { color: var(--muted); font-size: 13px; line-height: 1.6; }

    .pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        color: #D6E2DE;
        font-size: 13px;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .pill.ok { border-color: rgba(101,214,162,0.5); color: var(--green); }
    .pill.wait { border-color: rgba(255,204,102,0.4); color: var(--amber); }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# UI COMPONENTS
# =========================================================

def kpi_card(label: str, before: str, after: str, tone: str, note: str) -> None:
    html(
        f"""
        <div class="glass-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value"><span class="kpi-before">{before}</span><span class="kpi-arrow">→</span><span class="kpi-after {tone}">{after}</span></div>
        <div class="kpi-note">{note}</div>
        </div>
        """
    )


def flow_row(nodes: list[tuple[str, str]]) -> None:
    parts = []
    for index, (text, tone) in enumerate(nodes):
        if index:
            parts.append('<div class="flow-arrow">→</div>')
        parts.append(f'<div class="flow-node {tone}">{text}</div>')
    html(f'<div class="flow-row">{"".join(parts)}</div>')


def verdict_box(tone: str, title: str, text: str) -> None:
    html(
        f"""
        <div class="verdict {tone}">
        <div class="verdict-title">{title}</div>
        <div class="verdict-text">{text}</div>
        </div>
        """
    )


# =========================================================
# ANALYTICS HELPERS
# =========================================================

def count_inversion_units(history) -> int:
    """Time units where H is BLOCKED while M (not using R1) is RUNNING."""
    grid: dict = {}
    for event in history:
        grid.setdefault(event.get("time"), {})[event.get("task")] = event.get("state")
    return sum(
        1 for states in grid.values()
        if states.get("H") == "BLOCKED" and states.get("M") == "RUNNING"
    )


def count_inheritance_units(history, base_priorities: dict) -> int:
    """Time units where a task runs above its base priority (inheritance active)."""
    total = 0
    for event in history:
        if event.get("state") != "RUNNING":
            continue
        effective = event.get("effective_priority")
        base = base_priorities.get(event.get("task"))
        if effective is not None and base is not None and effective > base:
            total += 1
    return total


def build_segments(history, base_priorities: dict) -> pd.DataFrame:
    columns = ["task", "start", "end", "state", "eff_priority", "owner", "marker"]
    rows = []
    for event in history:
        state = str(event.get("state", "")).upper()
        state = STATE_ALIASES.get(state, state)
        time = event.get("time")
        if state not in STATE_COLORS or time is None:
            continue
        effective = event.get("effective_priority")
        base = base_priorities.get(event.get("task"))
        inherited = (
            state == "RUNNING" and effective is not None and base is not None and effective > base
        )
        rows.append(
            {
                "task": event.get("task"),
                "start": time,
                "end": time + 1,
                "state": state,
                "eff_priority": effective,
                "owner": event.get("resource_owner") or "None",
                "marker": "↑" if inherited else "",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_timeline(histories: list, base_priorities: dict, h_deadline) -> go.Figure:
    """Gantt chart with plain numeric time.

    px.timeline forces a *date* axis, so integer time units would be drawn as
    1970 timestamps. Horizontal go.Bar with a numeric ``base`` avoids that.
    """
    rows = len(histories)
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.14,
        subplot_titles=[label for label, _ in histories],
    )
    shown: set = set()
    max_time = 1

    for row, (label, history) in enumerate(histories, start=1):
        segments = build_segments(history, base_priorities)
        if not segments.empty:
            max_time = max(max_time, int(segments["end"].max()))
        for state, color in STATE_COLORS.items():
            part = segments[segments["state"] == state]
            if part.empty:
                continue
            fig.add_trace(
                go.Bar(
                    y=part["task"],
                    x=[1] * len(part),
                    base=part["start"],
                    orientation="h",
                    name=STATE_LABELS[state],
                    legendgroup=state,
                    showlegend=state not in shown,
                    marker=dict(color=color, line=dict(color="#07110F", width=1.5)),
                    text=part["marker"],
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(color="#04110C", size=15),
                    customdata=part[["state", "eff_priority", "owner", "start"]].to_numpy(),
                    hovertemplate=(
                        "<b>%{y}</b> · %{customdata[0]}<br>"
                        "t = %{customdata[3]}<br>"
                        "Effective priority: %{customdata[1]}<br>"
                        "R1 owner: %{customdata[2]}<extra></extra>"
                    ),
                ),
                row=row, col=1,
            )
            shown.add(state)

        fig.update_yaxes(
            categoryorder="array", categoryarray=["L", "M", "H"], row=row, col=1
        )
        line_kwargs = dict(x=h_deadline, line_dash="dot", line_color="#FFCC66", row=row, col=1)
        if row == 1:
            line_kwargs.update(annotation_text="H deadline", annotation_position="top left")
        fig.add_vline(**line_kwargs)

    fig.update_xaxes(
        range=[0, max(max_time, h_deadline) + 0.5],
        dtick=1 if max(max_time, h_deadline) <= 40 else None,
        gridcolor="rgba(255,255,255,0.06)",
    )
    fig.update_xaxes(title_text="Simulation time (units)", row=rows, col=1)
    style_fig(fig, height=250 * rows + 140)
    fig.update_layout(barmode="overlay", bargap=0.25)
    return fig


def assess_risk(state: dict, metrics: dict) -> dict:
    h_config = state["H"]["config"]
    l_config = state["L"]["config"]
    waiting = metrics["H"]["blocking_time"]
    # If H never blocked there is no lock contention to warn about.
    contended = 1 if waiting > 0 else 0

    result = predict_risk(
        task_priority=h_config.priority,
        owner_priority=l_config.priority,
        waiting_time=waiting,
        arrival_time=h_config.arrival_time,
        resource_locked=contended,
        higher_priority_waiting=contended,
    )
    probabilities = result["probabilities"]
    if not isinstance(probabilities, dict) or not probabilities:
        raise ValueError("Risk model returned no probability distribution.")
    probabilities = {str(k).upper(): float(v) for k, v in probabilities.items()}

    return {
        "prediction": str(result["prediction"]).upper(),
        "probabilities": probabilities,
        "confidence": max(probabilities.values()) * 100,
        "features": {
            "Task priority": h_config.priority,
            "Resource owner priority": l_config.priority,
            "Priority difference": h_config.priority - l_config.priority,
            "Waiting / blocking time": waiting,
            "Arrival time": h_config.arrival_time,
            "Resource locked": "YES" if contended else "NO",
            "Higher-priority task waiting": "YES" if contended else "NO",
        },
    }


def with_execution(task, execution_time: int):
    try:
        return replace(task, execution_time=execution_time)
    except TypeError:
        clone = copy.copy(task)
        clone.execution_time = execution_time
        return clone


def run_sweep(tasks, resources, low: int, high: int) -> pd.DataFrame:
    """Re-run both policies while M's execution time grows."""
    rows = []
    progress = st.progress(0.0)
    steps = high - low + 1
    for index, m_exec in enumerate(range(low, high + 1)):
        variant = [with_execution(t, m_exec) if t.name == "M" else t for t in tasks]
        for label, simulate in (
            ("Without PI", simulate_without_priority_inheritance),
            ("With PI", simulate_with_priority_inheritance),
        ):
            history, state, _log = simulate(variant, resources)
            h = calculate_all_metrics(history, state)["H"]
            rows.append(
                {
                    "M execution": m_exec,
                    "Policy": label,
                    "H blocking": h["blocking_time"],
                    "H response": h["response_time"],
                    "H completion": h["completion_time"],
                    "Deadline missed": bool(h["deadline_miss"]),
                }
            )
        progress.progress((index + 1) / steps)
    progress.empty()
    return pd.DataFrame(rows)


def build_verdict(h_base: dict, h_pi: dict, deadline) -> tuple[str, str, str]:
    base_miss, pi_miss = bool(h_base["deadline_miss"]), bool(h_pi["deadline_miss"])
    saved = h_base["completion_time"] - h_pi["completion_time"]

    if base_miss and not pi_miss:
        return (
            "success",
            "Priority Inheritance saved H's deadline",
            f"Without PI, H finished at t={fmt(h_base['completion_time'])} and missed its deadline of t={fmt(deadline)}. "
            f"With PI, H finished at t={fmt(h_pi['completion_time'])}, {fmt(saved)} time units earlier.",
        )
    if base_miss and pi_miss:
        return (
            "warning",
            "H misses its deadline even with Priority Inheritance",
            f"PI shortened H's completion from t={fmt(h_base['completion_time'])} to t={fmt(h_pi['completion_time'])}, "
            f"but the deadline (t={fmt(deadline)}) is tighter than L's critical section plus H's own execution. "
            "PI bounds the inversion; it cannot create CPU time.",
        )
    if not base_miss and pi_miss:
        return (
            "warning",
            "Unexpected result: H is later with PI",
            "H met its deadline without PI but missed it with PI. Check the execution logs for the scheduler's decisions.",
        )
    if h_base["blocking_time"] == 0:
        return (
            "info",
            "No priority inversion occurred in this workload",
            "H was never blocked on R1, so both policies behave the same. Make L arrive before H and keep H's arrival inside L's critical section.",
        )
    return (
        "info",
        "Both policies meet H's deadline, but PI still cuts the blocking",
        f"H waited {fmt(h_base['blocking_time'])} units without PI and {fmt(h_pi['blocking_time'])} with PI. "
        "Tighten H's deadline to see the case where the difference decides the outcome.",
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## Control center")
    st.caption("Configure and run the real-time scheduling experiment")

    st.markdown("### Scenario")
    scenario_choice = st.radio(
        "Scenario mode",
        ["Classic Demo", "Custom Demo"],
        key="scenario_mode",
        label_visibility="collapsed",
        on_change=on_mode_change,
    )

    if scenario_choice == "Classic Demo":
        st.markdown(
            "<div class='small-muted'>Classic priority-inversion workload is active.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("H and L share R1; M does not use R1. A higher number means a higher priority.")
        current = active_map()

        def number(label: str, low: int, high: int, value: int) -> int:
            return int(st.number_input(label, min_value=low, max_value=high, step=1,
                                       value=int(min(max(value, low), high))))

        with st.form("custom_scenario_form"):
            form_tabs = st.tabs(["H · High", "M · Medium", "L · Low"])
            values: dict = {}
            for tab, name in zip(form_tabs, TASK_ORDER):
                with tab:
                    task = current[name]
                    values[name] = {
                        "priority": number(f"{name} priority", 1, 10, task.priority),
                        "arrival": number(f"{name} arrival", 0, 50, task.arrival_time),
                        "execution": number(f"{name} execution", 1, 50, task.execution_time),
                        "deadline": number(f"{name} deadline", 1, 100, task.deadline),
                    }
            applied = stretch(st.form_submit_button, "Apply custom scenario", type="primary")

        if applied:
            problems = validate_custom(values)
            for problem in problems:
                st.error(problem)
            if not problems:
                def build(name: str, uses_r1: bool) -> TaskConfig:
                    v = values[name]
                    return TaskConfig(
                        name, v["priority"], v["arrival"], v["execution"], v["deadline"],
                        uses_r1, "R1" if uses_r1 else None,
                    )

                set_active_tasks([build("L", True), build("M", False), build("H", True)],
                                 [ResourceConfig("R1")])
                st.toast("Custom scenario applied; previous results cleared", icon="✅")
                st.rerun()

        stretch(st.button, "Restore classic scenario", on_click=reset_to_classic_scenario)

    st.markdown("---")

    if stretch(st.button, "Run complete experiment", type="primary"):
        run_experiment("both")

    run_col_a, run_col_b = st.columns(2)
    with run_col_a:
        if stretch(st.button, "Baseline only"):
            run_experiment("baseline")
    with run_col_b:
        if stretch(st.button, "PI only"):
            run_experiment("pi")

    if stretch(st.button, "Reset results"):
        clear_results()
        st.rerun()

    st.markdown("---")
    st.markdown("### Active workload")
    sidebar_tasks = active_map()
    for task_name in TASK_ORDER:
        t = sidebar_tasks[task_name]
        st.markdown(
            f"<div class='small-muted'><b>{t.name}</b> · P{t.priority} · arrival {t.arrival_time} · "
            f"exec {t.execution_time} · deadline {t.deadline}</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Metric definitions"):
        st.caption("Blocking: discrete time units in BLOCKED state")
        st.caption("Response: completion time − arrival time")
        st.caption("Deadline miss: completion time > deadline")
        st.caption("CPU utilization: busy time / total simulation time")
        st.caption("Fairness: Jain's Fairness Index")
        st.caption("Inversion window: units where H is blocked while M runs")


# =========================================================
# HEADER
# =========================================================

html(
    """
    <div class="hero">
    <div class="eyebrow">Operating systems · real-time scheduling</div>
    <div class="main-title">Real-Time Priority Inversion Analyzer</div>
    <div class="subtitle">
    A simulation-driven dashboard that demonstrates priority inversion,
    applies Priority Inheritance, measures scheduling performance,
    and uses a lightweight AI classifier for early risk detection.
    </div>
    </div>
    """
)


# =========================================================
# STATUS CARDS
# =========================================================

@st.cache_resource(show_spinner=False)
def check_model() -> bool:
    load_model()
    return True


try:
    check_model()
    ai_status, ai_note, ai_dot = "Loaded", "Decision Tree risk classifier", ""
except Exception:
    ai_status, ai_note, ai_dot = "Needs training", "Model will train automatically", " wait"

ready_count = sum(st.session_state[f"{p}_metrics"] is not None for p in POLICIES)
exp_status = {0: "Not run", 1: "Partial", 2: "Complete"}[ready_count]
exp_dot = "" if ready_count == 2 else " wait"

status_cards = [
    ("Scheduler simulator", "Ready", "", "Deterministic single-CPU experiment"),
    ("Priority Inheritance", "Available", "", "Dynamic effective-priority boosting"),
    ("AI risk model", ai_status, ai_dot, ai_note),
    ("Experiment", exp_status, exp_dot, "Baseline and PI runs for the active workload"),
]
for column, (title, value, dot, note) in zip(st.columns(4), status_cards):
    with column:
        html(
            f"""
            <div class="glass-card">
            <div class="status-title">{title}</div>
            <div class="status-value"><span class="status-dot{dot}">●</span>{value}</div>
            <div class="small-muted">{note}</div>
            </div>
            """
        )


# =========================================================
# SCENARIO OVERVIEW
# =========================================================

st.divider()
st.markdown('<div class="section-label">Scenario</div>', unsafe_allow_html=True)
scenario_title = (
    "Classic priority-inversion setup"
    if st.session_state["scenario_mode"] == "Classic Demo"
    else "Custom priority-inversion workload"
)
st.markdown(f'<div class="section-title">{scenario_title}</div>', unsafe_allow_html=True)

active = active_map()
base_priorities = {name: task.priority for name, task in active.items()}
h_deadline = active["H"].deadline

c1, c2 = st.columns([1.35, 1])

with c1:
    scenario_df = pd.DataFrame(
        [
            [
                name,
                PRIORITY_CLASS[name],
                active[name].priority,
                active[name].arrival_time,
                active[name].execution_time,
                active[name].deadline,
                active[name].resource_name or "CPU",
            ]
            for name in TASK_ORDER
        ],
        columns=["Task", "Priority class", "Priority", "Arrival", "Execution", "Deadline", "Resource"],
    )
    stretch(st.dataframe, scenario_df, hide_index=True)

with c2:
    html(
        """
        <div class="glass-card">
        <div class="card-title">Resource contention</div>
        <b>R1</b> is the shared resource.<br><br>
        L is configured to own R1 and H is configured to request R1.
        M does not use R1, so it can only delay L through CPU scheduling.<br><br>
        <b>Proposed mechanism:</b> L temporarily inherits H's
        effective priority while H is blocked on R1.
        </div>
        """
    )

if st.session_state["scenario_mode"] == "Custom Demo":
    for level, message in scenario_notes(active):
        (st.warning if level == "warning" else st.info)(message)


# =========================================================
# EXPERIMENT STATUS (waiting for results)
# =========================================================

if not both_results_ready():
    st.divider()
    st.markdown('<div class="section-label">Ready to run</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Launch the complete experiment</div>', unsafe_allow_html=True)

    pills = "".join(
        f'<span class="pill {"ok" if st.session_state[f"{p}_metrics"] is not None else "wait"}">'
        f'{label}: {"done" if st.session_state[f"{p}_metrics"] is not None else "not run yet"}</span>'
        for p, label in POLICIES.items()
    )
    html(
        f"""
        <div class="info-box">
        Choose <b>Run complete experiment</b> in the sidebar. The app runs both scheduling
        policies on the active workload, calculates every metric, draws the timeline, and
        sends the measured baseline conditions to the AI risk model.<br><br>
        {pills}
        </div>
        """
    )

    partial = [
        (label, st.session_state[f"{p}_history"])
        for p, label in POLICIES.items()
        if st.session_state[f"{p}_history"] is not None
    ]
    if partial:
        st.markdown("### Timeline so far")
        stretch(st.plotly_chart, build_timeline(partial, base_priorities, h_deadline), key="partial_timeline")

    st.stop()


# =========================================================
# RESULTS
# =========================================================

baseline = st.session_state["baseline_metrics"]
pi = st.session_state["pi_metrics"]
h_base, h_pi = baseline["H"], pi["H"]

inversion_base = count_inversion_units(st.session_state["baseline_history"])
inversion_pi = count_inversion_units(st.session_state["pi_history"])
inheritance_units = count_inheritance_units(st.session_state["pi_history"], base_priorities)

st.divider()
st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Before vs after Priority Inheritance</div>', unsafe_allow_html=True)

verdict_box(*build_verdict(h_base, h_pi, h_deadline))

k1, k2, k3, k4 = st.columns(4)

with k1:
    kpi_card("H blocking time", fmt(h_base["blocking_time"]), fmt(h_pi["blocking_time"]),
             compare_tone(h_base["blocking_time"], h_pi["blocking_time"]), "Lower is better")
with k2:
    kpi_card("H response time", fmt(h_base["response_time"]), fmt(h_pi["response_time"]),
             compare_tone(h_base["response_time"], h_pi["response_time"]), "Completion − arrival")
with k3:
    kpi_card("H completion time", fmt(h_base["completion_time"]), fmt(h_pi["completion_time"]),
             compare_tone(h_base["completion_time"], h_pi["completion_time"]), "Simulation time units")
with k4:
    base_outcome = "Missed" if h_base["deadline_miss"] else "Met"
    pi_outcome = "Missed" if h_pi["deadline_miss"] else "Met"
    deadline_tone = compare_tone(bool(h_base["deadline_miss"]), bool(h_pi["deadline_miss"]))
    kpi_card("H deadline", base_outcome, pi_outcome, deadline_tone, f"H deadline = t={h_deadline}")

st.write("")

tab_overview, tab_compare, tab_timeline, tab_sweep, tab_ai, tab_logs = st.tabs(
    ["Overview", "Comparison", "Scheduler timeline", "Sensitivity", "AI risk engine", "Execution logs"]
)


# =========================================================
# OVERVIEW TAB
# =========================================================

with tab_overview:
    st.markdown("### Quantified impact")

    block_reduction = h_base["blocking_time"] - h_pi["blocking_time"]
    response_reduction = h_base["response_time"] - h_pi["response_time"]
    block_reduction_pct = (
        block_reduction / h_base["blocking_time"] * 100 if h_base["blocking_time"] else 0.0
    )
    misses_base = baseline["system"]["deadline_misses"]
    misses_pi = pi["system"]["deadline_misses"]

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric(
            "H blocking reduction",
            f"{fmt(block_reduction)} units",
            f"{block_reduction_pct:+.1f}% vs baseline" if h_base["blocking_time"] else None,
        )
    with q2:
        st.metric("H response reduction", f"{fmt(response_reduction)} units")
    with q3:
        miss_delta = misses_pi - misses_base
        st.metric(
            "System deadline misses",
            f"{misses_base} → {misses_pi}",
            f"{miss_delta:+d}" if miss_delta else "no change",
            delta_color="inverse" if miss_delta else "off",
        )
    with q4:
        window_delta = inversion_pi - inversion_base
        st.metric(
            "Inversion window",
            f"{inversion_base} → {inversion_pi} units",
            f"{window_delta:+d}" if window_delta else "no change",
            delta_color="inverse" if window_delta else "off",
            help="Time units where H is blocked on R1 while M, which does not use R1, is running.",
        )

    st.markdown("### Without Priority Inheritance")
    base_end = ("H meets deadline", "good") if not h_base["deadline_miss"] else ("H misses deadline", "bad")
    if h_base["blocking_time"] == 0:
        flow_row([("L owns R1", ""), ("H arrives", ""), ("H never blocked", "good"), base_end])
    else:
        flow_row([
            ("L owns R1", ""),
            ("H requests R1", ""),
            (f"H blocked ({fmt(h_base['blocking_time'])} units)", "bad"),
            ("M runs ahead of L", "bad") if inversion_base else ("L keeps the CPU", ""),
            (f"Inversion window: {inversion_base} units", "bad") if inversion_base else ("L releases R1", ""),
            base_end,
        ])

    st.markdown("### With Priority Inheritance")
    pi_end = (f"H completes at t={fmt(h_pi['completion_time'])}", "good" if not h_pi["deadline_miss"] else "bad")
    if h_pi["blocking_time"] == 0:
        flow_row([("L owns R1", ""), ("H arrives", ""), ("H never blocked", "good"), pi_end])
    else:
        flow_row([
            ("L owns R1", ""),
            ("H requests R1", ""),
            (f"H blocked ({fmt(h_pi['blocking_time'])} units)", "warn"),
            ("L inherits H's priority", "good"),
            ("L runs ahead of M", "good"),
            ("L releases R1", ""),
            pi_end,
        ])


# =========================================================
# COMPARISON TAB
# =========================================================

with tab_compare:
    st.markdown("### Task-level evaluation")

    rows = []
    for name in ["L", "M", "H"]:
        rows.append(
            {
                "Task": name,
                "CPU time — Without PI": baseline[name]["cpu_time"],
                "CPU time — With PI": pi[name]["cpu_time"],
                "Blocking — Without PI": baseline[name]["blocking_time"],
                "Blocking — With PI": pi[name]["blocking_time"],
                "Response — Without PI": baseline[name]["response_time"],
                "Response — With PI": pi[name]["response_time"],
                "Completion — Without PI": baseline[name]["completion_time"],
                "Completion — With PI": pi[name]["completion_time"],
                "Deadline miss — Without PI": "YES" if baseline[name]["deadline_miss"] else "NO",
                "Deadline miss — With PI": "YES" if pi[name]["deadline_miss"] else "NO",
            }
        )
    comparison_df = pd.DataFrame(rows)
    stretch(st.dataframe, comparison_df, hide_index=True)
    st.download_button(
        "Download comparison (CSV)",
        data=comparison_df.to_csv(index=False).encode("utf-8"),
        file_name="priority_inversion_comparison.csv",
        mime="text/csv",
    )

    def grouped_bar(metric_key: str, title: str) -> go.Figure:
        frame = pd.DataFrame(
            [
                {"Task": name, "Policy": label, "Value": metrics[name][metric_key]}
                for label, metrics in (("Without PI", baseline), ("With PI", pi))
                for name in TASK_ORDER
            ]
        )
        fig = px.bar(
            frame, x="Task", y="Value", color="Policy", barmode="group", text="Value",
            category_orders={"Task": TASK_ORDER}, color_discrete_map=POLICY_COLORS,
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_title="Task", yaxis_title="Time units", legend_title_text="")
        return style_fig(fig, 400, title)

    chart_a, chart_b = st.columns(2)
    with chart_a:
        stretch(st.plotly_chart, grouped_bar("blocking_time", "Blocking time by policy"), key="chart_blocking")
    with chart_b:
        stretch(st.plotly_chart, grouped_bar("response_time", "Response time by policy"), key="chart_response")

    st.markdown("### System-level metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("CPU utilization — baseline", f'{as_percent(baseline["system"]["cpu_utilization"]):.1f}%')
    with m2:
        st.metric("CPU utilization — PI", f'{as_percent(pi["system"]["cpu_utilization"]):.1f}%')
    with m3:
        st.metric("Fairness — baseline", f'{baseline["system"]["fairness"]:.3f}')
    with m4:
        st.metric("Fairness — PI", f'{pi["system"]["fairness"]:.3f}')

    st.caption(
        "In this fixed workload, PI changes scheduling order and blocking behaviour; "
        "the total CPU service requirement remains unchanged."
    )


# =========================================================
# TIMELINE TAB
# =========================================================

with tab_timeline:
    st.markdown("### Scheduler timeline")
    st.caption(
        "Each block is one simulation time unit. ↑ marks a unit where a task runs above its base "
        "priority through inheritance. The dotted line is H's deadline."
    )

    timeline_fig = build_timeline(
        [
            ("Without PI", st.session_state["baseline_history"]),
            ("With PI", st.session_state["pi_history"]),
        ],
        base_priorities,
        h_deadline,
    )
    stretch(st.plotly_chart, timeline_fig, key="main_timeline")

    st.markdown("### Why the timeline changes")
    e1, e2 = st.columns(2)

    with e1:
        if inversion_base:
            body = (
                f"H is blocked by L. Because L keeps its low base priority, M runs for {inversion_base} "
                "time units while H waits. That indirectly extends how long H waits for R1."
            )
        elif h_base["blocking_time"]:
            body = "H is blocked by L, but M never runs during the wait, so there is no extra delay from M."
        else:
            body = "H is never blocked on R1 in this workload, so there is no inversion to show."
        html(f'<div class="danger-box"><h4>Without Priority Inheritance</h4><p>{body}</p></div>')

    with e2:
        if inheritance_units:
            body = (
                f"Once H blocks, L inherits H's effective priority and runs {inheritance_units} "
                "time units above its base priority. It finishes the critical section before M, "
                "releases R1 sooner, and lets H continue."
            )
        else:
            body = "Priority inheritance was never triggered because H did not block on R1."
        html(f'<div class="success-box"><h4>With Priority Inheritance</h4><p>{body}</p></div>')


# =========================================================
# SENSITIVITY TAB
# =========================================================

with tab_sweep:
    st.markdown("### Sensitivity analysis: how bad can the inversion get?")
    st.caption(
        "Re-runs both policies while M's execution time grows. Without PI, every extra unit of M can "
        "delay L and therefore H. With PI, H's wait is capped by L's critical section."
    )

    sweep_range = st.slider("M execution time range (units)", 1, 30, (1, 12), key="sweep_range")
    if stretch(st.button, "Run sensitivity sweep", key="run_sweep"):
        try:
            st.session_state["sweep_df"] = run_sweep(
                st.session_state["active_tasks"], st.session_state["active_resources"],
                sweep_range[0], sweep_range[1],
            )
        except Exception as error:
            st.error(f"Sweep failed: {type(error).__name__}: {error}")

    sweep_df = st.session_state["sweep_df"]
    if sweep_df is None:
        st.info("Run the sweep to see how H's blocking and completion time scale with M's workload.")
    else:
        without = sweep_df[sweep_df["Policy"] == "Without PI"]
        with_pi = sweep_df[sweep_df["Policy"] == "With PI"]

        def first_miss(frame: pd.DataFrame) -> str:
            missed = frame[frame["Deadline missed"]]
            return f"M = {int(missed['M execution'].min())}" if not missed.empty else "never"

        s1, s2, s3 = st.columns(3)
        s1.metric("H blocking range, without PI",
                  f"{fmt(without['H blocking'].min())} – {fmt(without['H blocking'].max())}")
        s2.metric("H blocking range, with PI",
                  f"{fmt(with_pi['H blocking'].min())} – {fmt(with_pi['H blocking'].max())}")
        s3.metric("First deadline miss (without / with PI)",
                  f"{first_miss(without)} / {first_miss(with_pi)}")

        left, right = st.columns(2)
        with left:
            fig_block = px.line(
                sweep_df, x="M execution", y="H blocking", color="Policy", markers=True,
                color_discrete_map=POLICY_COLORS,
            )
            fig_block.update_layout(xaxis_title="M execution time", yaxis_title="H blocking time", legend_title_text="")
            stretch(st.plotly_chart, style_fig(fig_block, 380, "H blocking time vs M workload"), key="sweep_block")
        with right:
            fig_done = px.line(
                sweep_df, x="M execution", y="H completion", color="Policy", markers=True,
                color_discrete_map=POLICY_COLORS,
            )
            fig_done.add_hline(y=h_deadline, line_dash="dot", line_color="#FFCC66",
                               annotation_text="H deadline", annotation_position="top left")
            fig_done.update_layout(xaxis_title="M execution time", yaxis_title="H completion time", legend_title_text="")
            stretch(st.plotly_chart, style_fig(fig_done, 380, "H completion time vs M workload"), key="sweep_done")

        st.download_button(
            "Download sweep data (CSV)",
            data=sweep_df.to_csv(index=False).encode("utf-8"),
            file_name="priority_inversion_sweep.csv",
            mime="text/csv",
        )


# =========================================================
# AI TAB
# =========================================================

with tab_ai:
    st.markdown("### AI-based early risk detection")
    st.caption("The classifier is an early-warning layer; the scheduling mechanism itself is Priority Inheritance.")

    try:
        risk_before = assess_risk(st.session_state["baseline_state"], baseline)
        risk_after = assess_risk(st.session_state["pi_state"], pi)
    except Exception as error:
        st.error("AI risk prediction failed.")
        st.code(f"{type(error).__name__}: {error}")
    else:
        card_a, card_b = st.columns(2)
        for column, title, risk in (
            (card_a, "Baseline assessment (without PI)", risk_before),
            (card_b, "After Priority Inheritance", risk_after),
        ):
            with column:
                html(
                    f"""
                    <div class="glass-card">
                    <div class="section-label">{title}</div>
                    <div class="{RISK_CLASS.get(risk["prediction"], "risk-low")}">{risk["prediction"]} risk</div>
                    <div class="small-muted">Model confidence: <b>{risk["confidence"]:.1f}%</b></div>
                    </div>
                    """
                )

        order = [lvl for lvl in RISK_ORDER if lvl in risk_before["probabilities"]] + [
            lvl for lvl in risk_before["probabilities"] if lvl not in RISK_ORDER
        ]
        probability_df = pd.DataFrame(
            [
                {"Risk level": level, "Scenario": label, "Probability (%)": risk["probabilities"].get(level, 0.0) * 100}
                for label, risk in (("Without PI", risk_before), ("With PI", risk_after))
                for level in order
            ]
        )
        fig_ai = px.bar(
            probability_df, x="Risk level", y="Probability (%)", color="Scenario", barmode="group",
            text="Probability (%)", category_orders={"Risk level": order}, color_discrete_map=POLICY_COLORS,
        )
        fig_ai.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        fig_ai.update_layout(yaxis_range=[0, 110], legend_title_text="")
        stretch(st.plotly_chart, style_fig(fig_ai, 360, "Model risk probabilities"), key="chart_ai")

        st.markdown("### Features supplied to the model")
        feature_df = pd.DataFrame(
            [
                {"Feature": name, "Without PI": str(risk_before["features"][name]), "With PI": str(risk_after["features"][name])}
                for name in risk_before["features"]
            ]
        )
        stretch(st.dataframe, feature_df, hide_index=True)

        if risk_before["prediction"] == "HIGH":
            st.error("High-risk condition detected: H is waiting for R1 while R1 is held by lower-priority L.")
        elif risk_before["prediction"] == "MEDIUM":
            st.warning("Medium-risk condition detected: the current scheduling state suggests potential inversion.")
        else:
            st.success("Low-risk condition: the current scheduling state shows limited inversion risk.")

        st.caption(
            "Academic note: the training dataset is synthetic and domain-informed. "
            "The classifier demonstrates the AI pipeline within the simulation and is not a production risk model."
        )


# =========================================================
# LOGS TAB
# =========================================================

with tab_logs:
    st.markdown("### Scheduler event logs")

    log_a, log_b = st.columns(2)
    for column, prefix, label in ((log_a, "baseline", "Without Priority Inheritance"), (log_b, "pi", "With Priority Inheritance")):
        with column:
            st.markdown(f"#### {label}")
            text = "\n".join(str(line) for line in (st.session_state[f"{prefix}_log"] or []))
            st.code(text, language="text")
            st.download_button(
                "Download log",
                data=text.encode("utf-8"),
                file_name=f"scheduler_log_{prefix}.txt",
                mime="text/plain",
                key=f"download_log_{prefix}",
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption("Educational simulation · Priority Inheritance for Reducing Priority Inversion in Real-Time Systems")