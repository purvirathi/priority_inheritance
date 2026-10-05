import pandas as pd
import plotly.express as px
import streamlit as st

from metrics import calculate_all_metrics
from risk_model import load_model, predict_risk
from scenarios import TaskConfig, ResourceConfig, create_priority_inversion_scenario
from scheduler_engine import (
    simulate_with_priority_inheritance,
    simulate_without_priority_inheritance,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Priority Inversion Analyzer",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "baseline_history": None,
    "baseline_state": None,
    "baseline_metrics": None,
    "baseline_log": None,
    "pi_history": None,
    "pi_state": None,
    "pi_metrics": None,
    "pi_log": None,
}

for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Active experiment configuration. The classic scenario is always
# available as the default/reset state.
if "active_tasks" not in st.session_state:
    default_tasks, default_resources = create_priority_inversion_scenario()
    st.session_state["active_tasks"] = default_tasks
    st.session_state["active_resources"] = default_resources

if "scenario_mode" not in st.session_state:
    st.session_state["scenario_mode"] = "Classic Demo"


def both_results_ready() -> bool:
    return all(
        st.session_state[key] is not None
        for key in [
            "baseline_history",
            "baseline_state",
            "baseline_metrics",
            "pi_history",
            "pi_state",
            "pi_metrics",
        ]
    )


def clear_results() -> None:
    for key, default in DEFAULT_STATE.items():
        st.session_state[key] = default


def set_active_tasks(tasks, resources) -> None:
    st.session_state["active_tasks"] = tasks
    st.session_state["active_resources"] = resources
    clear_results()


def reset_to_classic_scenario() -> None:
    tasks, resources = create_priority_inversion_scenario()
    set_active_tasks(tasks, resources)
    st.session_state["scenario_mode"] = "Classic Demo"


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    :root {
        --bg: #050908;
        --panel: rgba(255,255,255,0.055);
        --panel-strong: rgba(255,255,255,0.075);
        --border: rgba(255,255,255,0.11);
        --text: #F2F7F5;
        --muted: #9FB5AD;
        --green: #65D6A2;
        --green-dark: #183B35;
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

    [data-testid="stSidebar"] {
        background: rgba(5, 10, 9, 0.90);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .hero {
        padding: 8px 0 8px 0;
    }

    .eyebrow {
        color: var(--green);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
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
        background: linear-gradient(
            145deg,
            rgba(255,255,255,0.075),
            rgba(255,255,255,0.035)
        );
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 16px 50px rgba(0,0,0,0.24);
        backdrop-filter: blur(14px);
    }

    .section-label {
        color: var(--green);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 800;
        margin-bottom: 14px;
    }

    .status-title {
        color: #C6D5D0;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .status-value {
        font-size: 24px;
        font-weight: 800;
        margin-top: 5px;
    }

    .status-dot {
        color: var(--green);
        margin-right: 6px;
    }

    .kpi-label {
        color: #92A79F;
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .kpi-value {
        font-size: 31px;
        line-height: 1.15;
        font-weight: 850;
        margin-top: 6px;
    }

    .kpi-note {
        color: #8EA39B;
        font-size: 12px;
        margin-top: 5px;
    }

    .danger-box,
    .success-box,
    .info-box {
        border-radius: 18px;
        padding: 20px;
        height: 100%;
    }

    .danger-box {
        background: rgba(255,80,80,0.085);
        border: 1px solid rgba(255,80,80,0.22);
    }

    .success-box {
        background: rgba(60,200,140,0.085);
        border: 1px solid rgba(60,200,140,0.22);
    }

    .info-box {
        background: rgba(115,183,255,0.075);
        border: 1px solid rgba(115,183,255,0.20);
    }

    .flow-node {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 13px;
        text-align: center;
        font-weight: 750;
    }

    .arrow {
        color: var(--green);
        text-align: center;
        font-size: 22px;
        padding-top: 9px;
    }

    .risk-high {
        color: var(--red);
        font-size: 34px;
        font-weight: 900;
        letter-spacing: 0.5px;
    }

    .risk-medium {
        color: var(--amber);
        font-size: 34px;
        font-weight: 900;
    }

    .risk-low {
        color: var(--green);
        font-size: 34px;
        font-weight: 900;
    }

    .small-muted {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
    }

    .metric-chip {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        color: #D6E2DE;
        font-size: 12px;
        margin-right: 5px;
        margin-bottom: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## Control Center")
    st.caption("Configure and run the real-time scheduling experiment")

    active_tasks = {task.name: task for task in st.session_state["active_tasks"]}

    st.markdown("### Scenario")
    scenario_choice = st.radio(
        "Scenario mode",
        ["Classic Demo", "Custom Demo"],
        key="scenario_mode",
        label_visibility="collapsed",
    )

    if scenario_choice == "Classic Demo":
        st.markdown(
            "<div class='small-muted'>Classic priority-inversion workload is active.</div>",
            unsafe_allow_html=True,
        )
        if st.button("Reset to Classic Scenario", use_container_width=True):
            reset_to_classic_scenario()
            st.rerun()

    else:
        st.markdown("#### Task parameters")
        st.caption("H and L share R1. M does not use R1.")

        with st.form("custom_scenario_form"):
            st.markdown("**H — High Priority**")
            h_priority = st.number_input("H priority", min_value=1, max_value=10, value=active_tasks["H"].priority, step=1)
            h_arrival = st.number_input("H arrival", min_value=0, max_value=50, value=active_tasks["H"].arrival_time, step=1)
            h_execution = st.number_input("H execution", min_value=1, max_value=50, value=active_tasks["H"].execution_time, step=1)
            h_deadline = st.number_input("H deadline", min_value=1, max_value=100, value=active_tasks["H"].deadline, step=1)

            st.markdown("**M — Medium Priority**")
            m_priority = st.number_input("M priority", min_value=1, max_value=10, value=active_tasks["M"].priority, step=1)
            m_arrival = st.number_input("M arrival", min_value=0, max_value=50, value=active_tasks["M"].arrival_time, step=1)
            m_execution = st.number_input("M execution", min_value=1, max_value=50, value=active_tasks["M"].execution_time, step=1)
            m_deadline = st.number_input("M deadline", min_value=1, max_value=100, value=active_tasks["M"].deadline, step=1)

            st.markdown("**L — Low Priority**")
            l_priority = st.number_input("L priority", min_value=1, max_value=10, value=active_tasks["L"].priority, step=1)
            l_arrival = st.number_input("L arrival", min_value=0, max_value=50, value=active_tasks["L"].arrival_time, step=1)
            l_execution = st.number_input("L execution", min_value=1, max_value=50, value=active_tasks["L"].execution_time, step=1)
            l_deadline = st.number_input("L deadline", min_value=1, max_value=100, value=active_tasks["L"].deadline, step=1)

            applied = st.form_submit_button("Apply Custom Scenario", use_container_width=True, type="primary")

        if applied:
            if len({h_priority, m_priority, l_priority}) < 3:
                st.error("Use distinct priorities for H, M, and L so the scheduler has a clear priority order.")
            else:
                if h_deadline < h_arrival + h_execution:
                    st.warning("H's deadline is earlier than its nominal arrival + execution window; this creates a deliberately stressed case.")

                tasks = [
                    TaskConfig("L", int(l_priority), int(l_arrival), int(l_execution), int(l_deadline), True, "R1"),
                    TaskConfig("M", int(m_priority), int(m_arrival), int(m_execution), int(m_deadline), False, None),
                    TaskConfig("H", int(h_priority), int(h_arrival), int(h_execution), int(h_deadline), True, "R1"),
                ]
                resources = [ResourceConfig("R1")]
                set_active_tasks(tasks, resources)
                st.session_state["scenario_mode"] = "Custom Demo"
                st.success("Custom scenario applied. Previous results were cleared.")
                st.rerun()

    st.markdown("---")

    if st.button("Run Complete Experiment", use_container_width=True, type="primary"):
        tasks = st.session_state["active_tasks"]
        resources = st.session_state["active_resources"]

        baseline_history, baseline_state, baseline_log = simulate_without_priority_inheritance(tasks, resources)
        pi_history, pi_state, pi_log = simulate_with_priority_inheritance(tasks, resources)

        st.session_state["baseline_history"] = baseline_history
        st.session_state["baseline_state"] = baseline_state
        st.session_state["baseline_metrics"] = calculate_all_metrics(baseline_history, baseline_state)
        st.session_state["baseline_log"] = baseline_log

        st.session_state["pi_history"] = pi_history
        st.session_state["pi_state"] = pi_state
        st.session_state["pi_metrics"] = calculate_all_metrics(pi_history, pi_state)
        st.session_state["pi_log"] = pi_log

        st.success("Complete experiment finished.")

    if st.button("Run Baseline Only", use_container_width=True):
        tasks = st.session_state["active_tasks"]
        resources = st.session_state["active_resources"]
        history, state, log = simulate_without_priority_inheritance(tasks, resources)
        st.session_state["baseline_history"] = history
        st.session_state["baseline_state"] = state
        st.session_state["baseline_metrics"] = calculate_all_metrics(history, state)
        st.session_state["baseline_log"] = log
        st.info("Baseline simulation updated.")

    if st.button("Run PI Only", use_container_width=True):
        tasks = st.session_state["active_tasks"]
        resources = st.session_state["active_resources"]
        history, state, log = simulate_with_priority_inheritance(tasks, resources)
        st.session_state["pi_history"] = history
        st.session_state["pi_state"] = state
        st.session_state["pi_metrics"] = calculate_all_metrics(history, state)
        st.session_state["pi_log"] = log
        st.info("Priority Inheritance simulation updated.")

    if st.button("Reset Results", use_container_width=True):
        clear_results()
        st.rerun()

    st.markdown("---")
    st.markdown("### Active workload")
    for task_name in ["H", "M", "L"]:
        task = {task.name: task for task in st.session_state["active_tasks"]}[task_name]
        st.markdown(
            f"<div class='small-muted'><b>{task.name}</b> · P{task.priority} · arrival {task.arrival_time} · exec {task.execution_time} · deadline {task.deadline}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Metric definitions")
    st.caption("Blocking: discrete time units in BLOCKED state")
    st.caption("Response: completion time − arrival time")
    st.caption("Deadline miss: completion time > deadline")
    st.caption("CPU utilization: busy time / total simulation time")
    st.caption("Fairness: Jain's Fairness Index")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Operating Systems • Real-Time Scheduling</div>
        <div class="main-title">Real-Time Priority Inversion Analyzer</div>
        <div class="subtitle">
            A simulation-driven dashboard that demonstrates priority inversion,
            applies Priority Inheritance, measures scheduling performance,
            and uses a lightweight AI classifier for early risk detection.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# STATUS CARDS
# =========================================================

s1, s2, s3 = st.columns(3)

with s1:
    st.markdown(
        """
        <div class="glass-card">
            <div class="status-title">Scheduler Simulator</div>
            <div class="status-value"><span class="status-dot">●</span>READY</div>
            <div class="small-muted">Deterministic single-CPU experiment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s2:
    st.markdown(
        """
        <div class="glass-card">
            <div class="status-title">Priority Inheritance</div>
            <div class="status-value"><span class="status-dot">●</span>AVAILABLE</div>
            <div class="small-muted">Dynamic effective-priority boosting</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s3:
    try:
        load_model()
        ai_status = "LOADED"
        ai_note = "Decision Tree risk classifier"
    except Exception:
        ai_status = "NEEDS TRAINING"
        ai_note = "Model will train automatically"

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="status-title">AI Risk Model</div>
            <div class="status-value"><span class="status-dot">●</span>{ai_status}</div>
            <div class="small-muted">{ai_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SCENARIO OVERVIEW
# =========================================================

st.divider()
st.markdown('<div class="section-label">Scenario</div>', unsafe_allow_html=True)
scenario_title = "Classic Priority-Inversion Setup" if st.session_state["scenario_mode"] == "Classic Demo" else "Custom Priority-Inversion Workload"
st.markdown(f'<div class="section-title">{scenario_title}</div>', unsafe_allow_html=True)

c1, c2 = st.columns([1.35, 1])

with c1:
    active = {task.name: task for task in st.session_state["active_tasks"]}
    priority_class = {"H": "High", "M": "Medium", "L": "Low"}
    scenario_df = pd.DataFrame(
        [
            [
                name,
                priority_class[name],
                active[name].priority,
                active[name].arrival_time,
                active[name].execution_time,
                active[name].deadline,
                active[name].resource_name or "CPU",
            ]
            for name in ["H", "M", "L"]
        ],
        columns=[
            "Task", "Priority Class", "Priority", "Arrival",
            "Execution", "Deadline", "Resource",
        ],
    )

    st.dataframe(
        scenario_df,
        use_container_width=True,
        hide_index=True,
    )

with c2:
    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">Resource Contention</div>
            <b>R1</b> is the shared resource.
            <br><br>
            L is configured to own R1 and H is configured to request R1.
            M does not use R1, so it can only delay L through CPU scheduling.
            <br><br>
            <b>Proposed mechanism:</b> L temporarily inherits H's
            effective priority while H is blocked on R1.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# EXPERIMENT STATUS
# =========================================================

if not both_results_ready():
    st.divider()
    st.markdown('<div class="section-label">Ready to Run</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Launch the complete experiment</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            Run the complete experiment from the sidebar. The application will
            execute both scheduling policies for the active workload, calculate
            all evaluation metrics, render the timeline, and send the measured
            baseline conditions to the AI risk model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# =========================================================
# RESULTS
# =========================================================

baseline = st.session_state["baseline_metrics"]
pi = st.session_state["pi_metrics"]

h_base = baseline["H"]
h_pi = pi["H"]

st.divider()
st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Before vs After Priority Inheritance</div>', unsafe_allow_html=True)


# =========================================================
# KEY KPIs
# =========================================================

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">H Blocking Time</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="kpi-value">{h_base["blocking_time"]} → {h_pi["blocking_time"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="kpi-note">Lower is better</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with k2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">H Response Time</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="kpi-value">{h_base["response_time"]} → {h_pi["response_time"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="kpi-note">Completion − arrival</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with k3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">H Completion Time</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="kpi-value">{h_base["completion_time"]} → {h_pi["completion_time"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="kpi-note">Simulation time units</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with k4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">H Deadline</div>', unsafe_allow_html=True)
    deadline_result = "MISSED → MET" if h_base["deadline_miss"] and not h_pi["deadline_miss"] else (
        "MET → MET" if not h_base["deadline_miss"] and not h_pi["deadline_miss"] else "MISSED → MISSED"
    )
    st.markdown(
        f'<div class="kpi-value">{deadline_result}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="kpi-note">Deadline = 8 time units</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# TABS
# =========================================================

tab_overview, tab_compare, tab_timeline, tab_ai, tab_logs = st.tabs(
    [
        "Overview",
        "Comparison",
        "Scheduler Timeline",
        "AI Risk Engine",
        "Execution Logs",
    ]
)


# =========================================================
# OVERVIEW TAB
# =========================================================

with tab_overview:

    st.markdown("### Quantified impact")

    block_reduction = h_base["blocking_time"] - h_pi["blocking_time"]
    response_reduction = h_base["response_time"] - h_pi["response_time"]

    block_reduction_pct = (
        block_reduction / h_base["blocking_time"] * 100
        if h_base["blocking_time"]
        else 0.0
    )

    deadline_improvement = (
        baseline["system"]["deadline_misses"]
        - pi["system"]["deadline_misses"]
    )

    q1, q2, q3 = st.columns(3)

    with q1:
        st.metric(
            "H Blocking Reduction",
            f"{block_reduction} units",
            f"{block_reduction_pct:.1f}%",
        )

    with q2:
        st.metric(
            "H Response Reduction",
            f"{response_reduction} units",
        )

    with q3:
        st.metric(
            "System Deadline Misses",
            f'{baseline["system"]["deadline_misses"]} → {pi["system"]["deadline_misses"]}',
            f"{deadline_improvement} fewer",
        )

    st.markdown("### Mechanism")

    flow_cols = st.columns(7)
    flow_items = [
        "L owns R1",
        "H requests R1",
        "H BLOCKED",
        "M runs",
        "L delayed",
        "H waits longer",
        "H misses deadline",
    ]

    for index, item in enumerate(flow_items):
        with flow_cols[index]:
            st.markdown(
                f'<div class="flow-node">{item}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("### With Priority Inheritance")

    flow_cols = st.columns(7)
    flow_items = [
        "L owns R1",
        "H requests R1",
        "H BLOCKED",
        "L inherits H",
        "L preempts M",
        "L releases R1",
        "H completes",
    ]

    for index, item in enumerate(flow_items):
        with flow_cols[index]:
            st.markdown(
                f'<div class="flow-node">{item}</div>',
                unsafe_allow_html=True,
            )


# =========================================================
# COMPARISON TAB
# =========================================================

with tab_compare:

    st.markdown("### Task-level evaluation")

    rows = []
    for task in ["L", "M", "H"]:
        rows.append(
            {
                "Task": task,
                "CPU Time — Without PI": baseline[task]["cpu_time"],
                "CPU Time — With PI": pi[task]["cpu_time"],
                "Blocking — Without PI": baseline[task]["blocking_time"],
                "Blocking — With PI": pi[task]["blocking_time"],
                "Response — Without PI": baseline[task]["response_time"],
                "Response — With PI": pi[task]["response_time"],
                "Completion — Without PI": baseline[task]["completion_time"],
                "Completion — With PI": pi[task]["completion_time"],
                "Deadline Miss — Without PI": "YES" if baseline[task]["deadline_miss"] else "NO",
                "Deadline Miss — With PI": "YES" if pi[task]["deadline_miss"] else "NO",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Blocking time")

    chart_df = pd.DataFrame(
        {
            "Task": ["L", "M", "H", "L", "M", "H"],
            "Blocking Time": [
                baseline["L"]["blocking_time"],
                baseline["M"]["blocking_time"],
                baseline["H"]["blocking_time"],
                pi["L"]["blocking_time"],
                pi["M"]["blocking_time"],
                pi["H"]["blocking_time"],
            ],
            "Scheduler": [
                "Without PI", "Without PI", "Without PI",
                "With PI", "With PI", "With PI",
            ],
        }
    )

    fig = px.bar(
        chart_df,
        x="Task",
        y="Blocking Time",
        color="Scheduler",
        barmode="group",
        text="Blocking Time",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Blocking Time by Scheduling Policy",
        xaxis_title="Task",
        yaxis_title="Time Units",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### System-level metrics")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "CPU Utilization — Baseline",
            f'{baseline["system"]["cpu_utilization"]:.1f}%',
        )
    with m2:
        st.metric(
            "CPU Utilization — PI",
            f'{pi["system"]["cpu_utilization"]:.1f}%',
        )
    with m3:
        st.metric(
            "Fairness — Baseline",
            f'{baseline["system"]["fairness"]:.3f}',
        )
    with m4:
        st.metric(
            "Fairness — PI",
            f'{pi["system"]["fairness"]:.3f}',
        )

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
        "Each horizontal block represents one simulation time unit. "
        "Running and blocked states are shown for both policies."
    )

    def build_gantt(history, scheduler_name):
        rows = []
        for event in history:
            if event["state"] not in {"RUNNING", "BLOCKED"}:
                continue
            rows.append(
                {
                    "Task": f'{event["task"]} — {scheduler_name}',
                    "Start": event["time"],
                    "Finish": event["time"] + 1,
                    "State": event["state"],
                    "Scheduler": scheduler_name,
                    "Effective Priority": event["effective_priority"],
                    "Resource Owner": event["resource_owner"] or "None",
                }
            )
        return pd.DataFrame(rows)

    gantt_df = pd.concat(
        [
            build_gantt(st.session_state["baseline_history"], "Without PI"),
            build_gantt(st.session_state["pi_history"], "With PI"),
        ],
        ignore_index=True,
    )

    fig_gantt = px.timeline(
        gantt_df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="State",
        hover_data=[
            "Scheduler",
            "Effective Priority",
            "Resource Owner",
        ],
        color_discrete_map={
            "RUNNING": "#65D6A2",
            "BLOCKED": "#FF6B6B",
        },
    )

    fig_gantt.update_yaxes(autorange="reversed")
    fig_gantt.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Execution and Blocking Timeline",
        xaxis_title="Simulation Time",
        yaxis_title="Task",
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig_gantt, use_container_width=True)

    st.markdown("### Why the timeline changes")
    e1, e2 = st.columns(2)

    with e1:
        st.markdown(
            """
            <div class="danger-box">
                <h4>Without Priority Inheritance</h4>
                <p>
                H is blocked by L. Because L keeps its low base priority,
                M can execute before L finishes its critical section.
                That indirectly extends the time H waits for R1.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with e2:
        st.markdown(
            """
            <div class="success-box">
                <h4>With Priority Inheritance</h4>
                <p>
                Once H blocks, L inherits H's effective priority.
                L therefore executes before M, releases R1 sooner,
                and allows H to continue.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# AI TAB
# =========================================================

with tab_ai:

    st.markdown("### AI-based early risk detection")
    st.caption(
        "The classifier is an early-warning layer; the scheduling mechanism itself is Priority Inheritance."
    )

    try:
        h_config = st.session_state["baseline_state"]["H"]["config"]
        l_config = st.session_state["baseline_state"]["L"]["config"]

        waiting_time = baseline["H"]["blocking_time"]

        ai_result = predict_risk(
            task_priority=h_config.priority,
            owner_priority=l_config.priority,
            waiting_time=waiting_time,
            arrival_time=h_config.arrival_time,
            resource_locked=1,
            higher_priority_waiting=1,
        )

        prediction = ai_result["prediction"]
        probabilities = ai_result["probabilities"]
        confidence = max(probabilities.values()) * 100

        risk_class = {
            "HIGH": "risk-high",
            "MEDIUM": "risk-medium",
            "LOW": "risk-low",
        }.get(prediction, "risk-low")

        a1, a2 = st.columns([1, 1.25])

        with a1:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="section-label">Baseline assessment</div>
                    <div class="{risk_class}">{prediction} RISK</div>
                    <div class="small-muted">
                    Model confidence: <b>{confidence:.1f}%</b><br><br>
                    The model sees a resource lock, a higher-priority task waiting,
                    and a positive priority difference between H and the resource owner.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with a2:
            probability_df = pd.DataFrame(
                {
                    "Risk Level": list(probabilities.keys()),
                    "Probability (%)": [value * 100 for value in probabilities.values()],
                }
            )

            fig_ai = px.bar(
                probability_df,
                x="Risk Level",
                y="Probability (%)",
                text="Probability (%)",
            )
            fig_ai.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_ai.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title="Model Risk Probabilities",
                yaxis_range=[0, 100],
                height=340,
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(fig_ai, use_container_width=True)

        st.markdown("### Features supplied to the model")
        feature_df = pd.DataFrame(
            [
                ["Task priority", h_config.priority],
                ["Resource owner priority", l_config.priority],
                ["Priority difference", h_config.priority - l_config.priority],
                ["Waiting / blocking time", waiting_time],
                ["Arrival time", h_config.arrival_time],
                ["Resource locked", "YES"],
                ["Higher-priority task waiting", "YES"],
            ],
            columns=["Feature", "Value"],
        )
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

        if prediction == "HIGH":
            st.error(
                "High-risk condition detected: H is waiting for R1 while R1 is held by lower-priority L."
            )
        elif prediction == "MEDIUM":
            st.warning(
                "Medium-risk condition detected: the current scheduling state suggests potential inversion."
            )
        else:
            st.success(
                "Low-risk condition: the current scheduling state shows limited inversion risk."
            )

        st.caption(
            "Academic note: the training dataset is synthetic and domain-informed. "
            "The classifier demonstrates the AI pipeline within the simulation and is not a production risk model."
        )

    except Exception as error:
        st.error("AI risk prediction failed.")
        st.code(f"{type(error).__name__}: {error}")


# =========================================================
# LOGS TAB
# =========================================================

with tab_logs:

    st.markdown("### Scheduler event logs")

    log1, log2 = st.columns(2)

    with log1:
        st.markdown("#### Without Priority Inheritance")
        baseline_log = st.session_state["baseline_log"] or []
        st.code("\n".join(baseline_log), language="text")

    with log2:
        st.markdown("#### With Priority Inheritance")
        pi_log = st.session_state["pi_log"] or []
        st.code("\n".join(pi_log), language="text")


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption(
    "Educational simulation • Priority Inheritance for Reducing Priority Inversion in Real-Time Systems"
)
