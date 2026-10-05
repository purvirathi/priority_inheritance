import pandas as pd

from simulator import simulate_without_priority_inheritance
from priority_inheritance import simulate_with_priority_inheritance
from metrics import calculate_all_metrics


def run_comparison():
    baseline_history, baseline_state, baseline_log = (
        simulate_without_priority_inheritance()
    )

    pi_history, pi_state, pi_log = (
        simulate_with_priority_inheritance()
    )

    baseline_metrics = calculate_all_metrics(
        baseline_history,
        baseline_state,
    )

    pi_metrics = calculate_all_metrics(
        pi_history,
        pi_state,
    )

    task_rows = []

    for task in ["L", "M", "H"]:
        task_rows.append(
            {
                "Task": task,
                "Blocking — Without PI":
                    baseline_metrics[task]["blocking_time"],
                "Blocking — With PI":
                    pi_metrics[task]["blocking_time"],
                "Response — Without PI":
                    baseline_metrics[task]["response_time"],
                "Response — With PI":
                    pi_metrics[task]["response_time"],
                "Completion — Without PI":
                    baseline_metrics[task]["completion_time"],
                "Completion — With PI":
                    pi_metrics[task]["completion_time"],
                "Deadline Miss — Without PI":
                    "YES"
                    if baseline_metrics[task]["deadline_miss"]
                    else "NO",
                "Deadline Miss — With PI":
                    "YES"
                    if pi_metrics[task]["deadline_miss"]
                    else "NO",
            }
        )

    task_df = pd.DataFrame(task_rows)

    system_df = pd.DataFrame(
        [
            {
                "Metric": "CPU Utilization (%)",
                "Without PI":
                    baseline_metrics["system"]["cpu_utilization"],
                "With PI":
                    pi_metrics["system"]["cpu_utilization"],
            },
            {
                "Metric": "Fairness Index",
                "Without PI":
                    baseline_metrics["system"]["fairness"],
                "With PI":
                    pi_metrics["system"]["fairness"],
            },
            {
                "Metric": "Deadline Misses",
                "Without PI":
                    baseline_metrics["system"]["deadline_misses"],
                "With PI":
                    pi_metrics["system"]["deadline_misses"],
            },
        ]
    )

    return {
        "baseline": baseline_metrics,
        "pi": pi_metrics,
        "baseline_history": baseline_history,
        "pi_history": pi_history,
        "baseline_log": baseline_log,
        "pi_log": pi_log,
        "task_df": task_df,
        "system_df": system_df,
    }


if __name__ == "__main__":
    result = run_comparison()

    print("\nTASK COMPARISON")
    print(
        result["task_df"].to_string(
            index=False
        )
    )

    print("\nSYSTEM COMPARISON")
    print(
        result["system_df"].to_string(
            index=False
        )
    )
