from scheduler_engine import simulate_without_priority_inheritance
from metrics import calculate_all_metrics


if __name__ == "__main__":
    history, task_state, log = (
        simulate_without_priority_inheritance()
    )

    metrics = calculate_all_metrics(
        history,
        task_state,
    )

    print("\n" + "=" * 65)
    print("BASELINE — WITHOUT PRIORITY INHERITANCE")
    print("=" * 65)

    for line in log:
        print(line)

    print("\nMetrics:")
    for task_name, values in metrics.items():
        print(f"\n{task_name}:")
        for metric, value in values.items():
            print(f"  {metric}: {value}")
