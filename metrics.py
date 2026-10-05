def calculate_blocking_time(history, task_name):
    """
    Number of discrete time units during which the task
    is BLOCKED.
    """
    return sum(
        1
        for event in history
        if (
            event["task"] == task_name
            and event["state"] == "BLOCKED"
        )
    )


def calculate_response_time(completion_time, arrival_time):
    """
    Completion time - arrival time.
    """
    return completion_time - arrival_time


def check_deadline_miss(completion_time, deadline):
    return completion_time > deadline


def calculate_cpu_time(history, task_name):
    """
    CPU service actually received by a task.
    Each RUNNING event represents one time unit.
    """
    return sum(
        1
        for event in history
        if (
            event["task"] == task_name
            and event["state"] == "RUNNING"
        )
    )


def calculate_cpu_utilization(history):
    """
    Busy CPU time / total simulated time * 100.
    """
    if not history:
        return 0.0

    total_time = max(
        event["time"]
        for event in history
    ) + 1

    busy_time = len(
        {
            event["time"]
            for event in history
            if event["state"] == "RUNNING"
        }
    )

    if total_time <= 0:
        return 0.0

    return busy_time / total_time * 100.0


def calculate_fairness(history, task_names):
    """
    Jain's Fairness Index:

        F = (sum(x_i))^2 / (n * sum(x_i^2))

    where x_i is CPU time received by each task.
    """
    cpu_times = [
        calculate_cpu_time(history, task)
        for task in task_names
    ]

    if not cpu_times:
        return 0.0

    denominator = (
        len(cpu_times)
        * sum(value ** 2 for value in cpu_times)
    )

    if denominator == 0:
        return 0.0

    return (
        sum(cpu_times) ** 2
        / denominator
    )


def calculate_all_metrics(history, task_state):
    metrics = {}

    for name, state in task_state.items():

        config = state["config"]
        completion_time = state["completion_time"]

        metrics[name] = {
            "blocking_time": calculate_blocking_time(
                history,
                name,
            ),
            "response_time": calculate_response_time(
                completion_time,
                config.arrival_time,
            ),
            "completion_time": completion_time,
            "deadline": config.deadline,
            "deadline_miss": check_deadline_miss(
                completion_time,
                config.deadline,
            ),
            "cpu_time": calculate_cpu_time(
                history,
                name,
            ),
        }

    task_names = list(task_state.keys())

    total_deadline_misses = sum(
        int(values["deadline_miss"])
        for name, values in metrics.items()
        if name != "system"
    )

    metrics["system"] = {
        "cpu_utilization": calculate_cpu_utilization(history),
        "fairness": calculate_fairness(
            history,
            task_names,
        ),
        "deadline_misses": total_deadline_misses,
    }

    return metrics
