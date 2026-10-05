from copy import deepcopy
from typing import Dict, List, Tuple

from scenarios import create_priority_inversion_scenario


def run_scheduler(
    use_priority_inheritance: bool = False,
    tasks=None,
    resources=None,
):
    """
    Run a deterministic single-CPU, preemptive, priority-based simulation.

    History contains one state event for every task at every simulation time.
    This makes the results easy to measure and visualize.
    """

    if tasks is None or resources is None:
        tasks, resources = create_priority_inversion_scenario()

    task_state: Dict[str, dict] = {}

    for task in tasks:
        task_state[task.name] = {
            "config": task,
            "remaining": task.execution_time,
            "state": "NOT_ARRIVED",
            "completion_time": None,
            "original_priority": task.priority,
            "effective_priority": task.priority,
            "start_time": None,
        }

    resource_owner = {
        resource.name: None
        for resource in resources
    }

    history: List[dict] = []
    log: List[str] = []

    time = 0

    while not all(
        state["state"] == "COMPLETED"
        for state in task_state.values()
    ):

        # -------------------------------------------------
        # 1. Task arrivals
        # -------------------------------------------------
        for name, state in task_state.items():
            config = state["config"]

            if (
                state["state"] == "NOT_ARRIVED"
                and time >= config.arrival_time
            ):
                state["state"] = "READY"

                log.append(
                    f"Time {time}: Task {name} arrives -> READY"
                )

        # -------------------------------------------------
        # 2. Resource acquisition / blocking
        # -------------------------------------------------
        for name, state in task_state.items():
            if state["state"] != "READY":
                continue

            config = state["config"]

            if not config.needs_resource:
                continue

            resource = config.resource_name

            if resource_owner[resource] is None:
                resource_owner[resource] = name

                log.append(
                    f"Time {time}: Task {name} locks {resource}"
                )

            elif resource_owner[resource] != name:
                state["state"] = "BLOCKED"

                log.append(
                    f"Time {time}: Task {name} BLOCKED "
                    f"(R1 owned by {resource_owner[resource]})"
                )

        # -------------------------------------------------
        # 3. Unblock waiting task if resource became free
        # -------------------------------------------------
        for name, state in task_state.items():
            if state["state"] != "BLOCKED":
                continue

            config = state["config"]
            resource = config.resource_name

            if (
                resource is not None
                and resource_owner[resource] is None
            ):
                resource_owner[resource] = name
                state["state"] = "READY"

                log.append(
                    f"Time {time}: Task {name} acquires "
                    f"{resource} -> READY"
                )

        # -------------------------------------------------
        # 4. Set effective priorities
        # -------------------------------------------------
        for state in task_state.values():
            state["effective_priority"] = state["original_priority"]

        if use_priority_inheritance:
            # Each resource owner inherits the maximum priority
            # of any blocked task waiting for that resource.
            for name, blocked_state in task_state.items():

                if blocked_state["state"] != "BLOCKED":
                    continue

                blocked_config = blocked_state["config"]
                resource = blocked_config.resource_name

                if resource is None:
                    continue

                owner_name = resource_owner[resource]

                if owner_name is None:
                    continue

                owner_state = task_state[owner_name]

                if (
                    blocked_config.priority
                    > owner_state["effective_priority"]
                ):
                    old_priority = owner_state["effective_priority"]

                    owner_state["effective_priority"] = (
                        blocked_config.priority
                    )

                    log.append(
                        f"Time {time}: PRIORITY INHERITANCE "
                        f"{owner_name}: {old_priority} -> "
                        f"{owner_state['effective_priority']} "
                        f"because {name} is blocked"
                    )

        # -------------------------------------------------
        # 5. Select highest effective-priority READY task
        # -------------------------------------------------
        runnable_tasks = [
            (
                state["effective_priority"],
                name,
            )
            for name, state in task_state.items()
            if state["state"] == "READY"
        ]

        selected_name = None

        if runnable_tasks:
            runnable_tasks.sort(
                key=lambda item: (
                    item[0],
                    -task_state[item[1]]["config"].arrival_time,
                    item[1],
                ),
                reverse=True,
            )

            selected_name = runnable_tasks[0][1]
            task_state[selected_name]["state"] = "RUNNING"

            selected_state = task_state[selected_name]

            if selected_state["start_time"] is None:
                selected_state["start_time"] = time

            log.append(
                f"Time {time}: {selected_name} RUNNING"
            )

        else:
            log.append(
                f"Time {time}: CPU IDLE"
            )

        # -------------------------------------------------
        # 6. Record the state of every task for this time
        # -------------------------------------------------
        current_owner = resource_owner.get("R1")

        for name, state in task_state.items():

            history.append(
                {
                    "time": time,
                    "task": name,
                    "state": state["state"],
                    "effective_priority": state["effective_priority"],
                    "original_priority": state["original_priority"],
                    "resource_owner": current_owner,
                }
            )

        # -------------------------------------------------
        # 7. Execute one CPU time unit
        # -------------------------------------------------
        if selected_name is not None:

            selected = task_state[selected_name]
            selected["remaining"] -= 1

            if selected["remaining"] == 0:

                selected["state"] = "COMPLETED"
                selected["completion_time"] = time + 1

                config = selected["config"]

                if (
                    config.needs_resource
                    and config.resource_name
                    and resource_owner[config.resource_name]
                    == selected_name
                ):
                    resource_owner[config.resource_name] = None

                    log.append(
                        f"Time {time + 1}: {selected_name} "
                        f"releases {config.resource_name}"
                    )

                selected["effective_priority"] = (
                    selected["original_priority"]
                )

                log.append(
                    f"Time {time + 1}: {selected_name} COMPLETED"
                )

            else:
                selected["state"] = "READY"

        time += 1

        if time > 100:
            raise RuntimeError(
                "Simulation exceeded 100 time units. "
                "Check task/resource logic."
            )

    return history, task_state, log


def simulate_without_priority_inheritance(tasks=None, resources=None):
    return run_scheduler(
        use_priority_inheritance=False,
        tasks=tasks,
        resources=resources,
    )


def simulate_with_priority_inheritance(tasks=None, resources=None):
    return run_scheduler(
        use_priority_inheritance=True,
        tasks=tasks,
        resources=resources,
    )
