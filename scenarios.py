from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    name: str
    priority: int
    arrival_time: int
    execution_time: int
    deadline: int
    needs_resource: bool = False
    resource_name: str | None = None


@dataclass(frozen=True)
class ResourceConfig:
    name: str


def create_priority_inversion_scenario():
    """
    Classic three-task priority-inversion scenario.

    L: low priority, owns shared resource R1
    H: high priority, requests R1
    M: medium priority, does not use R1

    The timing is intentionally chosen so that:
    - Without PI, M can preempt L while H is blocked.
    - With PI, L inherits H's priority and releases R1 earlier.
    """

    tasks = [
        TaskConfig(
            name="L",
            priority=1,
            arrival_time=0,
            execution_time=6,
            deadline=15,
            needs_resource=True,
            resource_name="R1",
        ),
        TaskConfig(
            name="M",
            priority=2,
            arrival_time=2,
            execution_time=4,
            deadline=12,
            needs_resource=False,
        ),
        TaskConfig(
            name="H",
            priority=3,
            arrival_time=1,
            execution_time=2,
            deadline=8,
            needs_resource=True,
            resource_name="R1",
        ),
    ]

    resources = [
        ResourceConfig(name="R1"),
    ]

    return tasks, resources
