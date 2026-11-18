
from itertools import product

import pennylane as qml
from pennylane import numpy as np


def classical_scheduler(targets, available_time):
    """Find the exact classical optimum using exhaustive search."""
    best_schedule = []
    best_priority = 0
    best_time = 0

    for selection in product([0, 1], repeat=len(targets)):
        selected = [
            target for target, chosen in zip(targets, selection) if chosen
        ]

        total_time = sum(t["duration"] for t in selected)
        total_priority = sum(t["priority"] for t in selected)

        if total_time <= available_time and total_priority > best_priority:
            best_schedule = selected
            best_priority = total_priority
            best_time = total_time

    return {
        "selected_targets": best_schedule,
        "total_priority": best_priority,
        "total_observation_time": best_time,
        "available_time": available_time,
    }


def quantum_scheduler(targets, available_time, steps=100):
    """Experimental QAOA scheduler using PennyLane."""

    n = len(targets)

    if n == 0:
        return {
            "selected_targets": [],
            "total_priority": 0,
            "total_observation_time": 0,
            "available_time": available_time,
            "method": "PennyLane QAOA",
        }

    if n > 6:
        raise ValueError("QAOA demo supports a maximum of 6 targets.")

    if available_time < 0 or any(
        t["duration"] < 0 or t["priority"] < 0 for t in targets
    ):
        raise ValueError("Durations, priorities and available time must be non-negative.")

    durations = [t["duration"] for t in targets]
    priorities = [t["priority"] for t in targets]

    # Binary variables: x_i = 1 means observe target i.
    # Penalize schedules that exceed telescope time.
    penalty = 2.0 * max(priorities + [1]) / max(1, min(durations + [1]))

    def cost(bits):
        total_time = sum(durations[i] * bits[i] for i in range(n))
        total_priority = sum(priorities[i] * bits[i] for i in range(n))
        violation = max(0, total_time - available_time)
        return -total_priority + penalty * violation**2

    # Convert the classical cost into a diagonal Hamiltonian.
    energies = {}
    for bits in product([0, 1], repeat=n):
        energies[bits] = cost(bits)

    constant = sum(energies.values()) / (2**n)
    coefficients = []
    observables = []

    for mask in range(1, 2**n):
        coefficient = sum(
            energy * (-1) ** sum(
                bits[i] for i in range(n) if mask & (1 << i)
            )
            for bits, energy in energies.items()
        ) / (2**n)

        if abs(coefficient) > 1e-9:
            wires = [i for i in range(n) if mask & (1 << i)]
            observable = qml.PauliZ(wires[0])

            for wire in wires[1:]:
                observable = observable @ qml.PauliZ(wire)

            coefficients.append(float(coefficient))
            observables.append(observable)

    hamiltonian = qml.Hamiltonian(
        [constant] + coefficients,
        [qml.Identity(0)] + observables,
    )

    dev = qml.device("default.qubit", wires=n, shots=None)

    @qml.qnode(dev)
    def circuit(params):
        for wire in range(n):
            qml.Hadamard(wires=wire)

        for layer in range(2):
            qml.qaoa.cost_layer(params[layer, 0], hamiltonian)

            for wire in range(n):
                qml.RX(2 * params[layer, 1], wires=wire)

        return qml.probs(wires=range(n))

    def expected_cost(params):
        probabilities = circuit(params)
        return sum(
            probabilities[index] * energies[bits]
            for index, bits in enumerate(product([0, 1], repeat=n))
        )

    params = np.array(
        [[0.1, 0.1], [0.2, 0.2]],
        requires_grad=True,
    )

    optimizer = qml.AdamOptimizer(stepsize=0.05)

    for _ in range(steps):
        params = optimizer.step(expected_cost, params)

    probabilities = circuit(params)

    # Choose the highest-priority feasible solution.
    candidates = []

    for index, probability in enumerate(probabilities):
        bits = tuple(
            int(bit) for bit in format(index, f"0{n}b")
        )

        total_time = sum(
            durations[i] * bits[i] for i in range(n)
        )
        total_priority = sum(
            priorities[i] * bits[i] for i in range(n)
        )

        if total_time <= available_time:
            candidates.append(
                (total_priority, probability, bits)
            )

    best_priority, best_probability, best_bits = max(
        candidates,
        key=lambda item: (item[0], item[1]),
        default=(0, 0.0, tuple(0 for _ in range(n))),
    )


    selected = [
        targets[i] for i in range(n) if best_bits[i]
    ]

    return {
        "selected_targets": selected,
        "total_priority": sum(t["priority"] for t in selected),
        "total_observation_time": sum(t["duration"] for t in selected),
        "available_time": available_time,
        "method": "PennyLane QAOA",
    }

def vqe_scheduler(targets, available_time, steps=100):
    """Experimental VQE scheduler using PennyLane."""
    n = len(targets)

    if n == 0:
        return {
            "selected_targets": [],
            "total_priority": 0,
            "total_observation_time": 0,
            "available_time": available_time,
            "method": "PennyLane VQE",
        }

    if n > 6:
        raise ValueError("VQE demo supports a maximum of 6 targets.")

    if available_time < 0 or any(
        t["duration"] < 0 or t["priority"] < 0 for t in targets
    ):
        raise ValueError("Durations, priorities and available time must be non-negative.")

    durations = [t["duration"] for t in targets]
    priorities = [t["priority"] for t in targets]

    penalty = 2.0 * max(priorities + [1]) / max(1, min(durations + [1]))

    def cost(bits):
        total_time = sum(durations[i] * bits[i] for i in range(n))
        total_priority = sum(priorities[i] * bits[i] for i in range(n))
        violation = max(0, total_time - available_time)
        return -total_priority + penalty * violation**2

    energies = {
        bits: cost(bits)
        for bits in product([0, 1], repeat=n)
    }

    constant = sum(energies.values()) / (2**n)
    coefficients = []
    observables = []

    for mask in range(1, 2**n):
        coefficient = sum(
            energy * (-1) ** sum(
                bits[i] for i in range(n) if mask & (1 << i)
            )
            for bits, energy in energies.items()
        ) / (2**n)

        if abs(coefficient) > 1e-9:
            wires = [i for i in range(n) if mask & (1 << i)]
            observable = qml.PauliZ(wires[0])

            for wire in wires[1:]:
                observable = observable @ qml.PauliZ(wire)

            coefficients.append(float(coefficient))
            observables.append(observable)

    hamiltonian = qml.Hamiltonian(
        [constant] + coefficients,
        [qml.Identity(0)] + observables,
    )

    dev = qml.device("default.qubit", wires=n, shots=None)

    layers = 2
    shape = qml.StronglyEntanglingLayers.shape(
        n_layers=layers, n_wires=n
    )

    @qml.qnode(dev)
    def circuit(params):
        qml.StronglyEntanglingLayers(
            params, wires=range(n)
        )
        return qml.expval(hamiltonian)

    @qml.qnode(dev)
    def probability_circuit(params):
        qml.StronglyEntanglingLayers(
            params, wires=range(n)
        )
        return qml.probs(wires=range(n))

    params = np.array(
        np.random.uniform(0, 2 * np.pi, size=shape),
        requires_grad=True,
    )

    optimizer = qml.AdamOptimizer(stepsize=0.05)

    for _ in range(steps):
        params = optimizer.step(circuit, params)

    probabilities = probability_circuit(params)

    candidates = []

    for index, probability in enumerate(probabilities):
        bits = tuple(
            int(bit) for bit in format(index, f"0{n}b")
        )

        total_time = sum(
            durations[i] * bits[i] for i in range(n)
        )
        total_priority = sum(
            priorities[i] * bits[i] for i in range(n)
        )

        if total_time <= available_time:
            candidates.append((total_priority, probability, bits))

    best_priority, best_probability, best_bits = max(
        candidates,
        key=lambda item: (item[0], item[1]),
        default=(0, 0.0, tuple(0 for _ in range(n))),
    )

    selected = [
        targets[i] for i in range(n) if best_bits[i]
    ]

    return {
        "selected_targets": selected,
        "total_priority": sum(t["priority"] for t in selected),
        "total_observation_time": sum(t["duration"] for t in selected),
        "available_time": available_time,
        "method": "PennyLane VQE",
    }
