import warnings
import torch
import math
import einops


def autocrop(x, y):
    """Center crop the y image to the size of x"""
    xs, ys = x.shape, y.shape
    if xs > ys:
        warnings.warn("x is larger than y. Cropping x to match y")
        return autocrop(y, x)
    y_cropped = y[
        :,
        :,
        (ys[2] - xs[2]) // 2 : (ys[2] + xs[2]) // 2,
        (ys[3] - xs[3]) // 2 : (ys[3] + xs[3]) // 2,
    ]
    return x, y_cropped


def autopad(x, y):
    """Pad the y image to the size of x"""
    xs, ys = x.shape, y.shape
    if xs < ys:
        warnings.warn("x is smaller than y. Padding x to match y")
        return autopad(y, x)
    y_padded = torch.nn.functional.pad(
        y,
        (
            math.ceil((xs[3] - ys[3]) / 2),
            math.floor((xs[3] - ys[3]) / 2),
            math.ceil((xs[2] - ys[2]) / 2),
            math.floor((xs[2] - ys[2]) / 2),
        ),
        mode="constant",
        value=0,
    )
    return x, y_padded


def __get_label_embedding_1(labels: torch.Tensor, width: int, height: int):
    """Returns a mask for the labels"""
    batch = labels.shape[0]
    y = einops.repeat(labels, "b -> b w", w=width)
    mask = torch.arange(width, device=labels.device) / 20
    mask = einops.repeat(
        mask,
        "w -> b w",
        b=batch,
    )
    mask = torch.sin(y + mask)
    mask = mask * 0.1
    mask = einops.repeat(mask, "b w -> b 1 w h", h=height)
    return mask


def __get_label_embedding_2(labels: torch.Tensor, width: int, height: int):
    """Returns a mask for the labels"""
    assert (
        labels.unique().shape[0] == 2 and labels.min() == 0 and labels.max() == 1
    ), "Labels must be binary"
    batch = labels.shape[0]
    mask = torch.zeros((batch, 1, width, height), device=labels.device)
    mask[
        :,
        :,
        : width // 2,
    ] = (labels == 0).reshape(batch, 1, 1, 1).float() * 0.1
    mask[:, :, width // 2 :] = (labels == 1).reshape(batch, 1, 1, 1).float() * 0.1
    return mask


get_label_embedding = __get_label_embedding_1


def circuit_to_qasm(weights, wires, inp):
    import pennylane as qml

    dummy_qdev = qml.device("qiskit.aer", wires=wires)  # Create a dummy device

    def _dummy_circuit():
        qml.AngleEmbedding(inp, wires=range(wires))
        qml.StronglyEntanglingLayers(weights=weights, wires=range(wires))
        return qml.probs(wires=range(wires))

    dummy_qnode = qml.QNode(_dummy_circuit, dummy_qdev)  # Create a dummy qnode
    dummy_qnode()  # Execute the circuit once
    qasm_str = str(dummy_qdev._circuit.qasm(formatted=False))
    return qasm_str


def repeat_qasm(
    qasm: str,
    wires: int,
    ancilla: bool,
    reps: int,
):
    qasm_ = qasm.split("\n")
    header = "\n".join(qasm_[0:4])
    measurements = "\n".join(qasm_[-wires:])
    qasm_ = qasm_[4 : -wires - 1]
    if ancilla:
        qasm_ = [f"reset q[{wires-1}];"] + ["barrier q;"] + qasm_
    qasm_ = qasm_ + ["barrier q;"]
    qasm_mult = []
    for _ in range(reps):
        qasm_mult += qasm_
    qasm = "\n".join(qasm_mult)
    total = "\n".join([header, qasm, measurements])
    return total


def sample_from_qiskit(qasm_str, backend="statevector_simulator", shots=None):
    from qiskit import QuantumCircuit, Aer, execute
    import torch

    qc = QuantumCircuit.from_qasm_str(qasm_str)
    backend = Aer.get_backend(backend)
    job = execute(qc, backend, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    count_list = []
    for i in range(2**qc.num_qubits):
        # create keys which are padded with zeros to have the lenght of the number of qubits
        key = bin(i)[2:].zfill(qc.num_qubits)
        count_list.append(counts.get(key, 0))
    count_t = torch.tensor(count_list, dtype=torch.float32)
    return count_t
