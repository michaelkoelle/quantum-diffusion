import pennylane as qml
import typing
import warnings
import torch
import qw_map


class CircuitFactory:
    def __init__(self, wires: int) -> None:
        self.wires = wires
        self.dev = qml.device("default.qubit.torch", wires=wires)

    def classic_circuit(self, label, inp, weights):
        qml.AmplitudeEmbedding(
            inp, wires=range(0, self.wires - 1), normalize=True, pad_with=0.0
        )  # embed noisy input
        qml.RX(label, wires=self.wires - 1)  # embed label
        qml.StronglyEntanglingLayers(
            weights, wires=range(self.wires)
        )  # entangling subcircuit
        return qml.probs(wires=range(self.wires))  # return probabilities of each state

    def _embedding_only_circuit(self, label: torch.Tensor, inp):
        """
        Helper circuit as ground truth for manual testing.
        Embeds labels and input and returns the statevector.
        """
        if label.max() > 2 * torch.pi:
            warnings.warn("label out of range, using modulo")
            label = label % (torch.pi * 2)
        qml.AmplitudeEmbedding(
            inp, wires=range(0, self.wires - 1), normalize=True, pad_with=0.0
        )  # embed noisy input
        qml.RX(label, wires=self.wires - 1)  # embed label
        return qml.state()

    def manual_embedding(
        self,
        label: typing.Union[torch.Tensor, float],
        inp: torch.Tensor,
        auto_normalize=True,
    ):
        if isinstance(label, float):
            label = torch.tensor(label)
        if inp.dim() == 1:
            inp = inp.unsqueeze(0)
        if label.dim() == 1:
            label = label.unsqueeze(-1)
        if label.max() > 2 * torch.pi:
            warnings.warn("label out of range, using modulo")
            label = label % (torch.pi * 2)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        if auto_normalize:
            inp = torch.nn.functional.normalize(inp, dim=-1).double()
        zero = torch.zeros(inp.shape[-1]).double()
        factor = torch.where(
            label < torch.pi, torch.ones_like(label), -torch.ones_like(label)
        )
        real_0 = factor * (inp**2 * torch.cos(label * 0.5) ** 2) ** 0.5
        imag_1 = -((inp**2 * torch.sin(label * 0.5) ** 2) ** 0.5)
        stack_0 = torch.complex(real=real_0, imag=zero)
        stack_1 = torch.complex(real=zero, imag=imag_1)
        stack = torch.stack((stack_0, stack_1), dim=-1)
        stack = stack.flatten(start_dim=1, end_dim=2)
        return stack

    def ansatz_circuit(self, inp_state, weights):
        qml.QubitStateVector(inp_state, wires=range(self.wires))
        qml.StronglyEntanglingLayers(weights, wires=range(self.wires))
        return qml.state()

    def sampling_circuit(self, inp_state, weights, num_repeats):
        qml.QubitStateVector(inp_state, wires=range(self.wires))
        for _ in range(num_repeats):
            qml.StronglyEntanglingLayers(weights, wires=range(self.wires))
        return qml.state()

    def sampling_qnode(self, num_repeats: int):
        def __circuit(inp_image, weights):
            qml.QubitStateVector(inp_image, wires=range(self.wires))
            for _ in range(num_repeats):
                qml.StronglyEntanglingLayers(
                    qw_map.tanh(weights), wires=range(self.wires)
                )
            return qml.state()

        return qml.QNode(
            func=__circuit,
            device=self.dev,
            interface="torch",
            diff_method=None,
        )

    def sampling_qnode_with_swap(
        self, num_repeats: int, has_reuploads: bool, qnode_kwargs: dict = {}
    ):
        """
        Create a new qnode with more wires.
        For each num_repeat, an additional wire is added to inject the label rotation.
        THIS IS JUST A WORKAROUNG UNTIL PENNYLANE SUPPORTS THE RESET GATE.
        """

        def __circuit(inp_image, weights, label):
            qml.QubitStateVector(inp_image, wires=range(self.wires - 1))
            for rep in range(num_repeats):
                if has_reuploads:
                    for w in weights:
                        if label is not None:
                            qml.RX(label, wires=self.wires - 1)
                        qml.StronglyEntanglingLayers(
                            qw_map.tanh(w), wires=range(self.wires)
                        )
                else:
                    if label is not None:
                        qml.RX(label, wires=self.wires - 1)
                    qml.StronglyEntanglingLayers(
                        qw_map.tanh(weights), wires=range(self.wires)
                    )
                qml.SWAP(wires=[self.wires - 1, self.wires + rep])
            return qml.probs(wires=range(self.wires))

        return qml.QNode(
            func=__circuit,
            device=qml.device("default.qubit.torch", wires=self.wires + num_repeats),
            interface="torch",
            diff_method="backprop",
            **qnode_kwargs,
        )

    def get_qnode(self, circuit_function: typing.Callable):
        return qml.QNode(
            func=circuit_function,
            device=self.dev,
            interface="torch",
            diff_method="backprop",
        )
