from typing import Tuple
import torch
import pennylane as qml
import math
import qw_map
import einops
import numpy as np
import warnings
import circuits


class QDenseUndirected(torch.nn.Module):
    """Dense variational circuit. Undirected"""

    def __init__(self, qdepth, shape) -> None:
        super().__init__()
        self.qdepth = qdepth
        if isinstance(shape, int):
            shape = (shape, shape)
        self.width, self.height = shape
        self.pixels = self.width * self.height
        self.wires = math.ceil(math.log2(self.width * self.height))
        self.qdev = qml.device("default.qubit.torch", wires=self.wires)
        weight_shape = qml.StronglyEntanglingLayers.shape(self.qdepth, self.wires)
        self.weights = torch.nn.Parameter(
            torch.randn(weight_shape, requires_grad=True) * 0.4
        )
        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="backprop",
        )

    def _circuit(self, inp):
        qml.AmplitudeEmbedding(
            features=inp, wires=range(self.wires), normalize=True, pad_with=0.1
        )
        qml.StronglyEntanglingLayers(
            weights=qw_map.tanh(self.weights), wires=range(self.wires)
        )
        return qml.probs(wires=range(self.wires))

    def _post_process(self, probs):
        # probs = probs[:, ::2] # Drop all probabilities for |xxxx1>
        probs = probs[:, : self.pixels]
        probs = probs * self.pixels
        probs = torch.clamp(probs, 0, 1)
        return probs

    def forward(self, x):
        x = einops.rearrange(x, "b 1 w h -> b (w h)")
        x = self.qnode(x)
        x = self._post_process(x)
        x = einops.rearrange(x, "b (w h) -> b 1 w h", w=self.width, h=self.height)
        return x

    def __repr__(self):
        return f"QDenseUndirected(qdepth={self.qdepth}, wires={self.wires})"

    def save_name(self) -> str:
        return f"qdense_undirected_d{self.qdepth}_w{self.width}_h{self.height}"


class QDense2Undirected(torch.nn.Module):
    """Dense variational circuit. Undirected. Uses ancilla qubit"""

    def __init__(
        self, qdepth, shape, entangling_layer=qml.StronglyEntanglingLayers
    ) -> None:
        super().__init__()
        self.qdepth = qdepth
        if isinstance(shape, int):
            shape = (shape, shape)
        self.width, self.height = shape
        self.pixels = self.width * self.height
        self.entangling_layer = entangling_layer
        self.wires = math.ceil(math.log2(self.width * self.height)) + 1
        self.qdev = qml.device("default.qubit.torch", wires=self.wires)
        weight_shape = self.entangling_layer.shape(self.qdepth, self.wires)
        self.weights = torch.nn.Parameter(
            torch.randn(weight_shape, requires_grad=True) * 0.4
        )
        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="backprop",
        )

    def _circuit(self, inp):
        qml.AmplitudeEmbedding(
            features=inp, wires=range(self.wires - 1), normalize=True, pad_with=0.1
        )
        self.entangling_layer(
            weights=qw_map.tanh(self.weights), wires=range(self.wires)
        )
        return qml.probs(wires=range(self.wires))

    def _post_process(self, probs):
        probs = probs[:, ::2]  # Drop all probabilities for |xxxx1>
        probs = probs[:, : self.pixels]
        probs = probs * self.pixels
        probs = torch.clamp(probs, 0, 1)
        return probs

    def forward(self, x):
        x = einops.rearrange(x, "b 1 w h -> b (w h)")
        x = self.qnode(x)
        x = self._post_process(x)
        x = einops.rearrange(x, "b (w h) -> b 1 w h", w=self.width, h=self.height)
        return x

    def __strongly(self):
        return "" if "strongly" in str(self.entangling_layer).lower() else "_weakly"

    def __repr__(self):
        return f"QDense2Undirected(qdepth={self.qdepth}, wires={self.wires}{self.__strongly()})"

    def save_name(self) -> str:
        return f"qdense2_undirected_d{self.qdepth}_w{self.width}_h{self.height}{self.__strongly()}"


class QDenseDirected(QDense2Undirected):
    """Dense variatonal circuit with label"""

    def _circuit(self, inp, label):
        qml.AmplitudeEmbedding(
            features=inp, wires=range(self.wires - 1), normalize=True, pad_with=0.1
        )
        qml.RX(phi=label, wires=self.wires - 1)
        qml.StronglyEntanglingLayers(
            weights=qw_map.tanh(self.weights), wires=range(self.wires)
        )
        return qml.probs(wires=range(self.wires))

    def forward(self, x, y):
        x = einops.rearrange(x, "b 1 w h -> b (w h)")
        x = self.qnode(x, y)
        x = self._post_process(x)
        x = einops.rearrange(x, "b (w h) -> b 1 w h", w=self.width, h=self.height)
        return x

    def __repr__(self):
        return f"QDenseDirected(qdepth={self.qdepth}, wires={self.wires})"

    def save_name(self) -> str:
        return f"qdense_directed_d{self.qdepth}_w{self.width}_h{self.height}"


class QDenseDirectedReupload(torch.nn.Module):
    def __init__(self, qdepth, shape, num_reuploads=2):
        super().__init__()
        self.qdepth = qdepth
        if isinstance(shape, int):
            shape = (shape, shape)
        self.width, self.height = shape
        self.pixels = self.width * self.height
        self.wires = math.ceil(math.log2(self.width * self.height)) + 1
        self.qdev = qml.device("default.qubit.torch", wires=self.wires)
        self.num_reuploads = num_reuploads

        self.weights = torch.nn.ParameterList()
        for i in np.array_split(np.arange(qdepth), num_reuploads):
            weight_shape = qml.StronglyEntanglingLayers.shape(len(i), self.wires)
            self.weights.append(
                torch.nn.Parameter(torch.randn(weight_shape, requires_grad=True) * 0.4)
            )
        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="backprop",
        )

    def _post_process(self, probs):
        probs = probs[:, ::2]  # Drop all probabilities for |xxxx1>
        probs = probs[:, : self.pixels]
        probs = probs * self.pixels
        probs = torch.clamp(probs, 0, 1)
        return probs

    def forward(self, x, y):
        x = einops.rearrange(x, "b 1 w h -> b (w h)")
        x = self.qnode(x, y)
        x = self._post_process(x)
        x = einops.rearrange(x, "b (w h) -> b 1 w h", w=self.width, h=self.height)
        return x

    def _circuit(self, inp, label):
        qml.AmplitudeEmbedding(
            features=inp, wires=range(self.wires - 1), normalize=True, pad_with=0.1
        )
        for w in self.weights:
            qml.RX(phi=label, wires=self.wires - 1)
            qml.StronglyEntanglingLayers(
                weights=qw_map.tanh(w), wires=range(self.wires)
            )
        return qml.probs(wires=range(self.wires))

    def __repr__(self):
        return f"QDenseDirectedReupload(qdepth={self.qdepth}, wires={self.wires}, num_reuploads={self.num_reuploads})"

    def save_name(self) -> str:
        return f"qdense_directed_reupload_d{self.qdepth}_w{self.width}_h{self.height}_r{self.num_reuploads}"


class QDense4StatesUndirected(torch.nn.Module):
    def __init__(
        self,
        qdepth: int,
        shape: Tuple[int, int],
        directed=False,
    ) -> None:
        super().__init__()
        assert not directed, "Directed not supported"
        if isinstance(shape, int):
            shape = (shape, shape)
        self.directed = directed
        self.qdepth = qdepth
        self.width, self.height = shape
        self.pixels = self.width * self.height
        self.wires = math.ceil(math.log2(self.width * self.height))
        self.qdev = qml.device("default.qubit.torch", wires=self.wires)
        weight_shape = qml.StronglyEntanglingLayers.shape(qdepth, self.wires)
        self.weights = torch.nn.Parameter(
            torch.randn(weight_shape, requires_grad=True) * 0.4
        )
        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="backprop",
        )

    def _circuit(self, inp: torch.Tensor, reps: int) -> torch.Tensor:
        qml.QubitStateVector(state=inp, wires=range(self.wires))
        for _ in range(reps):
            qml.StronglyEntanglingLayers(
                weights=qw_map.tanh(self.weights), wires=range(self.wires)
            )
        return qml.state()  # type: ignore

    def forward(self, x: torch.Tensor, reps=1, other_node=None) -> torch.Tensor:
        assert (
            x.ndim == 4
        ), f"Input must be 4D tensor (batch, channels, width, height), but is {x.shape}"
        x = einops.rearrange(x, "b 1 w h -> b (w h)")
        old_norm = torch.norm(x, dim=-1, keepdim=True)
        x = x / old_norm
        x = torch.complex(real=x, imag=torch.zeros_like(x)).to(x.device)
        if other_node is not None:
            x = other_node(x, reps=reps)
        else:
            x = self.qnode(x, reps=reps)
        x = x * old_norm
        x = einops.rearrange(x, "b (w h) -> b 1 w h", w=self.width, h=self.height)
        return x

    def __repr__(self) -> str:
        return (
            f"QDense4StatesUndirected(qdepth={self.qdepth}, wires={self.wires})"
        )

    def save_name(self) -> str:
        return f"_qdense_undirected_for_states_d{self.qdepth}_w{self.width}_h{self.height}"

    def sample(self, x: torch.Tensor, num_repeats: int, **kwargs) -> torch.Tensor:
        return self.forward(x, reps=num_repeats).abs().detach().cpu()

    def sample_on_device(
        self, x: torch.Tensor, num_repeats: int, device: qml.Device = None
    ) -> torch.Tensor:
        assert (
            x.dim() == 2
        ), f"Input must be 2D tensor (batch, pixels), but is {x.shape}"

        def _sampling_circuit(inp, reps):
            # qml.QubitStateVector(state=inp, wires=range(self.wires))
            qml.AmplitudeEmbedding(
                features=inp, wires=range(self.wires), normalize=True
            )
            qml.StronglyEntanglingLayers(
                weights=qw_map.tanh(self.weights), wires=range(self.wires)
            )
            qml.StronglyEntanglingLayers(
                weights=qw_map.tanh(self.weights), wires=range(self.wires)
            )
            qml.StronglyEntanglingLayers(
                weights=qw_map.tanh(self.weights), wires=range(self.wires)
            )
            qml.StronglyEntanglingLayers(
                weights=qw_map.tanh(self.weights), wires=range(self.wires)
            )
            qml.StronglyEntanglingLayers(
                weights=qw_map.tanh(self.weights), wires=range(self.wires)
            )
            qml.StronglyEntanglingLayers(
                weights=qw_map.tanh(self.weights), wires=range(self.wires)
            )
            qml.StronglyEntanglingLayers(
                weights=qw_map.tanh(self.weights), wires=range(self.wires)
            )
            # qml.StronglyEntanglingLayers(weights=qw_map.tanh(self.weights), wires=range(self.wires))
            return qml.probs(wires=range(self.wires))

        if device is None:
            device = self.qdev
        sampling_qnode = qml.QNode(
            func=_sampling_circuit, device=device, interface="torch", diff_method="best"
        )
        with torch.no_grad():
            result = sampling_qnode(inp=x, reps=num_repeats)
        return result.abs().detach().cpu()


class QDense4StatesAncilla(torch.nn.Module):
    """Dense variational circuit, training on quantum states, directed or undirected, with reuploads"""

    def __init__(
        self, qdepth: int, shape: Tuple[int, int], directed=True, num_reuploads=1
    ) -> None:
        super().__init__()
        self.directed = directed
        self.qdepth = qdepth
        if isinstance(shape, int):
            shape = (shape, shape)
        self.width, self.height = shape
        self.pixels = self.width * self.height
        self.wires = math.ceil(math.log2(self.width * self.height)) + 1
        self.qdev = qml.device("default.qubit.torch", wires=self.wires)
        self.num_reuploads = num_reuploads
        self.weights = torch.nn.ParameterList()
        for i in np.array_split(np.arange(qdepth), num_reuploads):
            weight_shape = qml.StronglyEntanglingLayers.shape(len(i), self.wires)
            self.weights.append(
                torch.nn.Parameter(torch.randn(weight_shape, requires_grad=True) * 0.4)
            )
        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="backprop",
        )
        assert shape[0] * shape[1] == 2 ** (self.wires - 1), (
            "Shape must be compatible with number of wires. "
            f"{shape[0]}*{shape[1]} != 2**({self.wires}-1). "
            "Up/Downscale the image to 8x8, 32x32 or similar. "
        )

    def _circuit(self, inp: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        qml.QubitStateVector(state=inp, wires=range(self.wires - 1))
        for w in self.weights:
            qml.RX(phi=label, wires=self.wires - 1)
            qml.StronglyEntanglingLayers(
                weights=qw_map.tanh(w), wires=range(self.wires)
            )
        return qml.state()  # type: ignore

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        assert (
            x.ndim == 4
        ), f"Input must be 4D tensor (batch, channels, width, height), but is {x.shape}"
        x = einops.rearrange(x, "b 1 w h -> b (w h)")
        old_norm = torch.norm(x, dim=-1, keepdim=True)
        x = x / old_norm
        x = torch.complex(real=x, imag=torch.zeros_like(x)).to(x.device)
        if y is None:
            if self.directed:
                warnings.warn("Model is directed, but no label was provided.")
            y = torch.zeros(x.shape[0]).to(x.device)
        else:
            if not self.directed:
                warnings.warn("Model is undirected, but a label was provided.")
        x = self.qnode(x, y)
        x = x[..., ::2]
        x = x * old_norm
        x = einops.rearrange(x, "b (w h) -> b 1 w h", w=self.width, h=self.height)
        return x

    def __reupload_str(self) -> str:
        return "" if self.num_reuploads == 1 else f"_reup{self.num_reuploads}"

    def sample(self, first_x, num_repeats=10, labels=None):
        first_x = einops.rearrange(first_x, "b 1 w h -> b (w h)")
        first_x /= torch.linalg.norm(first_x, dim=-1, keepdim=True)
        cf = circuits.CircuitFactory(self.wires)
        qn = cf.sampling_qnode_with_swap(num_repeats=num_repeats, has_reuploads=True)
        with torch.no_grad():
            sample = qn(first_x, self.weights, labels)
            sample = sample.abs().detach().cpu()
        sample = einops.rearrange(
            sample, "b (w h drop) -> drop b 1 w h", w=self.width, h=self.height
        )[0]
        sample /= einops.reduce(sample, "b 1 w h -> b 1 () ()", reduction="max")
        return sample

    def __repr__(self):
        if self.directed:
            return f"QDenseDirected4States(qdepth={self.qdepth}, wires={self.wires} {self.__reupload_str()})"
        else:
            return f"QDenseUndirected4StatesAncilla(qdepth={self.qdepth}, wires={self.wires} {self.__reupload_str()})"

    def save_name(self) -> str:
        if self.directed:
            return f"_qdense_directed_for_states_d{self.qdepth}_w{self.width}_h{self.height}{self.__reupload_str()}"
        else:
            return f"_qdense_undirected_for_states_ancilla_d{self.qdepth}_w{self.width}_h{self.height}{self.__reupload_str()}"
