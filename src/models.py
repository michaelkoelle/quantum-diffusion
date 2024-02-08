import torch
import einops
import typing
import warnings
import tqdm


class Diffusion(torch.nn.Module):
    """
    Diffusion model.
    Can be used for any torch.nn.Module as a network.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        noise_f,
        prediction_goal: str,
        shape: typing.Tuple[int, int],
        loss: torch.nn.Module = torch.nn.MSELoss(reduction="none"),
        directed=False,
        on_states=False,
        skip_name_test=False,
    ) -> None:
        super().__init__()
        self.net = net
        if not skip_name_test:
            assert (
                net._get_name().lower().__contains__("directed")
            ), f"Unknown network type {net._get_name()}"
            if directed and net._get_name().lower().__contains__("undirected"):
                raise ValueError(
                    f"Directed model cannot be used with undirected network {net._get_name()}"
                )
            if not directed and not net._get_name().lower().__contains__("undirected"):
                raise ValueError(
                    f"Undirected model cannot be used with directed network {net._get_name()}"
                )
        assert prediction_goal in [
            "data",
            "noise",
        ], "prediction_goal must be either 'data' or 'noise'"
        self.prediction_goal = prediction_goal
        self.add_noise = noise_f
        self.width, self.height = shape
        self.loss = loss
        self.directed = directed
        self.on_states = on_states
        if self.on_states:
            self.loss = StateLoss()

    def forward(
        self, x: typing.Union[torch.Tensor, None], **kwargs
    ) -> typing.Union[torch.Tensor, None]:
        """
        If training, executes training step. If not, samples from the model.
        """
        if self.training:
            x = typing.cast(torch.Tensor, x)
            if self.prediction_goal == "data":
                return self.run_training_step_data(x, **kwargs)
            else:
                return self.run_training_step_noise(x, **kwargs)
        else:
            return self.sample(first_x=x, **kwargs)

    def run_training_step_data(self, x: torch.Tensor, **kwargs) -> typing.Any:
        assert self.prediction_goal == "data", "prediction goal is not data"
        T = kwargs["T"]
        if self.directed:
            y = kwargs["y"]
            assert y.shape[0] == x.shape[0], "batch size of x and y must be the same"
            labels = einops.repeat(y, "b -> (b tau)", tau=T)
        else:
            if "y" in kwargs and kwargs["y"] is not None:
                warnings.warn("y is not used because the model is not directed")
        whole_noisy = self.add_noise(x, tau=T + 1, decay_mod=3.0)
        whole_noisy = einops.rearrange(
            whole_noisy, "(batch tau) pixels -> batch tau pixels", tau=T + 1
        )
        batches_noisy = whole_noisy[:, 1:, :]
        batches_clean = whole_noisy[:, :-1, :]
        batches_noisy = einops.rearrange(
            batches_noisy,
            "batch tau (width height) -> (batch tau) 1 width height",
            width=self.width,
            height=self.height,
        )
        batches_clean = einops.rearrange(
            batches_clean,
            "batch tau (width height) -> (batch tau) 1 width height",
            width=self.width,
            height=self.height,
        )
        if self.directed:
            batches_reconstructed = self.net.forward(x=batches_noisy, y=labels)  # type: ignore
        else:
            batches_reconstructed = self.net.forward(x=batches_noisy)
        batch_loss = self.loss(batches_reconstructed, batches_clean)
        batch_loss_mean = batch_loss.mean()
        batch_loss_mean.backward()
        verbose = kwargs.get("verbose", False)
        if verbose:
            return batch_loss.abs(), batches_reconstructed.abs()
        else:
            return (batch_loss_mean.abs(),)

    def run_training_step_noise(self, x: torch.Tensor, **kwargs) -> typing.Any:
        assert self.prediction_goal == "noise", "prediction goal is not noise"
        T = kwargs["T"]
        if self.directed:
            y = kwargs["y"]
            assert y.shape[0] == x.shape[0], "batch size of x and y must be the same"
            labels = einops.repeat(y, "b -> (b tau)", tau=T)
        whole_noisy = self.add_noise(x, tau=T + 1, decay_mod=3.0)
        whole_noisy = einops.rearrange(
            whole_noisy, "(batch tau) pixels -> batch tau pixels", tau=T + 1
        )
        batches_noisy = whole_noisy[:, 1:, :]
        batches_clean = whole_noisy[:, :-1, :]
        batches_noisy = einops.rearrange(
            batches_noisy,
            "batch tau (width height) -> (batch tau) 1 width height",
            width=self.width,
            height=self.height,
        )
        batches_clean = einops.rearrange(
            batches_clean,
            "batch tau (width height) -> (batch tau) 1 width height",
            width=self.width,
            height=self.height,
        )
        if self.directed:
            predicted_noise = self.net.forward(x=batches_noisy, y=labels)  # type: ignore
        else:
            predicted_noise = self.net.forward(x=batches_noisy)
        predicted_noise = (predicted_noise - 0.5) * 0.1
        real_noise = batches_noisy - batches_clean
        batch_loss = self.loss(predicted_noise, real_noise)
        batch_loss_mean = batch_loss.mean()
        batch_loss_mean.backward()
        verbose = kwargs.get("verbose", False)
        if verbose:
            return batch_loss, torch.clamp(batches_noisy - predicted_noise, 0, 1)
        else:
            return (batch_loss_mean,)

    def sample(
        self,
        n_iters,
        first_x: typing.Union[torch.Tensor, None] = None,
        labels: typing.Union[torch.Tensor, None] = None,
        show_progress: bool = False,
        only_last=False,
        step=1,
        noise_factor=1.0,
    ) -> torch.Tensor:
        """ " Samples from the model for n_iters iterations."""
        if first_x is None:
            first_x = torch.rand((10, 1, self.width, self.height))
        if self.on_states:
            return self._sample_on_states(n_iters, first_x, only_last, labels=labels)
        if labels is None and self.directed:
            labels = torch.zeros((first_x.shape[0], 1))
        if labels is not None and not self.directed:
            warnings.warn("labels are not used because the model is not directed")
        outp = [first_x]
        if show_progress:
            iters = tqdm.tqdm(range(n_iters))
        else:
            iters = range(n_iters)
        with torch.no_grad():
            x = first_x
            for i in iters:
                if self.directed:
                    predicted = self.net(x, labels)
                else:
                    predicted = self.net(x)
                if self.prediction_goal == "data":
                    x = predicted
                else:
                    predicted = (predicted - 0.5) * 0.1 * noise_factor
                    new_x = x - predicted
                    new_x = torch.clamp(new_x, 0, 1)
                    x = new_x
                if i % step == 0:
                    outp.append(x)
        if only_last:
            return outp[-1]
        else:
            outp = torch.stack(outp)
            outp = einops.rearrange(
                outp, "iters batch 1 height width -> (iters height) (batch width)"
            )
            return outp

    def _sample_on_states(
        self, n_iters: int, first_x: torch.Tensor, only_last=True, labels=None
    ) -> torch.Tensor:
        assert only_last, "can't sample intermediate states, set `only_last=True`"
        assert self.prediction_goal == "data", "can't sample noise"
        assert self.on_states, "use sample() instead"
        return self.net.sample(first_x, num_repeats=n_iters, labels=labels)  # type: ignore

    def get_variance_sample(self, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the sample and the variance over the iterations.
        """
        sample = self.sample(**kwargs).abs()
        sample = einops.rearrange(
            sample,
            "(iters height) (batch width) -> iters batch height width",
            height=self.height,
            width=self.width,
        )
        vars = sample.var(dim=1)
        sample = einops.rearrange(
            sample, "iters batch height width -> (iters height) (batch width)"
        )
        vars = einops.rearrange(vars, "iters height width -> (iters height) (width)")
        return sample, vars

    def save_name(self):
        return f"{self.net.save_name()}{'_noise' if self.prediction_goal == 'noise' else ''}"  # type: ignore


class StateLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert input.is_complex(), "input must be complex"
        assert not target.is_complex(), "target must be real"
        return (input.real - target) ** 2 + input.imag**2
