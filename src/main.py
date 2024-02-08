from turtle import width
import matplotlib.pyplot as plt
import torch
import einops
import data
import noise
import models
import nn
import argparse
import sys
import inspect
import warnings
import tqdm
import pathlib
import webbrowser

all_nn = [name for name, obj in inspect.getmembers(nn) if inspect.isclass(obj)]
all_ds = [
    name
    for name, obj in inspect.getmembers(data)
    if inspect.isfunction(obj) and not name.startswith("_")
]


def parse_args(args):
    parser = argparse.ArgumentParser(description="Quantum Denoising Diffusion Model")
    parser.add_argument(
        "--data",
        type=str,
        default="mnist_8x8",
        help=f"Dataset to use. Available datasets: {', '.join(all_ds)}.",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=2,
        help="Number of label classes to use. \
                        Smaller models perform better on a smaller number of classes.",
    )
    parser.add_argument(
        "--target", type=str, default="noise", help="Generate noise or data."
    )
    parser.add_argument(
        "--save-path", type=str, default="results", help="Path to save results."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Load model from path. \
            If no path is given, train a new model.\
            The trained model will be saved in --save-path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=["QDenseUndirected", "55", "8"],
        nargs="+",
        help=f"Model name and parameters. \
            Models are defined in the nn module, including {', '.join(all_nn)}.",
    )
    parser.add_argument(
        "--guidance", type=bool, default=False, help="Toggle guidance. Default: False"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use.",
    )
    parser.add_argument(
        "--tau",
        type=int,
        default=10,
        help="Number of iterations. \
            Models perform better with more iterations on higher resolution images, \
            for low-res, tau=10 suffices.",
    )
    parser.add_argument(
        "--ds-size",
        type=int,
        default=100,
        help="Dataset size. 80%% is used for training.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs.")
    return parser.parse_args(args)


def train():
    print("Training model")
    diff.train()
    pbar = tqdm.tqdm(total=args.epochs)
    opt = torch.optim.Adam(diff.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for x, y in ds:
            opt.zero_grad()
            batch_loss, _ = diff(x=x, y=y, T=args.tau, verbose=True)
            epoch_loss += batch_loss.mean()
            opt.step()
        pbar.set_postfix({"loss": epoch_loss.item()})  # type: ignore
        pbar.update(1)
    pbar.close()
    sp = pathlib.Path(args.save_path) / f"{diff.save_name()}.pt"
    if not sp.parent.exists():
        sp.parent.mkdir(parents=True)
    torch.save(diff.state_dict(), sp)


def test():
    print("Testing model")
    diff.eval()
    first_x = torch.rand(15, 1, 8, 8) * 0.5 + 0.75
    outp = diff.sample(first_x=first_x, n_iters=args.tau * 2, show_progress=True)
    plt.imshow(outp.cpu(), cmap="gray")
    plt.axis("off")
    sp = pathlib.Path(args.save_path) / f"{diff.save_name()}.png"
    plt.savefig(sp)
    webbrowser.open(sp.absolute().as_uri())


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.device == "cuda":
        warnings.warn("CUDA performance is worse than CPU for most models.")
        if not torch.cuda.is_available():
            warnings.warn("CUDA is not available, using CPU.")
            args.device = "cpu"

    x_train, y_train, height, width = eval(f"data.{args.data}")(
        n_classes=args.n_classes, ds_size=args.ds_size
    )
    x_train = x_train.to(args.device)
    y_train = y_train.to(args.device)
    train_cutoff = int(len(x_train) * 0.8)
    x_train, x_test = x_train[:train_cutoff], x_train[train_cutoff:]
    y_train, y_test = y_train[:train_cutoff], y_train[train_cutoff:]
    ds = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=10,
        shuffle=False,
    )

    net = eval(f"nn.{args.model[0]}")(*[int(a) for a in args.model[1:]])
    diff = models.Diffusion(
        net=net,
        shape=(height, width),
        noise_f=noise.add_normal_noise_multiple,
        prediction_goal=args.target,
        directed=args.guidance,
        loss=torch.nn.MSELoss(),
    ).to(args.device)

    run_train = False
    if args.load_path is not None:
        print("Loading model")
        try:
            if args.load_path.endswith(".pt"):
                diff.load_state_dict(torch.load(args.load_path))
            else:
                lp = pathlib.Path(args.load_path) / f"{diff.save_name()}.pt"
                diff.load_state_dict(torch.load(lp))
        except FileNotFoundError:
            print("Failed to load model")
            run_train = True

    if args.load_path is None or run_train:
        train()

    test()
