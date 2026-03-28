import argparse
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen


DEPLOY_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = DEPLOY_DIR.parent

for path in (DEPLOY_DIR, REPO_ROOT_DEFAULT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from .app_function import load_model, predict_to_file
except ImportError:
    from app_function import load_model, predict_to_file


def _get_env_path(name: str, default):
    value = os.environ.get(name, str(default))
    value = value.strip().strip("\"'")
    return Path(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test deploy inference with a local image path or an online image URL."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(_get_env_path("MODEL_PATH", "/content/kneeSeg/RMKV_kneeSeg.pth")),
        help="Path to the model checkpoint (.pth). Defaults to MODEL_PATH env if set.",
    )
    parser.add_argument(
        "--repo_root",
        type=str,
        default=str(_get_env_path("REPO_ROOT", REPO_ROOT_DEFAULT)),
        help="Path to the repository root. Defaults to REPO_ROOT env if set.",
    )
    parser.add_argument(
        "--output_path",
        "--output_link",
        dest="output_path",
        type=str,
        required=True,
        help="Where to save the output mask image.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("DEVICE", "auto"),
        help="auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ.get("THRESHOLD", "0.5")),
        help="Binary threshold for sigmoid output.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for downloading online images.",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_local",
        type=str,
        help="Local image path on the machine.",
    )
    input_group.add_argument(
        "--input_online",
        type=str,
        help="Online image URL to download and run inference on.",
    )
    return parser.parse_args()


def load_input_image(args):
    if args.input_local:
        input_path = Path(args.input_local).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Local input image not found: {input_path}")
        return input_path, str(input_path)

    request = Request(
        args.input_online,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urlopen(request, timeout=args.timeout) as response:
        return response.read(), args.input_online


def main():
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not repo_root.is_dir():
        raise FileNotFoundError(f"Repository root not found: {repo_root}")

    runtime = load_model(
        weight_path=model_path,
        repo_root=repo_root,
        device=args.device,
        threshold=args.threshold,
    )
    image_input, input_source = load_input_image(args)
    saved_path = predict_to_file(
        runtime=runtime,
        image_input=image_input,
        output_path=output_path,
        threshold=args.threshold,
    )

    if not Path(saved_path).is_file():
        raise RuntimeError(f"Output mask was not created: {saved_path}")

    print("=" * 72)
    print("Deploy Test Summary")
    print("=" * 72)
    print(f"Model      : {model_path}")
    print(f"Input      : {input_source}")
    print(f"Output     : {saved_path}")
    print(f"Device     : {runtime['device']}")
    print(f"Threshold  : {args.threshold}")


if __name__ == "__main__":
    main()
