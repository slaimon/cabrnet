import os.path
from argparse import ArgumentParser, Namespace
from pathlib import Path

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.exceptions import ArgumentError
from cabrnet.core.utils.load_3d import VideoClip
from cabrnet.core.utils.save import safe_copy
from cabrnet.core.visualization.visualizer import SimilarityVisualizer
from cabrnet.core.visualization.visualizer3d import SimilarityVisualizer3D

description = "explains the decision of a 3D CaBRNet model"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    r"""Creates the argument parser for explaining the decision of a 3D CaBRNet model.

    Args:
        parser (ArgumentParser, optional): Parent parser (if any).
            Default: None

    Returns:
        The parser itself.
    """
    if parser is None:
        parser = ArgumentParser(description)
    parser = CaBRNet.create_parser(parser)
    # Relies on dataset configuration of the test to deduce the type of preprocessing
    # that needs to be applied on the source image
    parser = DatasetManager.create_parser(parser)
    parser = SimilarityVisualizer.create_parser(parser, mandatory_config=True)
    parser.add_argument(
        "-c",
        "--checkpoint-dir",
        type=str,
        required=False,
        metavar="/path/to/checkpoint/dir",
        help="path to a checkpoint directory (alternative to --model-arch, --model-state-dict, --dataset)",
    )
    parser.add_argument(
        "--sample-path",
        type=str,
        required=True,
        metavar="path/to/video",
        help="path to video to be explained",
    )
    parser.add_argument(
        "--duration",
        type=int,
        required=False,
        default=1,
        metavar="duration in seconds",
        help="length of the video clip to be explained, in seconds",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        required=False,
        default=9,
        metavar="number of frames",
        help="length of the video clip to be explained, in frames",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=False,
        default=180,
        metavar="height of frames",
        help="height of each frame in the video clip",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        metavar="path/to/output/directory",
        help="path to output directory",
    )
    parser.add_argument(
        "-y",
        "--prototype-dir",
        type=str,
        required=True,
        metavar="path/to/prototype/directory",
        help="path to directory containing prototype visualizations",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="pdf",
        required=False,
        metavar="extension",
        help="output file format",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing explanation (if any)",
    )
    return parser


def check_args(args: Namespace) -> Namespace:
    r"""Checks the validity of the arguments and updates the namespace if necessary.

    Args:
        args (Namespace): Parsed arguments.

    Returns:
        Modified argument namespace.
    """
    if args.checkpoint_dir is not None:
        # Fetch all files from directory
        for param, name in zip(
            [args.model_arch, args.model_state_dict, args.dataset],
            ["--model-arch", "--model-state-dict", "--dataset"],
        ):
            if param is not None:
                raise ArgumentError(f"Cannot specify both options {name} and --checkpoint-dir")
        args.model_arch = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_MODEL_CONFIG)
        args.model_state_dict = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_MODEL_STATE)
        args.dataset = os.path.join(args.checkpoint_dir, DatasetManager.DEFAULT_DATASET_CONFIG)

    # Check configuration completeness
    for param, name, option in zip(
        [args.model_arch, args.model_state_dict, args.dataset],
        ["model configuration", "state dictionary", "dataset configuration"],
        ["-m", "-s", "-d"],
    ):
        if param is None:
            raise ArgumentError(f"Missing {name} file (option {option}).")

    output_dir = os.path.join(args.output_dir, Path(args.sample_path).stem)
    if os.path.exists(output_dir) and not args.overwrite:
        raise ArgumentError(
            f"Output directory {output_dir} is not empty. " f"To overwrite existing results, use --overwrite option."
        )

    if not os.path.exists(args.prototype_dir):
        raise ArgumentError(
            "Prototype directory does not exist. "
            "Please use cabrnet explain_global first to generate prototype visualizations."
        )
    return args


def execute(args: Namespace) -> None:
    r"""Explains the decision of a CaBRNet model.

    Args:
        args (Namespace): Parsed arguments.

    """
    # Check and post-process options
    args = check_args(args)

    # Build model and load state dictionary
    model: CaBRNet = CaBRNet.build_from_config(config=args.model_arch, state_dict_path=args.model_state_dict)

    # Init visualizer
    visualizer = SimilarityVisualizer3D.build_from_config(config=args.visualization, model=model)

    # Recover preprocessing function
    preprocess = DatasetManager.get_dataset_transform(config=args.dataset, dataset="test_set")

    # Dedicated directory for target video
    output_dir = os.path.join(args.output_dir, Path(args.sample_path).stem)

    sample = VideoClip(
        path=args.sample_path,
        start_sec=0,
        duration_sec=args.duration,
        num_frames=args.num_frames,
        height=args.height
    )

    # Generate explanation
    model.explain(
        sample=sample,
        preprocess=preprocess,
        visualizer=visualizer,
        prototype_dir=args.prototype_dir,
        output_dir=output_dir,
        output_format=args.format,
        device=args.device,
        exist_ok=args.overwrite,
    )

    # Save visualization config
    safe_copy(
        args.visualization,
        os.path.join(output_dir, SimilarityVisualizer.DEFAULT_VISUALIZATION_CONFIG),
    )
