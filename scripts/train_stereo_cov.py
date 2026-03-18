"""Starter entrypoint for stereo self-supervised covariance training.

This script is intentionally thin: the training logic will live in the package
under ``uncertainty_estimation.training``. Keeping the CLI here makes it easy
to evolve the internal training code without turning package modules into
monolithic scripts.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stereo covariance predictor")
    parser.add_argument("--config", type=str, default="configs/training/stereo_ssl.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "Training runner not implemented yet. Next step is wiring this CLI to "
        "the new uncertainty_estimation.training stereo trainer."
    )


if __name__ == "__main__":
    main()
