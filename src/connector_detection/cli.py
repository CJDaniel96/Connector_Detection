"""Console script entry points (installed via pyproject.toml [project.scripts])."""

from connector_detection.commands.normalize import app as _normalize_app
from connector_detection.commands.train import app as _train_app
from connector_detection.commands.validate import app as _validate_app


def normalize_orientation_cli() -> None:
    _normalize_app()


def train_patchcore_cli() -> None:
    _train_app()


def validate_patchcore_cli() -> None:
    _validate_app()
