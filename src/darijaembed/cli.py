"""A small CLI; importing it does not load PyTorch or download anything."""

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from darijaembed.config import load_config


def _prepare(config) -> dict:
    from darijaembed.data import load_source, prepare_pairs, write_jsonl

    if config.dataset.role != "train":
        raise ValueError("Data preparation requires a dataset with role='train'.")
    rows = load_source(config.dataset)
    train, validation, audit = prepare_pairs(rows, config.training.validation_fraction, config.seed)
    directory = Path(config.output_dir) / "data"
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "train.jsonl", train)
    write_jsonl(directory / "validation.jsonl", validation)
    audit["files_sha256"] = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("train.jsonl", "validation.jsonl")
    }
    (directory / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (directory / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"directory": str(directory), **audit}


def _doctor() -> dict:
    details = {"python": platform.python_version(), "packages": {}}
    for name in ("darijaembed", "torch", "sentence-transformers", "datasets", "accelerate"):
        try:
            details["packages"][name] = version(name)
        except PackageNotFoundError:
            details["packages"][name] = None
    try:
        import torch
    except ImportError:
        details["cuda_available"] = False
        details["hint"] = 'Install the ML extras: pip install -e ".[ml]"'
        return details
    details["cuda_available"] = torch.cuda.is_available()
    details["cuda_version"] = torch.version.cuda
    details["devices"] = [
        {"name": torch.cuda.get_device_name(i), "capability": torch.cuda.get_device_capability(i)}
        for i in range(torch.cuda.device_count())
    ]
    return details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DarijaEmbed reproducible embedding experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect Python, ML packages and CUDA")
    for command in ("prepare", "train", "evaluate"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--config", required=True, type=Path)
        if command == "train":
            subparser.add_argument("--prepared-dir", type=Path)
        if command == "evaluate":
            subparser.add_argument(
                "--model", help="Override the configured model with a model/path"
            )
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            report = _doctor()
        else:
            config = load_config(args.config)
            if args.command == "prepare":
                report = _prepare(config)
            elif args.command == "train":
                from darijaembed.training import train

                report = train(config, prepared_dir=args.prepared_dir)
            else:
                from darijaembed.evaluation import evaluate

                report = evaluate(config, model_name=args.model)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as error:
        print(f"darijaembed: {error}", file=sys.stderr)
        return 1
