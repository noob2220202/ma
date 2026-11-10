from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import LipsyncError, weights


def _positive(convert):
    def parse(text: str):
        value = convert(text)
        if value <= 0:
            raise argparse.ArgumentTypeError(f"must be positive, not {text}")
        return value

    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lipsync", description="Animate the mouth of one portrait to match a speech recording.")
    parser.add_argument("--models", type=Path, default=weights.DEFAULT_DIR, help="directory holding the model weights (default: %(default)s)")
    commands = parser.add_subparsers(dest="command", required=True)

    render = commands.add_parser("render", help="render a video from an image and an audio file")
    render.add_argument("face", type=Path, help="portrait image containing a face")
    render.add_argument("audio", type=Path, help="speech recording in any format ffmpeg reads")
    render.add_argument("-o", "--out", type=Path, required=True, help="output .mp4 path")
    render.add_argument("--fps", type=_positive(float), default=25.0, help="output frame rate; the model was trained at 25 (default: %(default)s)")
    render.add_argument("--pads", type=int, nargs=4, default=(0, 10, 0, 0), metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
                        help="pixels to grow the detected face box by; raise BOTTOM if the chin is cut (default: 0 10 0 0)")
    render.add_argument("--batch-size", type=_positive(int), default=64, help="frames generated per model call (default: %(default)s)")

    verify = commands.add_parser("verify", help="check a rendered video against its inputs")
    verify.add_argument("video", type=Path)
    verify.add_argument("--face", type=Path, required=True)
    verify.add_argument("--audio", type=Path, required=True)

    commands.add_parser("fetch", help="download and checksum the model weights")
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.command == "fetch":
        for name in weights.WEIGHTS:
            print(weights.ensure(name, args.models))
        return 0

    if args.command == "render":
        from .pipeline import render

        render(args.face, args.audio, args.out, fps=args.fps, pads=tuple(args.pads), batch_size=args.batch_size, models_dir=args.models)
        print(args.out)
        return 0

    from .verify import verify

    checks = verify(args.video, args.face, args.audio, models_dir=args.models)
    for check in checks:
        status = {True: "ok", False: "FAIL", None: "skip"}[check.passed]
        print(f"{status:4} {check.name}: {check.detail}")
    return 1 if any(check.passed is False for check in checks) else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        return _run(args)
    except (LipsyncError, OSError) as error:
        print(f"lipsync: {error}", file=sys.stderr)
        return 1
