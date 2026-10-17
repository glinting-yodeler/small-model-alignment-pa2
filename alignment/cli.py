"""Thin command dispatch; heavy imports only in selected workflows."""
import argparse
from pathlib import Path
from .io import load_config, read_json, read_jsonl, write_json, append_jsonl


def main(command):
    parser = argparse.ArgumentParser(description=f"PA2 {command}")
    if command in ("prepare_data", "train_sft", "train_dpo", "train_persona", "generate"):
        parser.add_argument("--config", required=True)
        if command.startswith("train"):
            parser.add_argument("--resume")
        if command == "prepare_data":
            parser.add_argument("--pilot", action="store_true")
        args = parser.parse_args()
        config = load_config(args.config)
        if command == "prepare_data":
            from .preparation import prepare
            prepare(config, args.pilot)
        elif command == "generate":
            from .generation import generate
            generate(config)
        elif command == "train_persona":
            from .persona import run_persona
            run_persona(config, args.resume)
        else:
            from .training import run_training
            if (command == "train_sft") != (config["track"] == "sft"):
                raise ValueError("Command and track disagree")
            run_training(config, args.resume)
    elif command == "build_pairs":
        parser.add_argument("--baseline", required=True)
        parser.add_argument("--candidate", required=True)
        parser.add_argument("--track", required=True)
        parser.add_argument("--output", required=True)
        parser.add_argument("--count", type=int, default=300)
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args()
        from .pairwise import build_pairs
        pairs = build_pairs(read_json(args.baseline), read_json(args.candidate), args.track, args.seed, args.count)
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        for pair in pairs:
            append_jsonl(args.output, pair)
    elif command == "judge_pairwise_kaggle":
        parser.add_argument("--pairs-path", "--pairs_path", dest="pairs_path", required=True)
        parser.add_argument("--output-path", "--output_path", dest="output_path", required=True)
        parser.add_argument("--judge-model", "--judge_model", dest="judge_model", required=True)
        parser.add_argument("--rubric", choices=["general", "hh", "persona_style", "persona_quality"], default="general")
        parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=5)
        parser.add_argument("--sleep-seconds", "--sleep_seconds", dest="sleep_seconds", type=float, default=5)
        parser.add_argument("--max-attempts", "--max_retries", dest="max_attempts", type=int, default=3)
        parser.add_argument("--reuse-judgments", "--reuse_judgments", action="store_true",
                            help="Explicitly reuse compatible saved judgments (the default resume behavior).")
        args = parser.parse_args()
        from .judging import judge, kaggle_backend
        pairs = read_jsonl(args.pairs_path)
        judge(pairs, args.output_path, kaggle_backend(args.judge_model), args.judge_model,
              args.rubric, args.batch_size, args.sleep_seconds, args.max_attempts)
        from .pairwise import summarize
        attempts = Path(args.output_path).with_suffix(".attempts.jsonl")
        summary = summarize(pairs, read_jsonl(args.output_path), read_jsonl(attempts))
        for track, row in summary.items():
            rate = lambda value: "null" if value is None else f"{value:.3f}"
            print(f"{track}: W={row['W']} L={row['L']} T={row['T']} U={row['U']} F={row['F']} "
                  f"win={rate(row['raw_win_rate'])} tie={row['T']/row['N']:.3f} "
                  f"unknown={rate(row['unknown_rate'])} failure={rate(row['failure_rate'])}")
    elif command == "summarize_results":
        parser.add_argument("--pairs", required=True)
        parser.add_argument("--judgments", required=True)
        parser.add_argument("--output", required=True)
        args = parser.parse_args()
        from .pairwise import summarize
        write_json(args.output, summarize(read_jsonl(args.pairs), read_jsonl(args.judgments),
                   read_jsonl(Path(args.judgments).with_suffix(".attempts.jsonl"))))
    elif command == "plot_metrics":
        parser.add_argument("--metrics", nargs="+", required=True)
        parser.add_argument("--output", required=True)
        args = parser.parse_args()
        from .plotting import plot_metrics
        plot_metrics(args.metrics, args.output)
