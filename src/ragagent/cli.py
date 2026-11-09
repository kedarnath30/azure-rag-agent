"""Command line: `ragagent ask|agent|eval`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragagent.config import Settings
from ragagent.evaluate import evaluate, load_golden
from ragagent.factory import build_agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragagent", description="HR policy RAG + agent")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="answer a question with citations")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)
    agent = sub.add_parser("agent", help="run the tool-using agent")
    agent.add_argument("question")
    ev = sub.add_parser("eval", help="run the golden-set evaluation")
    ev.add_argument("--golden", type=Path, default=Path("data/eval/golden.jsonl"))
    ev.add_argument("--out", type=Path, default=None, help="write results JSON here")
    ev.add_argument("--min-hit-at-5", type=float, default=0.0, help="fail below this score")
    args = parser.parse_args(argv)

    built = build_agent(Settings.from_env())

    if args.command == "ask":
        result = built.pipeline.ask(args.question, args.top_k)
        print(result.answer)
        for c in result.citations:
            print(f"  [{c.number}] {c.title} > {c.section} (score {c.score})")
    elif args.command == "agent":
        out = built.run(args.question)
        for step in out.steps:
            print(f"-> {step.call.name}({json.dumps(step.call.arguments)})")
        print(out.answer)
    else:
        report = evaluate(built.pipeline, load_golden(args.golden))
        print(json.dumps(report["metrics"], indent=2))
        print(f"{report['n_questions']} questions, {len(report['failures'])} imperfect")
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if report["metrics"]["hit@5"] < args.min_hit_at_5:
            print(f"hit@5 below threshold {args.min_hit_at_5}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
