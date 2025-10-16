import argparse
import json

from .thinking_pipeline import ThinkingPipeline


def run_pipeline(prompt: str, verbose: bool = False) -> dict:
	pipeline = ThinkingPipeline(verbose=verbose)
	return pipeline(prompt)


def main() -> None:
	parser = argparse.ArgumentParser(description="Run the Thinking Without Thinking pipeline")
	parser.add_argument("prompt", help="User prompt for the pipeline to solve")
	parser.add_argument(
		"-v",
		"--verbose",
		action="store_true",
		help="Enable verbose logging during pipeline execution",
	)
	args = parser.parse_args()

	result = run_pipeline(args.prompt, verbose=args.verbose)
	print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()