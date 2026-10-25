"""MLPilot — agentic AutoML pipeline.

Usage:
    python main.py <dataset.csv> --target <column_name>
"""
import argparse
import logging

from graph import build_graph


def main():
    parser = argparse.ArgumentParser(description="MLPilot — agentic AutoML pipeline")
    parser.add_argument("file_path", help="Path to the CSV dataset")
    parser.add_argument("--target", required=True, help="Name of the target column")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("mlpilot")

    logger.info("Building MLPilot graph...")
    app = build_graph()

    inputs = {
        "file_path": args.file_path,
        "target_column": args.target,
    }

    logger.info("Pipeline starting...")
    # recursion_limit = seatbelt: if a loop guard ever failed,
    # LangGraph raises GraphRecursionError instead of hanging forever
    result = app.invoke(inputs, config={"recursion_limit": 50})

    print("\n" + result["report"])
    logger.info("Done. Best model: %s", result.get("best_model_name"))
    
    # show the LLM insight (generated on the finish branch)
    insight_text = result.get("insight")
    if insight_text:
        print("\n" + "=" * 60)
        print("MLPILOT INSIGHT (LLM)")
        print("=" * 60)
        print(insight_text, "\n")


if __name__ == "__main__":
    main()