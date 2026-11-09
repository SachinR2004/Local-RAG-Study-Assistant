
import csv
import json
import sys
import time

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_utils import RAGIndex
from rag_pipeline import RAGPipeline
from config import TOP_K


DATASET_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "answer_results.csv"
VECTORSTORE_PATH = PROJECT_ROOT / "data" / "vectorstore"

REQUEST_PAUSE = 30


FIELDNAMES = [
    "id",
    "question",
    "validation_status",
    "supported",
    "attempts",
    "issue_count",
    "missing_aspect_count",
    "unsupported_claim_count",
    "answer",
    "validation_summary",
    "issues",
    "missing_aspects",
    "unsupported_claims",
    "error"
]


def save_results(results):

    with open(
        RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()
        writer.writerows(results)


def load_existing_results():

    if not RESULTS_PATH.exists():
        return {}

    with open(
        RESULTS_PATH,
        newline="",
        encoding="utf-8"
    ) as file:

        return {
            row["id"]: row
            for row in csv.DictReader(file)
        }


def main():

    print("=" * 75)
    print("RAG ANSWER QUALITY EVALUATION")
    print("=" * 75)

    with open(DATASET_PATH, encoding="utf-8") as file:
        dataset = json.load(file)

    existing = load_existing_results()

    # Preserve successful results and retry everything else.
    results = dict(existing)

    pending = [
        item for item in dataset
        if existing.get(item["id"], {}).get(
            "validation_status"
        ) != "passed"
    ]

    print("Total questions:", len(dataset))
    print("Already passed:", len(dataset) - len(pending))
    print("Questions remaining:", len(pending))
    print("Pause between questions:", REQUEST_PAUSE, "seconds")

    if not pending:
        print("\nAll questions have already passed.")
        return

    rag = RAGIndex()
    rag.load(str(VECTORSTORE_PATH))

    pipeline = RAGPipeline(rag, top_k=TOP_K)

    for position, item in enumerate(pending, start=1):

        if position > 1:
            print(f"\nWaiting {REQUEST_PAUSE} seconds...")
            time.sleep(REQUEST_PAUSE)

        print("\n" + "-" * 75)
        print(item["id"], ":", item["question"])

        try:

            result = pipeline.ask(item["question"])

            report = result.get("validation_report", {})

            issues = report.get("issues", [])
            missing = report.get("missing_aspects", [])
            unsupported = report.get("unsupported_claims", [])

            row = {
                "id": item["id"],
                "question": item["question"],
                "validation_status": result["validation_status"],
                "supported": report.get("supported", False),
                "attempts": result["attempts"],
                "issue_count": len(issues),
                "missing_aspect_count": len(missing),
                "unsupported_claim_count": len(unsupported),
                "answer": result["answer"],
                "validation_summary": report.get("summary", ""),
                "issues": json.dumps(issues),
                "missing_aspects": json.dumps(missing),
                "unsupported_claims": json.dumps(unsupported),
                "error": result["error"]
            }

            results[item["id"]] = row

            print("Validation:", row["validation_status"])
            print("Supported:", row["supported"])
            print("Attempts:", row["attempts"])
            print("Issues:", issues)
            print("Missing aspects:", missing)
            print("Unsupported claims:", unsupported)

        except Exception as exc:

            print("Evaluation error:", str(exc))

            results[item["id"]] = {
                "id": item["id"],
                "question": item["question"],
                "validation_status": "evaluation_error",
                "supported": False,
                "attempts": 0,
                "issue_count": 0,
                "missing_aspect_count": 0,
                "unsupported_claim_count": 0,
                "answer": "",
                "validation_summary": "",
                "issues": "[]",
                "missing_aspects": "[]",
                "unsupported_claims": "[]",
                "error": str(exc)
            }

        # Save immediately, even if a later question fails.
        ordered_results = [
            results[item["id"]]
            for item in dataset
            if item["id"] in results
        ]

        save_results(ordered_results)

    # Final summary.
    ordered_results = [
        results[item["id"]]
        for item in dataset
        if item["id"] in results
    ]

    total = len(ordered_results)

    passed = sum(
        row["validation_status"] == "passed"
        for row in ordered_results
    )

    failed = sum(
        row["validation_status"] == "failed"
        for row in ordered_results
    )

    review = sum(
        row["validation_status"] == "review_required"
        for row in ordered_results
    )

    errors = sum(
        row["validation_status"] == "evaluation_error"
        for row in ordered_results
    )

    print("\n" + "=" * 75)
    print("AGGREGATE RESULTS")
    print("=" * 75)

    print("Completed records:", total)
    print("Passed:", passed)
    print("Failed:", failed)
    print("Review required:", review)
    print("Evaluation errors:", errors)

    print(
        "Evidence pass rate:",
        f"{passed / total:.3f}" if total else "N/A"
    )

    print("\nResults saved to:", RESULTS_PATH)


if __name__ == "__main__":
    main()
