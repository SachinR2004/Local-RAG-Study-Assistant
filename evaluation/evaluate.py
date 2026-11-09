
import csv
import json
import sys

from pathlib import Path
from statistics import mean

# Ensure project root is available when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_utils import RAGIndex
from config import TOP_K


DATASET_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results.csv"
VECTORSTORE_PATH = PROJECT_ROOT / "data" / "vectorstore"


def normalize(value):
    """Normalize metadata for consistent comparisons."""
    if value is None:
        return ""
    return str(value).strip().lower()


def evaluate_question(rag, item):

    question = item["question"]

    expected_activities = {
        normalize(activity)
        for activity in item["expected_activities"]
    }

    expected_pages = {
        int(page)
        for page in item.get("expected_pages", [])
    }

    chunks = rag.query_with_metadata(
        question,
        top_k=TOP_K
    )

    retrieved_activities = [
        normalize(chunk.get("activity"))
        for chunk in chunks
    ]

    retrieved_pages = set()

    for chunk in chunks:
        page = chunk.get("page")

        if page is not None:
            try:
                retrieved_pages.add(int(page))
            except (TypeError, ValueError):
                pass

    # Activity Recall@K:
    # Fraction of expected activities represented in retrieved chunks.
    activities_found = (
        expected_activities.intersection(retrieved_activities)
    )

    activity_recall = (
        len(activities_found) / len(expected_activities)
        if expected_activities else 0.0
    )

    # Activity Precision@K:
    # Fraction of retrieved chunks belonging to expected activities.
    relevant_chunks = sum(
        1 for activity in retrieved_activities
        if activity in expected_activities
    )

    activity_precision = (
        relevant_chunks / len(chunks)
        if chunks else 0.0
    )

    # Mean Reciprocal Rank:
    # Position of the first chunk from an expected activity.
    reciprocal_rank = 0.0

    for rank, activity in enumerate(retrieved_activities, start=1):
        if activity in expected_activities:
            reciprocal_rank = 1.0 / rank
            break

    # Page coverage is evaluated only when expected pages are provided.
    page_coverage = None

    if expected_pages:
        page_coverage = (
            len(expected_pages.intersection(retrieved_pages))
            / len(expected_pages)
        )

    return {
        "id": item["id"],
        "question": question,
        "activity_precision": activity_precision,
        "activity_recall": activity_recall,
        "mrr": reciprocal_rank,
        "page_coverage": page_coverage,
        "retrieved_chunks": len(chunks),
        "expected_activities": ", ".join(
            sorted(expected_activities)
        ),
        "retrieved_activities": ", ".join(
            retrieved_activities
        ),
        "retrieved_pages": ", ".join(
            str(page) for page in sorted(retrieved_pages)
        )
    }


def main():

    print("=" * 75)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 75)

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    print("Test questions:", len(dataset))
    print("Retrieval cutoff (K):", TOP_K)

    rag = RAGIndex()
    rag.load(str(VECTORSTORE_PATH))

    results = []

    for item in dataset:

        result = evaluate_question(rag, item)
        results.append(result)

        print("\n" + "-" * 75)
        print(f"{result['id']}: {result['question']}")
        print("Expected activities:", result["expected_activities"])
        print("Retrieved activities:", result["retrieved_activities"])
        print("Retrieved pages:", result["retrieved_pages"])
        print("Retrieved chunks:", result["retrieved_chunks"])

        print(
            f"Activity Precision@{TOP_K}: "
            f"{result['activity_precision']:.3f}"
        )

        print(
            f"Activity Recall@{TOP_K}: "
            f"{result['activity_recall']:.3f}"
        )

        print(f"MRR: {result['mrr']:.3f}")

        if result["page_coverage"] is not None:
            print(
                f"Page Coverage: "
                f"{result['page_coverage']:.3f}"
            )
        else:
            print("Page Coverage: N/A")

    # Aggregate metrics.
    print("\n" + "=" * 75)
    print("AGGREGATE RESULTS")
    print("=" * 75)

    print(
        f"Mean Activity Precision@{TOP_K}: "
        f"{mean(r['activity_precision'] for r in results):.3f}"
    )

    print(
        f"Mean Activity Recall@{TOP_K}: "
        f"{mean(r['activity_recall'] for r in results):.3f}"
    )

    print(
        f"Mean Reciprocal Rank: "
        f"{mean(r['mrr'] for r in results):.3f}"
    )

    page_scores = [
        r["page_coverage"]
        for r in results
        if r["page_coverage"] is not None
    ]

    if page_scores:
        print(
            f"Mean Page Coverage: "
            f"{mean(page_scores):.3f}"
        )

    # Export per-question results.
    fieldnames = list(results[0].keys())

    with open(
        RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nDetailed results saved to:")
    print(RESULTS_PATH)

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
