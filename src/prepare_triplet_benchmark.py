import argparse
import json
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

from generate_dataset import ROOT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert generated negation data into a triplet-style retrieval benchmark."
    )
    parser.add_argument(
        "--input-path",
        default=str(ROOT_DIR / "outputs" / "msmarco_negative_rag_random50.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        default=str(ROOT_DIR / "outputs" / "triplet_benchmark.jsonl"),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Maximum number of rows to export.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--extra-docs-per-query",
        type=int,
        default=18,
        help="Number of random background documents to add to each query corpus.",
    )
    return parser.parse_args()


def load_rows(input_path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "generated" not in row:
            continue
        generated = row["generated"]
        if not generated.get("query_negative"):
            continue
        if not generated.get("original_passage"):
            continue
        if not generated.get("evidence"):
            continue
        rows.append(row)
    return rows


def build_background_pool(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    pool = []
    seen_ids = set()

    for row in rows:
        generated = row["generated"]
        candidates = [
            (f"{row['query_id']}_gold", generated["original_passage"], "gold"),
            (f"{row['query_id']}_dist", generated["evidence"], "distractor"),
        ]
        for doc_id, text, source in candidates:
            normalized = text.strip()
            if not normalized:
                continue
            key = (doc_id, normalized)
            if key in seen_ids:
                continue
            seen_ids.add(key)
            pool.append({"doc_id": doc_id, "text": normalized, "source": source})

    return pool


def build_triplet_record(
    row: dict[str, Any],
    pool: list[dict[str, str]],
    extra_docs_per_query: int,
    rng: random.Random,
) -> dict[str, Any]:
    generated = row["generated"]
    query_id = str(row["query_id"])
    gold_text = generated["original_passage"].strip()
    distractor_text = generated["evidence"].strip()

    negatives = [
        doc
        for doc in pool
        if doc["doc_id"] not in {f"{query_id}_gold", f"{query_id}_dist"}
    ]
    extra_docs = rng.sample(negatives, k=min(extra_docs_per_query, len(negatives)))

    corpus = [
        {
            "doc_id": f"{query_id}_gold",
            "text": gold_text,
            "label": "gold",
        },
        {
            "doc_id": f"{query_id}_dist",
            "text": distractor_text,
            "label": "distractor",
        },
    ]
    corpus.extend(
        {
            "doc_id": doc["doc_id"],
            "text": doc["text"],
            "label": "background",
        }
        for doc in extra_docs
    )
    rng.shuffle(corpus)

    return {
        "sample_id": generated.get("id", query_id),
        "source_dataset": row["source_dataset"],
        "subset": row["subset"],
        "split": row["split"],
        "query_id": row["query_id"],
        "query_type": row.get("query_type"),
        "query_pos": generated.get("query_positive", row.get("query")),
        "query_neg": generated["query_negative"],
        "doc_gold": {
            "doc_id": f"{query_id}_gold",
            "text": gold_text,
        },
        "doc_distractor": {
            "doc_id": f"{query_id}_dist",
            "text": distractor_text,
        },
        "gold_answers": row.get("gold_answers", []),
        "answer_positive": generated.get("answer_positive"),
        "answer_negative": generated.get("answer_negative"),
        "domain": generated.get("domain"),
        "negation_type": generated.get("negation_type"),
        "adapter_note": (
            "doc_distractor is currently derived from generated.evidence as a proxy "
            "for an exclusion-violating hard negative. Replace with a curated "
            "distractor passage for the final benchmark."
        ),
        "corpus": corpus,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    if not rows:
        raise ValueError(f"No usable generated rows found in {input_path}")

    if len(rows) > args.sample_size:
        rows = rng.sample(rows, k=args.sample_size)

    background_pool = build_background_pool(rows)

    with output_path.open("w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc="Building triplets"):
            record = build_triplet_record(
                row=row,
                pool=background_pool,
                extra_docs_per_query=args.extra_docs_per_query,
                rng=rng,
            )
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved triplet benchmark to: {output_path}")
    print(f"Total samples: {len(rows)}")


if __name__ == "__main__":
    main()
