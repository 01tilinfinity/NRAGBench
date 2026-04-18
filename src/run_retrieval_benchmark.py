import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from generate_dataset import ROOT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Track 1 retrieval bottleneck benchmark on triplet JSONL data."
    )
    parser.add_argument(
        "--input-path",
        default=str(ROOT_DIR / "outputs" / "track1_benchmark_1k.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "outputs" / "retrieval_benchmark_1k"),
    )
    parser.add_argument(
        "--dense-models",
        nargs="*",
        default=["BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2"],
    )
    parser.add_argument("--tsne-samples", type=int, default=200)
    parser.add_argument("--collapse-samples", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_triplets(input_path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if "error" in row:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")
    return rows


def safe_type_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def reciprocal_rank_fusion(
    dense_scores: np.ndarray,
    bm25_scores: np.ndarray,
    k: int,
) -> np.ndarray:
    dense_order = np.argsort(-dense_scores)
    bm25_order = np.argsort(-bm25_scores)

    dense_rank = np.empty_like(dense_order)
    bm25_rank = np.empty_like(bm25_order)

    dense_rank[dense_order] = np.arange(1, len(dense_scores) + 1)
    bm25_rank[bm25_order] = np.arange(1, len(bm25_scores) + 1)

    return 1.0 / (k + dense_rank) + 1.0 / (k + bm25_rank)


def tokenize_for_bm25(text: str) -> list[str]:
    return text.lower().split()


def mean_reciprocal_rank(rank_positions: list[int | None]) -> float:
    reciprocal_ranks = []
    for position in rank_positions:
        if position is None:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / position)
    return float(np.mean(reciprocal_ranks))


def model_prompt_prefix(model_name: str, text: str, is_query: bool) -> str:
    if "e5" in model_name.lower():
        prefix = "query: " if is_query else "passage: "
        return prefix + text
    return text


def encode_with_model(
    model: SentenceTransformer,
    texts: list[str],
    model_name: str,
    is_query: bool,
    batch_size: int,
) -> np.ndarray:
    prepared = [model_prompt_prefix(model_name, text, is_query) for text in texts]
    embeddings = model.encode(
        prepared,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings)


def prepare_dense_inputs(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], dict[str, int]]:
    query_texts = [row["query_neg"] for row in rows]

    doc_text_to_index: dict[str, int] = {}
    doc_texts: list[str] = []

    for row in rows:
        for doc in row["corpus"]:
            text = doc["text"]
            if text not in doc_text_to_index:
                doc_text_to_index[text] = len(doc_texts)
                doc_texts.append(text)

    return query_texts, doc_texts, doc_text_to_index


def evaluate_bm25(rows: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    srn_flags = []
    absence_flags = []
    rank_positions = []
    for row in tqdm(rows, desc="Evaluating BM25"):
        corpus_docs = row["corpus"]
        doc_texts = [doc["text"] for doc in corpus_docs]
        bm25 = BM25Okapi([tokenize_for_bm25(text) for text in doc_texts])
        query_tokens = tokenize_for_bm25(row["query_neg"])
        scores = np.asarray(bm25.get_scores(query_tokens), dtype=float)

        gold_idx = next(i for i, doc in enumerate(corpus_docs) if doc["label"] == "gold")
        dist_idx = next(i for i, doc in enumerate(corpus_docs) if doc["label"] == "distractor")

        srn_flags.append(float(scores[gold_idx] > scores[dist_idx]))

        ranked = np.argsort(-scores)
        gold_rank = int(np.where(ranked == gold_idx)[0][0]) + 1
        rank_positions.append(gold_rank)

        absence_scores = np.delete(scores, gold_idx)
        absence_docs = [doc for i, doc in enumerate(corpus_docs) if i != gold_idx]
        top_doc = absence_docs[int(np.argmax(absence_scores))]
        absence_flags.append(float(top_doc["label"] == "distractor"))

    metrics = {
        "retriever": "bm25",
        "srn": float(np.mean(srn_flags)),
        "mrr": mean_reciprocal_rank(rank_positions),
        "distance_gap": float("nan"),
        "absence_top1": float(np.mean(absence_flags)),
        "semantic_collapse_rate": float("nan"),
        "samples": len(rows),
    }
    return metrics, []


def evaluate_dense_family(
    rows: list[dict[str, Any]],
    model_name: str,
    output_dir: Path,
    batch_size: int,
    tsne_samples: int,
    collapse_samples: int,
    rrf_k: int,
) -> list[dict[str, float]]:
    model = SentenceTransformer(model_name)

    query_texts, doc_texts, doc_text_to_index = prepare_dense_inputs(rows)
    query_embeddings = encode_with_model(
        model=model,
        texts=query_texts,
        model_name=model_name,
        is_query=True,
        batch_size=batch_size,
    )
    doc_embeddings = encode_with_model(
        model=model,
        texts=doc_texts,
        model_name=model_name,
        is_query=False,
        batch_size=batch_size,
    )

    dense_metrics = []
    dense_srn = []
    dense_absence = []
    dense_ranks = []
    distance_gaps = []
    collapse_records = []

    hybrid_srn = []
    hybrid_absence = []
    hybrid_ranks = []

    tsne_points = []
    tsne_labels = []
    tsne_meta = []

    for row_index, row in enumerate(tqdm(rows, desc=f"Evaluating {model_name}")):
        corpus_docs = row["corpus"]
        corpus_doc_indices = [doc_text_to_index[doc["text"]] for doc in corpus_docs]
        corpus_embeddings = doc_embeddings[corpus_doc_indices]
        query_embedding = query_embeddings[row_index : row_index + 1]

        dense_scores = cosine_similarity_matrix(query_embedding, corpus_embeddings)[0]
        dense_distances = 1.0 - dense_scores

        gold_idx = next(i for i, doc in enumerate(corpus_docs) if doc["label"] == "gold")
        dist_idx = next(i for i, doc in enumerate(corpus_docs) if doc["label"] == "distractor")

        dense_srn.append(float(dense_scores[gold_idx] > dense_scores[dist_idx]))

        dense_rank_order = np.argsort(-dense_scores)
        dense_gold_rank = int(np.where(dense_rank_order == gold_idx)[0][0]) + 1
        dense_ranks.append(dense_gold_rank)

        gap = float(dense_distances[dist_idx] - dense_distances[gold_idx])
        distance_gaps.append(gap)

        if gap < 0:
            collapse_records.append(
                {
                    "sample_id": row["sample_id"],
                    "query_neg": row["query_neg"],
                    "doc_gold": row["doc_gold"]["text"],
                    "doc_distractor": row["doc_distractor"]["text"],
                    "distance_gold": float(dense_distances[gold_idx]),
                    "distance_distractor": float(dense_distances[dist_idx]),
                    "gap": gap,
                }
            )

        absence_dense_scores = np.delete(dense_scores, gold_idx)
        absence_dense_docs = [doc for i, doc in enumerate(corpus_docs) if i != gold_idx]
        top_dense_doc = absence_dense_docs[int(np.argmax(absence_dense_scores))]
        dense_absence.append(float(top_dense_doc["label"] == "distractor"))

        doc_texts_local = [doc["text"] for doc in corpus_docs]
        bm25 = BM25Okapi([tokenize_for_bm25(text) for text in doc_texts_local])
        bm25_scores = np.asarray(bm25.get_scores(tokenize_for_bm25(row["query_neg"])), dtype=float)
        hybrid_scores = reciprocal_rank_fusion(dense_scores, bm25_scores, k=rrf_k)

        hybrid_rank_order = np.argsort(-hybrid_scores)
        hybrid_gold_rank = int(np.where(hybrid_rank_order == gold_idx)[0][0]) + 1
        hybrid_ranks.append(hybrid_gold_rank)
        hybrid_srn.append(float(hybrid_scores[gold_idx] > hybrid_scores[dist_idx]))

        absence_hybrid_scores = np.delete(hybrid_scores, gold_idx)
        absence_hybrid_docs = [doc for i, doc in enumerate(corpus_docs) if i != gold_idx]
        top_hybrid_doc = absence_hybrid_docs[int(np.argmax(absence_hybrid_scores))]
        hybrid_absence.append(float(top_hybrid_doc["label"] == "distractor"))

        if len(tsne_points) < tsne_samples * 3:
            tsne_points.extend(
                [
                    query_embedding[0],
                    corpus_embeddings[gold_idx],
                    corpus_embeddings[dist_idx],
                ]
            )
            tsne_labels.extend(["query_neg", "doc_gold", "doc_distractor"])
            tsne_meta.extend(
                [
                    row["sample_id"],
                    row["sample_id"],
                    row["sample_id"],
                ]
            )

    dense_metrics.append(
        {
            "retriever": f"dense::{model_name}",
            "srn": float(np.mean(dense_srn)),
            "mrr": mean_reciprocal_rank(dense_ranks),
            "distance_gap": float(np.mean(distance_gaps)),
            "absence_top1": float(np.mean(dense_absence)),
            "semantic_collapse_rate": float(np.mean([gap < 0 for gap in distance_gaps])),
            "samples": len(rows),
        }
    )
    dense_metrics.append(
        {
            "retriever": f"hybrid::{model_name}",
            "srn": float(np.mean(hybrid_srn)),
            "mrr": mean_reciprocal_rank(hybrid_ranks),
            "distance_gap": float(np.mean(distance_gaps)),
            "absence_top1": float(np.mean(hybrid_absence)),
            "semantic_collapse_rate": float(np.mean([gap < 0 for gap in distance_gaps])),
            "samples": len(rows),
        }
    )

    save_tsne_plot(
        output_dir=output_dir,
        model_name=model_name,
        vectors=np.asarray(tsne_points),
        labels=tsne_labels,
    )
    save_collapse_examples(
        output_dir=output_dir,
        model_name=model_name,
        collapse_records=collapse_records[:collapse_samples],
    )

    if collapse_records:
        print(f"\nSemantic collapse examples for {model_name}:")
        for example in collapse_records[:collapse_samples]:
            print(f"- sample_id={example['sample_id']} gap={example['gap']:.4f}")
            print(f"  query_neg: {example['query_neg']}")
            print(f"  gold: {example['doc_gold'][:180]}")
            print(f"  distractor: {example['doc_distractor'][:180]}")

    return dense_metrics


def save_tsne_plot(
    output_dir: Path,
    model_name: str,
    vectors: np.ndarray,
    labels: list[str],
) -> None:
    if len(vectors) < 3:
        return

    perplexity = min(30, max(2, len(vectors) - 1))
    tsne = TSNE(
        n_components=2,
        random_state=42,
        init="random",
        learning_rate="auto",
        perplexity=perplexity,
    )
    transformed = tsne.fit_transform(vectors)

    label_to_style = {
        "query_neg": {"color": "#1f77b4", "marker": "o"},
        "doc_gold": {"color": "#2ca02c", "marker": "^"},
        "doc_distractor": {"color": "#d62728", "marker": "x"},
    }

    plt.figure(figsize=(8, 6))
    for label, style in label_to_style.items():
        indices = [i for i, item in enumerate(labels) if item == label]
        if not indices:
            continue
        points = transformed[indices]
        plt.scatter(
            points[:, 0],
            points[:, 1],
            label=label,
            c=style["color"],
            marker=style["marker"],
            alpha=0.7,
        )

    plt.title(f"t-SNE Embedding Distribution: {model_name}")
    plt.legend()
    plt.tight_layout()

    safe_name = model_name.replace("/", "__")
    output_path = output_dir / f"tsne_{safe_name}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_collapse_examples(
    output_dir: Path,
    model_name: str,
    collapse_records: list[dict[str, Any]],
) -> None:
    safe_name = model_name.replace("/", "__")
    output_path = output_dir / f"collapse_examples_{safe_name}.jsonl"
    with output_path.open("w", encoding="utf-8") as fout:
        for record in collapse_records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_subset(
    rows: list[dict[str, Any]],
    subset_name: str,
    output_dir: Path,
    dense_models: list[str],
    batch_size: int,
    tsne_samples: int,
    collapse_samples: int,
    rrf_k: int,
) -> pd.DataFrame:
    subset_output_dir = output_dir / safe_type_name(subset_name)
    subset_output_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    bm25_metrics, _ = evaluate_bm25(rows)
    bm25_metrics["subset"] = subset_name
    metrics.append(bm25_metrics)

    for model_name in dense_models:
        for item in evaluate_dense_family(
            rows=rows,
            model_name=model_name,
            output_dir=subset_output_dir,
            batch_size=batch_size,
            tsne_samples=tsne_samples,
            collapse_samples=collapse_samples,
            rrf_k=rrf_k,
        ):
            item["subset"] = subset_name
            metrics.append(item)

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df[
        [
            "subset",
            "retriever",
            "srn",
            "mrr",
            "distance_gap",
            "absence_top1",
            "semantic_collapse_rate",
            "samples",
        ]
    ]
    metrics_df.to_csv(subset_output_dir / "metrics.csv", index=False)
    (subset_output_dir / "metrics.json").write_text(
        json.dumps(metrics_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics_df


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_triplets(input_path)
    subsets = {"overall": rows}
    for negation_type in sorted({row["negation_type"] for row in rows}):
        subsets[negation_type] = [row for row in rows if row["negation_type"] == negation_type]

    frames = []
    for subset_name, subset_rows in subsets.items():
        frames.append(
            evaluate_subset(
                rows=subset_rows,
                subset_name=subset_name,
                output_dir=output_dir,
                dense_models=args.dense_models,
                batch_size=args.batch_size,
                tsne_samples=min(args.tsne_samples, len(subset_rows)),
                collapse_samples=args.collapse_samples,
                rrf_k=args.rrf_k,
            )
        )

    metrics_df = pd.concat(frames, ignore_index=True)
    metrics_path = output_dir / "metrics_all.csv"
    metrics_df.to_csv(metrics_path, index=False)
    (output_dir / "metrics_all.json").write_text(
        json.dumps(metrics_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nRetrieval Benchmark Results")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
