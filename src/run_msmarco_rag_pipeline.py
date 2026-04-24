import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "outputs" / "msmarco_negative_1k" / "data" / "train-00000.parquet"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "outputs" / "msmarco_rag_qwen_small_worst_1k.json"
DEFAULT_WORK_DIR = ROOT_DIR / "outputs" / "msmarco_rag_qwen_small_worst_work"
DEFAULT_PROMPT_PATH = ROOT_DIR / "prompts" / "msmarco_rag_answer.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Contriever + Qwen3 reranker + Qwen2.5 generator RAG pipeline."
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--prompt-path", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--retriever-model", default="facebook/contriever-msmarco")
    parser.add_argument("--reranker-model", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--retrieval-candidates", type=int, default=6)
    parser.add_argument("--rerank-candidates", type=int, default=3)
    parser.add_argument("--context-docs", type=int, default=2)
    parser.add_argument("--retriever-batch-size", type=int, default=32)
    parser.add_argument("--reranker-batch-size", type=int, default=2)
    parser.add_argument("--reranker-max-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-context-chars", type=int, default=5000)
    parser.add_argument("--max-doc-chars", type=int, default=1800)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument(
        "--selection-mode",
        choices=["top", "worst"],
        default="worst",
        help="Use highest reranker scores or lowest reranker scores as generator context.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "retrieve", "rerank", "generate"],
        default="all",
    )
    return parser.parse_args()


def get_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        if preferred == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        if preferred == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available.")
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_dtype(device: str) -> torch.dtype:
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def cleanup_model(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def normalize_text(text: str) -> str:
    cleaned = text.strip().replace("\r", " ")
    cleaned = cleaned.strip("\"'` ")
    if "\n" in cleaned:
        cleaned = cleaned.splitlines()[0].strip()
    cleaned = re.sub(r"^(answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def load_rows(input_path: Path, max_rows: int | None) -> list[dict[str, Any]]:
    rows = [dict(row) for row in load_dataset("parquet", data_files=str(input_path), split="train")]
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def read_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()


def trim(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0].strip()


def build_corpus(rows: list[dict[str, Any]], max_doc_chars: int) -> tuple[list[str], list[str]]:
    doc_id_to_text: dict[str, str] = {}
    for row in rows:
        doc_id = str(row["sample_id"])
        doc_id_to_text[doc_id] = trim(row.get("document") or "", max_doc_chars)
    doc_ids = sorted(doc_id_to_text, key=lambda value: int(value))
    doc_texts = [doc_id_to_text[doc_id] for doc_id in doc_ids]
    return doc_ids, doc_texts


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@torch.no_grad()
def encode_contriever(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        outputs = model(**inputs)
        pooled = mean_pooling(outputs[0], inputs["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.detach().cpu().float().numpy())
    return np.vstack(embeddings)


def retrieve(args: argparse.Namespace, rows: list[dict[str, Any]], work_dir: Path) -> None:
    device = get_device(args.device)
    doc_ids, doc_texts = build_corpus(rows, args.max_doc_chars)
    positive_queries = [str(row.get("query") or "") for row in rows]
    negative_queries = [str(row.get("negative_query") or "") for row in rows]

    tokenizer = AutoTokenizer.from_pretrained(args.retriever_model)
    model = AutoModel.from_pretrained(args.retriever_model).to(device).eval()

    doc_embeddings = encode_contriever(
        doc_texts, tokenizer, model, device, args.retriever_batch_size
    )
    positive_embeddings = encode_contriever(
        positive_queries, tokenizer, model, device, args.retriever_batch_size
    )
    negative_embeddings = encode_contriever(
        negative_queries, tokenizer, model, device, args.retriever_batch_size
    )
    cleanup_model(model, tokenizer)

    k = min(args.retrieval_candidates, len(doc_ids))
    retrieval_records = []
    for row_idx, row in enumerate(tqdm(rows, desc="Retrieving", unit="row")):
        record: dict[str, Any] = {"sample_id": int(row["sample_id"])}
        for query_name, query_embeddings in [
            ("positive", positive_embeddings),
            ("negative", negative_embeddings),
        ]:
            scores = query_embeddings[row_idx] @ doc_embeddings.T
            candidate_indices = np.argpartition(-scores, kth=k - 1)[:k]
            candidate_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
            record[f"{query_name}_candidates"] = [
                {
                    "doc_id": doc_ids[int(doc_idx)],
                    "score": float(scores[int(doc_idx)]),
                }
                for doc_idx in candidate_indices
            ]
        retrieval_records.append(record)

    (work_dir / "retrieval_candidates.json").write_text(
        json.dumps(retrieval_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_retrieval(work_dir: Path) -> dict[int, dict[str, Any]]:
    path = work_dir / "retrieval_candidates.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    return {int(record["sample_id"]): record for record in records}


class QwenYesNoReranker:
    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        max_length: int,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype(device),
        ).to(device).eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix_tokens = self.tokenizer.encode(
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
            "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n<|im_start|>user\n",
            add_special_tokens=False,
        )
        self.suffix_tokens = self.tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )

    def _format_pair(self, query: str, document: str) -> str:
        return (
            "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    @torch.no_grad()
    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        scores: list[float] = []
        body_max_length = max(64, self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens))
        for start in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[start : start + self.batch_size]
            texts = [self._format_pair(query, document) for query, document in batch_pairs]
            tokenized = self.tokenizer(
                texts,
                padding=False,
                truncation="longest_first",
                max_length=body_max_length,
                return_attention_mask=False,
            )
            input_ids = [
                self.prefix_tokens + ids + self.suffix_tokens
                for ids in tokenized["input_ids"]
            ]
            padded = self.tokenizer.pad(
                {"input_ids": input_ids},
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**padded).logits[:, -1, :]
            yes_no_logits = logits[:, [self.token_false_id, self.token_true_id]]
            batch_scores = torch.nn.functional.log_softmax(yes_no_logits, dim=1)[:, 1].exp()
            scores.extend(batch_scores.detach().cpu().float().tolist())
        return np.asarray(scores, dtype=float)

    def close(self) -> None:
        cleanup_model(self.model, self.tokenizer)


def rerank_one(
    reranker: QwenYesNoReranker,
    query: str,
    candidate_doc_ids: list[str],
    doc_text_by_id: dict[str, str],
    selection_mode: str,
) -> list[dict[str, Any]]:
    pairs = [(query, doc_text_by_id[doc_id]) for doc_id in candidate_doc_ids]
    scores = reranker.predict(pairs)
    scored = [
        {"doc_id": doc_id, "score": float(score)}
        for doc_id, score in zip(candidate_doc_ids, list(scores))
    ]
    reverse = selection_mode == "top"
    return sorted(scored, key=lambda item: item["score"], reverse=reverse)


def rerank(args: argparse.Namespace, rows: list[dict[str, Any]], work_dir: Path) -> None:
    device = get_device(args.device)
    doc_ids, doc_texts = build_corpus(rows, args.max_doc_chars)
    doc_text_by_id = dict(zip(doc_ids, doc_texts))
    retrieval_by_id = load_retrieval(work_dir)
    checkpoint_path = work_dir / f"reranked_{args.selection_mode}.jsonl"

    existing: dict[int, dict[str, Any]] = {}
    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    existing[int(record["sample_id"])] = record

    reranker_model = QwenYesNoReranker(
        model_name=args.reranker_model,
        device=device,
        batch_size=args.reranker_batch_size,
        max_length=args.reranker_max_length,
    )

    pending = [row for row in rows if int(row["sample_id"]) not in existing]
    with checkpoint_path.open("a", encoding="utf-8") as handle:
        for row in tqdm(pending, desc="Reranking", unit="row"):
            sample_id = int(row["sample_id"])
            retrieval = retrieval_by_id[sample_id]
            record = {"sample_id": sample_id}
            for query_name, query_text in [
                ("positive", str(row.get("query") or "")),
                ("negative", str(row.get("negative_query") or "")),
            ]:
                candidate_doc_ids = [
                    item["doc_id"]
                    for item in retrieval[f"{query_name}_candidates"][: args.rerank_candidates]
                ]
                record[f"{query_name}_reranked"] = rerank_one(
                    reranker=reranker_model,
                    query=query_text,
                    candidate_doc_ids=candidate_doc_ids,
                    doc_text_by_id=doc_text_by_id,
                    selection_mode=args.selection_mode,
                )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    reranker_model.close()


def load_reranked(work_dir: Path, selection_mode: str) -> dict[int, dict[str, Any]]:
    path = work_dir / f"reranked_{selection_mode}.jsonl"
    records: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[int(record["sample_id"])] = record
    return records


def build_context(
    ranked_docs: list[dict[str, Any]],
    doc_text_by_id: dict[str, str],
    context_docs: int,
    max_context_chars: int,
) -> str:
    chunks = []
    for rank, item in enumerate(ranked_docs[:context_docs], start=1):
        doc_text = doc_text_by_id[item["doc_id"]]
        chunks.append(f"[Document {rank}]\n{doc_text}")
    context = "\n\n".join(chunks)
    return trim(context, max_context_chars)


@torch.no_grad()
def generate_answer(
    query: str,
    context: str,
    prompt_template: str,
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> str:
    prompt = prompt_template.format(query=query, context=context)
    messages = [
        {
            "role": "system",
            "content": "You are a concise retrieval-augmented question answering assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    answer = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0]
    return normalize_text(answer)


def load_generation_checkpoint(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    records: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[int(record["sample_id"])] = record
    return records


def generate(args: argparse.Namespace, rows: list[dict[str, Any]], work_dir: Path, output_path: Path) -> None:
    device = get_device(args.device)
    prompt_template = read_prompt(Path(args.prompt_path))
    doc_ids, doc_texts = build_corpus(rows, args.max_doc_chars)
    doc_text_by_id = dict(zip(doc_ids, doc_texts))
    reranked = load_reranked(work_dir, args.selection_mode)
    checkpoint_path = work_dir / f"generation_{args.selection_mode}.jsonl"
    existing = load_generation_checkpoint(checkpoint_path)

    tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.generator_model,
        torch_dtype=model_dtype(device),
    ).to(device).eval()

    pending = [row for row in rows if int(row["sample_id"]) not in existing]
    with checkpoint_path.open("a", encoding="utf-8") as handle:
        for row in tqdm(pending, desc="Generating answers", unit="row"):
            sample_id = int(row["sample_id"])
            ranked = reranked[sample_id]
            positive_context = build_context(
                ranked["positive_reranked"],
                doc_text_by_id,
                args.context_docs,
                args.max_context_chars,
            )
            negative_context = build_context(
                ranked["negative_reranked"],
                doc_text_by_id,
                args.context_docs,
                args.max_context_chars,
            )
            positive_answer = generate_answer(
                query=str(row.get("query") or ""),
                context=positive_context,
                prompt_template=prompt_template,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=args.max_new_tokens,
            )
            negative_answer = generate_answer(
                query=str(row.get("negative_query") or ""),
                context=negative_context,
                prompt_template=prompt_template,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=args.max_new_tokens,
            )
            record = {
                "sample_id": sample_id,
                "positive_answer": positive_answer,
                "negative_answer": negative_answer,
            }
            existing[sample_id] = record
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    cleanup_model(model, tokenizer)

    final_rows = []
    for row in rows:
        sample_id = int(row["sample_id"])
        generated = existing[sample_id]
        final_rows.append(
            {
                "positive_query": row.get("query") or "",
                "negative_query": row.get("negative_query") or "",
                "document": row.get("document") or "",
                "positive_answer": generated["positive_answer"],
                "negative_answer": generated["negative_answer"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(final_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path, args.max_rows)

    if args.stage in {"all", "retrieve"}:
        retrieve(args, rows, work_dir)
    if args.stage in {"all", "rerank"}:
        rerank(args, rows, work_dir)
    if args.stage in {"all", "generate"}:
        generate(args, rows, work_dir, output_path)
        print(f"Saved RAG output: {output_path}")


if __name__ == "__main__":
    main()
