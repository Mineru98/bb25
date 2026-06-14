#!/usr/bin/env python3
"""Export BEIR/ir_datasets data into bb25 bench JSONL inputs.

Optional dependencies:
  pip install ir_datasets
  pip install snowballstemmer
  pip install sentence-transformers
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence


BEIR_DATASETS = {"arguana", "fiqa", "nfcorpus", "scidocs", "scifact"}
BEIR_SPLIT_DATASETS = {"fiqa", "nfcorpus", "scifact"}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="BEIR short name or ir_datasets id")
    parser.add_argument("--split", default="test", help="BEIR split for short dataset names")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--tokenizer", choices=["none", "split", "snowball"], default="snowball")
    parser.add_argument("--embed-model", default=None, help="Optional sentence-transformers model")
    parser.add_argument(
        "--embed-cache-dir",
        default=None,
        help="Optional sentence-transformers cache directory for the embedding model",
    )
    parser.add_argument(
        "--embed-local-files-only",
        action="store_true",
        help="Load the embedding model from the local cache only",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    return parser.parse_args()


def text_of(record: object) -> str:
    parts: list[str] = []
    for field in ("title", "text", "body", "abstract"):
        value = getattr(record, field, None)
        if value:
            parts.append(str(value))
    return " ".join(parts).strip()


def doc_field_texts(record: object) -> dict[str, str]:
    title = str(getattr(record, "title", "") or "")
    body = str(getattr(record, "text", None) or getattr(record, "body", None) or getattr(record, "abstract", None) or "")
    return {"title": title, "body": body}


def id_of(record: object, kind: str) -> str:
    value = getattr(record, f"{kind}_id", None)
    if value is None:
        value = getattr(record, "doc_id" if kind == "doc" else "query_id")
    return str(value)


def build_tokenizer(kind: str):
    if kind == "none":
        return None
    if kind == "split":
        return lambda text: text.lower().split()
    try:
        import snowballstemmer
    except ImportError as exc:
        raise SystemExit("snowball tokenizer requires: pip install snowballstemmer") from exc

    stemmer = snowballstemmer.stemmer("english")

    def tokenize(text: str) -> list[str]:
        raw = re.findall(r"[A-Za-z0-9]+", text.lower())
        return [stemmer.stemWord(term) for term in raw if term not in STOPWORDS]

    return tokenize


def embed_texts(
    model_name: str | None,
    texts: Sequence[str],
    batch_size: int,
    cache_dir: Path | None,
    local_files_only: bool,
) -> list[list[float]] | None:
    if model_name is None:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("embedding export requires: pip install sentence-transformers") from exc

    kwargs = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs["cache_folder"] = str(cache_dir)
    if local_files_only:
        kwargs["local_files_only"] = True
    try:
        model = SentenceTransformer(model_name, **kwargs)
    except TypeError as exc:
        if kwargs:
            options = ", ".join(sorted(kwargs))
            raise SystemExit(
                "installed sentence-transformers does not accept requested "
                f"constructor options ({options}); upgrade sentence-transformers "
                "or omit the matching embedding cache options"
            ) from exc
        raise
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [[float(x) for x in row] for row in vectors]


def limited(iterable: Iterable[object], max_items: int | None) -> list[object]:
    out: list[object] = []
    for item in iterable:
        if max_items is not None and len(out) >= max_items:
            break
        out.append(item)
    return out


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def resolve_dataset_id(dataset: str, split: str) -> str:
    if dataset in BEIR_DATASETS:
        if dataset not in BEIR_SPLIT_DATASETS:
            return f"beir/{dataset}"
        return f"beir/{dataset}/{split}"
    return dataset


def env_value(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def file_manifest(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    data = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    try:
        import ir_datasets
    except ImportError as exc:
        raise SystemExit("BEIR export requires: pip install ir_datasets") from exc

    dataset_id = resolve_dataset_id(args.dataset, args.split)
    dataset = ir_datasets.load(dataset_id)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    embed_cache_dir = Path(args.embed_cache_dir).expanduser().resolve() if args.embed_cache_dir else None

    tokenizer = build_tokenizer(args.tokenizer)
    docs = limited(dataset.docs_iter(), args.max_docs)
    queries = limited(dataset.queries_iter(), args.max_queries)
    query_ids = {id_of(query, "query") for query in queries}
    doc_ids = {id_of(doc, "doc") for doc in docs}

    doc_texts = [text_of(doc) for doc in docs]
    query_texts = [text_of(query) for query in queries]
    doc_embeddings = embed_texts(
        args.embed_model,
        doc_texts,
        args.batch_size,
        embed_cache_dir,
        args.embed_local_files_only,
    )
    query_embeddings = embed_texts(
        args.embed_model,
        query_texts,
        args.batch_size,
        embed_cache_dir,
        args.embed_local_files_only,
    )

    doc_rows = []
    for i, doc in enumerate(docs):
        row = {"doc_id": id_of(doc, "doc"), "text": doc_texts[i]}
        fields = doc_field_texts(doc)
        row["fields"] = fields
        if tokenizer is not None:
            row["terms"] = tokenizer(doc_texts[i])
            row["field_terms"] = {field: tokenizer(text) for field, text in fields.items()}
        if doc_embeddings is not None:
            row["embedding"] = doc_embeddings[i]
        doc_rows.append(row)

    query_rows = []
    for i, query in enumerate(queries):
        row = {"query_id": id_of(query, "query"), "text": query_texts[i]}
        if tokenizer is not None:
            row["terms"] = tokenizer(query_texts[i])
        if query_embeddings is not None:
            row["embedding"] = query_embeddings[i]
        query_rows.append(row)

    docs_path = out / "docs.jsonl"
    queries_path = out / "queries.jsonl"
    qrels_path = out / "qrels.tsv"
    write_jsonl(docs_path, doc_rows)
    write_jsonl(queries_path, query_rows)
    with qrels_path.open("w", encoding="utf8") as f:
        f.write("query_id\tQ0\tdoc_id\trelevance\n")
        for qrel in dataset.qrels_iter():
            qid = str(qrel.query_id)
            did = str(qrel.doc_id)
            if query_ids and qid not in query_ids:
                continue
            if doc_ids and did not in doc_ids:
                continue
            f.write(f"{qid}\t0\t{did}\t{int(qrel.relevance)}\n")

    meta = {
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "irDatasetId": dataset_id,
        "tokenizer": args.tokenizer,
        "embedModel": args.embed_model,
        "embedCacheDir": str(embed_cache_dir) if embed_cache_dir is not None else None,
        "embedLocalFilesOnly": args.embed_local_files_only,
        "embedding": {
            "model": args.embed_model,
            "cacheDir": str(embed_cache_dir) if embed_cache_dir is not None else None,
            "localFilesOnly": args.embed_local_files_only,
            "normalize": True,
        },
        "batchSize": args.batch_size,
        "maxDocs": args.max_docs,
        "maxQueries": args.max_queries,
        "docs": len(doc_rows),
        "queries": len(query_rows),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cacheEnvironment": {
                "SENTENCE_TRANSFORMERS_HOME": env_value("SENTENCE_TRANSFORMERS_HOME"),
                "HF_HOME": env_value("HF_HOME"),
                "TRANSFORMERS_CACHE": env_value("TRANSFORMERS_CACHE"),
            },
            "packages": {
                "ir_datasets": package_version("ir-datasets", "ir_datasets"),
                "sentence_transformers": package_version("sentence-transformers"),
                "snowballstemmer": package_version("snowballstemmer"),
            },
        },
        "files": {
            "docs": file_manifest(docs_path),
            "queries": file_manifest(queries_path),
            "qrels": file_manifest(qrels_path),
        },
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
