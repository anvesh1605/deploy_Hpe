from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent


def read_records(path: Path) -> Iterable[Dict[str, object]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(payload, dict):
            for key in ("results", "items", "predictions", "rows", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            yield item
                    return
            yield payload
            return
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    decoded_any = False
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            payload, next_index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            break
        decoded_any = True
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
        elif isinstance(payload, dict):
            for key in ("results", "items", "predictions", "rows", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            yield item
                    break
            else:
                yield payload
        index = next_index

    if decoded_any:
        return

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(value).lower()).strip()


def detect_row_format(row: Dict[str, object]) -> str:
    if isinstance(row.get("messages"), list):
        return "chat_messages"
    if row.get("input_text") is not None or row.get("target_value") is not None:
        return "qa_jsonl"
    if row.get("source_type") is not None or row.get("data_family") is not None:
        return "metadata_rich"
    return "unknown"


def extract_message_text(messages: object) -> str:
    if not isinstance(messages, list):
        return ""
    user_parts: List[str] = []
    all_parts: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = normalize_text(message.get("content"))
        if not content:
            continue
        all_parts.append(content)
        role = normalize_text(message.get("role")).lower()
        if role == "user":
            user_parts.append(content)
    if user_parts:
        return "\n".join(user_parts).strip()
    return "\n".join(all_parts).strip()


def extract_question_text(row: Dict[str, object]) -> str:
    candidates: Sequence[object] = (
        row.get("input_text"),
        row.get("source_question"),
        row.get("question"),
        row.get("prompt"),
        row.get("text"),
        row.get("query"),
        row.get("instruction"),
    )
    for candidate in candidates:
        text = normalize_text(candidate)
        if text:
            return text

    messages_text = extract_message_text(row.get("messages"))
    if messages_text:
        return messages_text

    return ""


def extract_metadata(row: Dict[str, object]) -> Dict[str, object]:
    keys = [
        "intent",
        "source_type",
        "data_family",
        "switch",
        "version",
        "sub_version",
        "document_title",
        "section",
        "topic",
        "command",
        "source_file",
        "source_excerpt_file",
    ]
    metadata: Dict[str, object] = {}
    for key in keys:
        value = row.get(key)
        if value is not None and normalize_text(value):
            metadata[key] = value
    if isinstance(row.get("slots"), dict):
        metadata["slots"] = row["slots"]
    return metadata


@dataclass
class Record:
    source_file: str
    row_index: int
    row_format: str
    question: str
    norm_key: str
    metadata: Dict[str, object]


def load_records(paths: Sequence[Path]) -> List[Record]:
    records: List[Record] = []
    for path in paths:
        for index, row in enumerate(read_records(path)):
            if not isinstance(row, dict):
                continue
            question = extract_question_text(row)
            if not question:
                continue
            records.append(
                Record(
                    source_file=str(path),
                    row_index=index,
                    row_format=detect_row_format(row),
                    question=question,
                    norm_key=normalize_key(question),
                    metadata=extract_metadata(row),
                )
            )
    return records


def collect_paths(raw_paths: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in raw_paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            paths.extend(sorted(item for item in path.rglob("*.jsonl") if item.is_file()))
        elif path.is_file():
            paths.append(path)
    unique: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def load_transformer(model_name: str, device: torch.device):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        raise SystemExit(
            "transformers is required for this analyzer. Run it from the local venv that has transformers installed."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(
    texts: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 256,
) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty((0, 0))


def top_k_matches(
    source_records: Sequence[Record],
    compare_records: Sequence[Record],
    source_embeddings: torch.Tensor,
    compare_embeddings: torch.Tensor,
    top_k: int,
) -> List[Dict[str, object]]:
    if not len(source_records) or not len(compare_records):
        return []

    similarity = source_embeddings @ compare_embeddings.T
    results: List[Dict[str, object]] = []
    for idx, source in enumerate(source_records):
        scores = similarity[idx]
        k = min(top_k, len(compare_records))
        values, indices = torch.topk(scores, k=k)
        matches = []
        for value, index in zip(values.tolist(), indices.tolist()):
            candidate = compare_records[index]
            matches.append(
                {
                    "similarity": round(float(value), 6),
                    "question": candidate.question,
                    "source_file": candidate.source_file,
                    "row_index": candidate.row_index,
                    "row_format": candidate.row_format,
                    "metadata": candidate.metadata,
                }
            )
        best = matches[0] if matches else {}
        results.append(
            {
                "source_question": source.question,
                "source_file": source.source_file,
                "source_row_index": source.row_index,
                "source_row_format": source.row_format,
                "source_metadata": source.metadata,
                "best_similarity": best.get("similarity", 0.0),
                "best_match_question": best.get("question", ""),
                "best_match_file": best.get("source_file", ""),
                "best_match_row_index": best.get("row_index", -1),
                "best_match_row_format": best.get("row_format", ""),
                "best_match_metadata": best.get("metadata", {}),
                "top_matches": matches,
            }
        )
    return results


def build_summary(matches: Sequence[Dict[str, object]], source_records: Sequence[Record], compare_records: Sequence[Record]) -> Dict[str, object]:
    best_scores = [float(item.get("best_similarity", 0.0)) for item in matches]
    exact_matches = sum(
        1
        for item in matches
        if normalize_key(item.get("source_question", "")) == normalize_key(item.get("best_match_question", ""))
    )
    row_formats = {
        "source": {},
        "compare": {},
    }
    for record in source_records:
        row_formats["source"][record.row_format] = row_formats["source"].get(record.row_format, 0) + 1
    for record in compare_records:
        row_formats["compare"][record.row_format] = row_formats["compare"].get(record.row_format, 0) + 1

    def count_at_least(threshold: float) -> int:
        return sum(1 for score in best_scores if score >= threshold)

    return {
        "source_rows": len(source_records),
        "compare_rows": len(compare_records),
        "question_only_mode": True,
        "exact_text_matches": exact_matches,
        "average_best_similarity": round(sum(best_scores) / len(best_scores), 6) if best_scores else 0.0,
        "max_best_similarity": round(max(best_scores), 6) if best_scores else 0.0,
        "min_best_similarity": round(min(best_scores), 6) if best_scores else 0.0,
        "matches_over_0_90": count_at_least(0.90),
        "matches_over_0_80": count_at_least(0.80),
        "matches_over_0_70": count_at_least(0.70),
        "row_format_counts": row_formats,
    }


def markdown_escape(value: object) -> str:
    text = normalize_text(value)
    return text.replace("|", "\\|")


def render_markdown(summary: Dict[str, object], matches: Sequence[Dict[str, object]], top_n: int = 20) -> str:
    lines = [
        "# Transformer Question Format Analysis",
        "",
        "## Summary",
        f"- Source rows: {summary['source_rows']}",
        f"- Compare rows: {summary['compare_rows']}",
        f"- Exact text matches: {summary['exact_text_matches']}",
        f"- Average best similarity: {summary['average_best_similarity']}",
        f"- Matches over 0.90: {summary['matches_over_0_90']}",
        f"- Matches over 0.80: {summary['matches_over_0_80']}",
        "",
        "## Row Formats",
        f"- Source: {json.dumps(summary['row_format_counts']['source'], ensure_ascii=False)}",
        f"- Compare: {json.dumps(summary['row_format_counts']['compare'], ensure_ascii=False)}",
        "",
        "## Top Matches",
        "",
        "| # | Source question | Best match | Similarity | Source format | Match format |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for idx, item in enumerate(matches[:top_n], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    markdown_escape(item.get("source_question", "")),
                    markdown_escape(item.get("best_match_question", "")),
                    str(item.get("best_similarity", "")),
                    markdown_escape(item.get("source_row_format", "")),
                    markdown_escape(item.get("best_match_row_format", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines).strip() + "\n"


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze question similarity across different Aruba JSONL formats using a transformer encoder."
    )
    parser.add_argument(
        "--source",
        nargs="+",
        required=True,
        help="Source JSONL file or directory containing question rows.",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        required=True,
        help="Comparison JSONL file or directory containing alternative question rows.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "outputs_question_format_analysis",
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--model_name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face model used for embedding questions.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of nearest matches to keep.")
    parser.add_argument("--batch_size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--max_length", type=int, default=256, help="Tokenizer max length.")
    args = parser.parse_args()

    source_paths = collect_paths(args.source)
    compare_paths = collect_paths(args.compare)
    if not source_paths:
        raise SystemExit("No source JSONL files were found.")
    if not compare_paths:
        raise SystemExit("No compare JSONL files were found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_transformer(args.model_name, device)

    source_records = load_records(source_paths)
    compare_records = load_records(compare_paths)
    if not source_records:
        raise SystemExit("No usable source questions were extracted.")
    if not compare_records:
        raise SystemExit("No usable comparison questions were extracted.")

    source_embeddings = embed_texts(
        [record.question for record in source_records],
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    compare_embeddings = embed_texts(
        [record.question for record in compare_records],
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    matches = top_k_matches(source_records, compare_records, source_embeddings, compare_embeddings, args.top_k)
    summary = build_summary(matches, source_records, compare_records)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    matches_path = output_dir / "question_format_matches.jsonl"
    report_path = output_dir / "question_format_report.json"
    md_path = output_dir / "question_format_report.md"

    write_jsonl(matches_path, matches)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(summary, matches), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote:\n- {matches_path}\n- {report_path}\n- {md_path}")


if __name__ == "__main__":
    main()
