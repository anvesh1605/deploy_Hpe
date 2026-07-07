from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


DEFAULT_DATA_PATH = Path(r"C:\Hpe\Train\imporved_data_addition\aruba_aoscx_bilstm_balanced_2040_merged.jsonl")
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.55

TEST_QUESTIONS = [
    "For 4100i AOS-CX 10.16, what is the supported capacity for VSF member number range?",
    "What is the supported route scale on Aruba 6200?",
    "What is the maximum supported IPv4 route scale for Aruba 6300 switch running AOS-CX 10.16?",
    "How can I bring up a 6300 in VSX mode?",
    "Since which version does Aruba 4100 support VSF?",
    "For 6300_6400 AOS-CX 10.16, what is the syntax of the bfd <IPv4-ADDR> command?",
    "What is the CLI syntax for clear erps statistics in AOS-CX?",
    "For Aruba 8320 AOS-CX 10.06, what is the output of the show ip route command?",
]


def read_jsonl_like(path: Path) -> Iterable[Dict[str, object]]:
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
            for key in ("rows", "items", "data", "results", "predictions"):
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
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
            decoded_any = True
        elif isinstance(payload, dict):
            for key in ("rows", "items", "data", "results", "predictions"):
                value = payload.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            yield item
                    break
            else:
                yield payload
            decoded_any = True
        else:
            yield payload
            decoded_any = True
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
    return re.sub(r"\s+", " ", str(value).replace("\r\n", "\n").replace("\r", "\n")).strip()


def normalize_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(value).lower()).strip()


def stringify_slots(slots: object) -> str:
    if not isinstance(slots, dict):
        return ""
    parts = []
    for key in sorted(slots):
        value = slots.get(key)
        if value is None:
            continue
        value_text = normalize_text(value)
        if value_text:
            parts.append(f"{key}: {value_text}")
    return "; ".join(parts)


def build_candidate_text(row: Dict[str, object]) -> str:
    input_text = normalize_text(row.get("input_text") or row.get("question") or row.get("source_question"))
    intent = normalize_text(row.get("intent"))
    slots = stringify_slots(row.get("slots"))
    target_value = normalize_text(row.get("target_value") or row.get("answer") or row.get("reference"))
    return "\n".join(
        [
            f"Question: {input_text}",
            f"Intent: {intent}",
            f"Slots: {slots}",
            f"Answer: {target_value}",
        ]
    ).strip()


def detect_question_type(question: str) -> str:
    text = normalize_key(question)
    if not text:
        return "unknown"
    if "syntax" in text or re.search(r"\bcommand\s+syntax\b", text):
        return "cli_syntax"
    if "output" in text:
        return "cli_output"
    if re.search(r"\bsince which version\b|\bfrom which version\b|\bwhich version\b", text):
        return "version_support"
    if "vsx" in text and any(word in text for word in ("how", "configure", "bring up", "setup", "set up")):
        return "vsx_procedure"
    if any(word in text for word in ("route scale", "route capacity", "supported route", "maximum supported ipv4 route", "maximum supported ipv6 route")):
        return "route_scale"
    if "vsf" in text:
        return "vsf_related"
    return "generic_semantic"


def question_keywords(question: str) -> List[str]:
    text = normalize_key(question)
    return [token for token in text.split() if token]


def semantic_guard(question: str, row: Dict[str, object], question_type: str) -> Tuple[bool, str]:
    q = normalize_key(question)
    candidate_blob = " ".join(
        [
            normalize_key(row.get("input_text")),
            normalize_key(row.get("intent")),
            normalize_key(row.get("target_value")),
            normalize_key(stringify_slots(row.get("slots"))),
        ]
    )
    intent = normalize_key(row.get("intent"))

    if question_type == "route_scale":
        route_terms = (
            "route",
            "routes",
            "route scale",
            "route capacity",
            "ipv4 route",
            "ipv6 route",
            "long prefix",
            "asic",
        )
        if not any(term in candidate_blob for term in route_terms):
            return False, "candidate does not mention route-scale language"
        blocked_terms = ("next hop", "next hops", "vsf member", "lag", "lag links")
        if any(term in candidate_blob for term in blocked_terms) and not any(term in q for term in blocked_terms):
            return False, "candidate is about a different scale topic"

    if question_type == "vsx_procedure":
        if "vsx" not in candidate_blob:
            return False, "candidate does not mention VSX"
        if intent == "concept_explanation":
            return False, "prefer procedural intent over concept explanation"

    if question_type == "version_support":
        requested = []
        for token in ("vsf", "vsx", "issu", "bfd", "route", "acl", "qos"):
            if token in q:
                requested.append(token)
        if requested and not any(token in candidate_blob for token in requested):
            return False, "feature keyword mismatch"
        if intent != "version_support":
            return False, "prefer version_support intent"

    if question_type == "cli_syntax":
        if intent != "cli_syntax" and "syntax" not in intent:
            return False, "prefer cli_syntax intent"

    if question_type == "cli_output":
        if intent != "cli_output" and "output" not in candidate_blob and "show" not in candidate_blob:
            return False, "prefer cli_output intent"

    return True, ""


def detect_guard_priority(question_type: str, row: Dict[str, object]) -> int:
    intent = normalize_key(row.get("intent"))
    if question_type == "route_scale":
        return 0 if any(token in intent for token in ("capacity", "scale", "version", "concept_explanation")) else 1
    if question_type == "vsx_procedure":
        if intent == "configuration_procedure":
            return 0
        if intent == "concept_explanation":
            return 2
        return 1
    if question_type == "version_support":
        return 0 if intent == "version_support" else 2
    if question_type == "cli_syntax":
        return 0 if intent == "cli_syntax" else 2
    if question_type == "cli_output":
        return 0 if intent == "cli_output" else 2
    return 1


@dataclass
class Candidate:
    row: Dict[str, object]
    candidate_text: str
    embedding: torch.Tensor


def load_model():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        return ("sentence_transformers", model)
    except ModuleNotFoundError:
        print("sentence-transformers is not installed. Install it with:")
        print("pip install sentence-transformers")
    except Exception as exc:
        print(f"sentence-transformers could not be loaded: {exc}")
        print("Falling back to transformers AutoModel embedding.")

    try:
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "transformers is required. Install it with: pip install transformers"
        ) from exc

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return ("transformers", (tokenizer, model, device))


def encode_texts(model_kind: str, model_obj, texts: Sequence[str], batch_size: int = 32) -> torch.Tensor:
    if not texts:
        return torch.empty((0, 0))

    if model_kind == "sentence_transformers":
        vectors = model_obj.encode(list(texts), convert_to_tensor=True, normalize_embeddings=True)
        if not isinstance(vectors, torch.Tensor):
            vectors = torch.tensor(vectors)
        return vectors.cpu()

    tokenizer, model, device = model_obj
    vectors: List[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            token_embeddings = output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).type_as(token_embeddings)
            pooled = (token_embeddings * attention_mask).sum(dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            pooled = F.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu())
    return torch.cat(vectors, dim=0)


def load_rows(data_path: Path) -> List[Dict[str, object]]:
    paths: List[Path]
    if data_path.is_dir():
        paths = sorted(p for p in data_path.rglob("*.jsonl") if p.is_file())
    else:
        paths = [data_path]

    rows: List[Dict[str, object]] = []
    for path in paths:
        for row in read_jsonl_like(path):
            if not isinstance(row, dict):
                continue
            if not normalize_text(row.get("input_text")) and not normalize_text(row.get("question")):
                continue
            rows.append(row)
    return rows


def rank_rows(
    question: str,
    question_embedding: torch.Tensor,
    candidates: Sequence[Candidate],
    top_k: int,
    threshold: float,
) -> List[Dict[str, object]]:
    question_type = detect_question_type(question)
    scored: List[Tuple[float, int, Candidate, bool, str]] = []
    for idx, candidate in enumerate(candidates):
        score = float(torch.dot(question_embedding, candidate.embedding).item())
        passed, reason = semantic_guard(question, candidate.row, question_type)
        intent_priority = detect_guard_priority(question_type, candidate.row)
        if not passed:
            score -= 1.0
        score += max(0.0, 0.08 - intent_priority * 0.03)
        scored.append((score, idx, candidate, passed, reason))

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    results: List[Dict[str, object]] = []
    for rank, (score, _, candidate, passed, reason) in enumerate(scored[:top_k], start=1):
        row = candidate.row
        results.append(
            {
                "rank": rank,
                "score": round(score, 6),
                "intent": normalize_text(row.get("intent")),
                "slots": row.get("slots") if isinstance(row.get("slots"), dict) else {},
                "input_text": normalize_text(row.get("input_text") or row.get("question") or row.get("source_question")),
                "target_value": normalize_text(row.get("target_value") or row.get("answer") or row.get("reference")),
                "semantic_guard_passed": passed and score >= threshold,
                "rejection_reason": "" if passed and score >= threshold else reason or "below threshold",
            }
        )
    return results


def choose_final_answer(
    question: str, ranked: Sequence[Dict[str, object]], threshold: float
) -> Tuple[str, Optional[str], float, str, str]:
    top_score = float(ranked[0]["score"]) if ranked else 0.0
    selected_item: Optional[Dict[str, object]] = None
    for item in ranked:
        if item["semantic_guard_passed"] and float(item["score"]) >= threshold:
            selected_item = item
            break

    if selected_item is not None:
        return (
            normalize_text(selected_item["target_value"]),
            normalize_text(selected_item["intent"]),
            float(selected_item["score"]),
            "found",
            "",
        )

    rejection_reason = ""
    for item in ranked:
        if item["rejection_reason"]:
            rejection_reason = str(item["rejection_reason"])
            break

    lookup_status = "low_similarity" if ranked else "not_found"
    return (
        "No reliable semantic match found in the current dataset.",
        None,
        top_score,
        lookup_status,
        rejection_reason,
    )


def print_question_report(
    question: str,
    ranked: Sequence[Dict[str, object]],
    final_answer: str,
    final_intent: Optional[str],
    final_score: float,
    lookup_status: str,
    rejection_reason: str,
) -> None:
    qtype = detect_question_type(question)
    print("=" * 90)
    print(f"Original question: {question}")
    print(f"Detected question type: {qtype}")
    print("Top 5 matches:")
    for item in ranked:
        print(f"  Rank {item['rank']}")
        print(f"    Score: {item['score']}")
        print(f"    Intent: {item['intent']}")
        print(f"    Slots: {json.dumps(item['slots'], ensure_ascii=False)}")
        print(f"    Input text: {item['input_text']}")
        print(f"    Target value: {item['target_value']}")
        print(f"    Semantic guard passed: {str(item['semantic_guard_passed']).lower()}")
        if item["rejection_reason"]:
            print(f"    Rejection reason: {item['rejection_reason']}")
    print("Final selected answer:", final_answer if final_answer else "[no confident match]")
    print("Final selected intent:", final_intent if final_intent else "None")
    print("Final selected score:", round(final_score, 6))
    print("Final lookup_status:", lookup_status)
    if rejection_reason:
        print("Rejection reason:", rejection_reason)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer-based semantic lookup ranking experiment.")
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH, help="JSONL dataset file or directory.")
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Question to test. Use multiple --query flags to run multiple questions.",
    )
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Top matches to print.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Minimum score for final selection.")
    args = parser.parse_args()

    rows = load_rows(args.data_path)
    if not rows:
        raise SystemExit(f"No usable rows found under: {args.data_path}")

    model_kind, model_obj = load_model()
    candidate_texts = [build_candidate_text(row) for row in rows]
    candidate_embeddings = encode_texts(model_kind, model_obj, candidate_texts)
    if candidate_embeddings.numel() == 0:
        raise SystemExit("Failed to create candidate embeddings.")

    candidates = [
        Candidate(row=row, candidate_text=text, embedding=candidate_embeddings[idx])
        for idx, (row, text) in enumerate(zip(rows, candidate_texts))
    ]

    queries = [normalize_text(query) for query in args.query if normalize_text(query)]
    if queries:
        print(f"Using custom query list with {len(queries)} question(s).")
    elif sys.stdin.isatty():
        print("Interactive mode. Type a question and press Enter.")
        while True:
            try:
                question = input("Question> ").strip()
            except EOFError:
                break
            if not question:
                break
            queries.append(question)
            break
        if queries:
            print("Using interactive single-question mode.")
    if not queries:
        queries = TEST_QUESTIONS
        print("Using built-in test question set.")

    for query in queries:
        question_embedding = encode_texts(model_kind, model_obj, [query])
        ranked = rank_rows(query, question_embedding[0], candidates, args.top_k, args.threshold)
        final_answer, final_intent, final_score, lookup_status, rejection_reason = choose_final_answer(
            query, ranked, args.threshold
        )
        print_question_report(
            query,
            ranked,
            final_answer,
            final_intent,
            final_score,
            lookup_status,
            rejection_reason,
        )


if __name__ == "__main__":
    main()
