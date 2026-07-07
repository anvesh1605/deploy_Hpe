from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for train_unified_bilstm_v2.py.") from exc


ROOT = Path(__file__).resolve().parent
DEFAULT_RELEASE_DATA_DIR = ROOT / "Data" / "Release_Notes"
DEFAULT_PRODUCT_DATA_DIR = ROOT / "Data" / "product_docs_final_repaired"
DEFAULT_MERGED_2040_PATH = ROOT / "imporved_data_addition" / "aruba_aoscx_bilstm_balanced_2040_merged.jsonl"
DEFAULT_RELEASE_EVAL_PATH = ROOT / "outputs_release_lstm" / "lookup_eval.jsonl"
DEFAULT_PRODUCT_EVAL_PATH = ROOT / "outputs_product_question_tests" / "good_product_questions_30.jsonl"
DEFAULT_BROADER_EVAL_PATH = ROOT / "test_14_results_latest.jsonl"
DEFAULT_MODEL_DIR = ROOT / "models" / "bilstm_unified_v2"
DEFAULT_OUTPUT_DIR = ROOT / "outputs_unified_lstm_v2"

DEFAULT_EMBEDDING_DIM = 192
DEFAULT_HIDDEN_SIZE = 192
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.3
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 6
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42
DEFAULT_MAX_LENGTH = 96
DEFAULT_PATIENCE = 2
DEFAULT_NUM_WORKERS = 0

RELEASE_LABEL_ORDER = [
    "bug_category",
    "bug_scenario",
    "bug_symptom",
    "bug_workaround",
    "release_caveat",
    "data_not_available",
    "out_of_domain",
]

PRODUCT_LABEL_ORDER = [
    "cli_meaning",
    "cli_syntax",
    "concept_explanation",
    "configuration_procedure",
    "event_log_meaning",
    "product_caveat",
    "product_generic",
    "product_limitation",
    "product_requirement",
    "show_command_meaning",
    "show_command_syntax",
    "snmp_mib_info",
    "troubleshooting",
]

NEW_LABEL_ORDER = [
    "cli_output",
    "version_support",
    "support_matrix",
    "capacity_or_scale",
    "limitation",
    "requirement",
    "event_id_meaning",
    "event_id_action",
]

UNIFIED_LABELS = RELEASE_LABEL_ORDER + PRODUCT_LABEL_ORDER + NEW_LABEL_ORDER

RELEASE_NEGATIVE_ROWS = [
    {
        "input_text": "what is my name?",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba switch assistant, so I cannot answer this question because it is not related to Aruba switches.",
        "reference": "This is a domain-specific Aruba switch assistant, so I cannot answer this question because it is not related to Aruba switches.",
    },
    {
        "input_text": "tell me a joke",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba switch assistant, so I cannot answer this question because it is not related to Aruba switches.",
        "reference": "This is a domain-specific Aruba switch assistant, so I cannot answer this question because it is not related to Aruba switches.",
    },
    {
        "input_text": "what is the weather today?",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba switch assistant, so I cannot answer this question because it is not related to Aruba switches.",
        "reference": "This is a domain-specific Aruba switch assistant, so I cannot answer this question because it is not related to Aruba switches.",
    },
    {
        "input_text": "what is 2 plus 2?",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba switch assistant, so I cannot answer this question because it is not related to Aruba switches.",
        "reference": "This is a domain-specific Aruba switch assistant, so I cannot answer this question because it is not related to Aruba switches.",
    },
    {
        "input_text": "For 9999 AOS-CX 10.18, what caveat is documented for SNMP?",
        "intent": "data_not_available",
        "slots": {"switch": "9999", "version": "10_18", "feature": "SNMP"},
        "target_value": "This particular data is not available in the current Aruba switch dataset.",
        "reference": "This particular data is not available in the current Aruba switch dataset.",
    },
    {
        "input_text": "For 4100i AOS-CX 10.99, what is the workaround for Bug 123456?",
        "intent": "data_not_available",
        "slots": {"switch": "4100i", "version": "10_99", "bug_id": "123456"},
        "target_value": "This particular data is not available in the current Aruba switch dataset.",
        "reference": "This particular data is not available in the current Aruba switch dataset.",
    },
    {
        "input_text": "For 6200 AOS-CX 10.18, what product documentation command explains SNMP?",
        "intent": "data_not_available",
        "slots": {"switch": "6200", "version": "10_18", "feature": "SNMP"},
        "target_value": "This particular data is not available in the current Aruba switch dataset.",
        "reference": "This particular data is not available in the current Aruba switch dataset.",
    },
]

PRODUCT_NEGATIVE_ROWS = [
    {
        "input_text": "what is my name?",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
        "reference": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
    },
    {
        "input_text": "tell me a joke",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
        "reference": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
    },
    {
        "input_text": "what is the weather today?",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
        "reference": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
    },
    {
        "input_text": "what is 2 plus 2?",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
        "reference": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
    },
    {
        "input_text": "For 9999 AOS-CX 10.18, what CLI syntax is documented for SNMP?",
        "intent": "data_not_available",
        "slots": {"switch": "9999", "version": "10_18", "feature": "SNMP"},
        "target_value": "This particular data is not available in the current Aruba product documentation dataset.",
        "reference": "This particular data is not available in the current Aruba product documentation dataset.",
    },
    {
        "input_text": "For 4100i AOS-CX 10.99, what is the REST API usage for a missing feature?",
        "intent": "data_not_available",
        "slots": {"switch": "4100i", "version": "10_99", "feature": "REST"},
        "target_value": "This particular data is not available in the current Aruba product documentation dataset.",
        "reference": "This particular data is not available in the current Aruba product documentation dataset.",
    },
    {
        "input_text": "For 6200 AOS-CX 10.18, what product documentation command explains SNMP?",
        "intent": "data_not_available",
        "slots": {"switch": "6200", "version": "10_18", "feature": "SNMP"},
        "target_value": "This particular data is not available in the current Aruba product documentation dataset.",
        "reference": "This particular data is not available in the current Aruba product documentation dataset.",
    },
]


@dataclass
class EvalSample:
    question: str
    gold_label: str
    predicted_label: str
    confidence: float
    correct: bool
    source: str


def normalize_whitespace(text: object) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dotted_version(version: str, sub_version: str) -> str:
    return f"{version.replace('_', '.')}.{sub_version}"


def clean_question_text(question: str, switch: str, version: str, sub_version: str) -> str:
    text = normalize_whitespace(question)
    if not (switch and version and sub_version):
        return text
    version_tag = dotted_version(version, sub_version)
    repeated_prefix = re.compile(
        rf"^(?:For\s+{re.escape(switch)}\s+AOS-CX\s+{re.escape(version_tag)},\s*)+",
        flags=re.IGNORECASE,
    )
    replacement = f"For {switch} AOS-CX {version_tag}, "
    text = repeated_prefix.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def extract_message_question(row: Dict[str, object]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    first = messages[0]
    if not isinstance(first, dict):
        return ""
    return normalize_whitespace(first.get("content", ""))


def extract_message_answer(row: Dict[str, object]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return ""
    second = messages[1]
    if not isinstance(second, dict):
        return ""
    return normalize_whitespace(second.get("content", ""))


def infer_path_context(path: Path) -> Dict[str, str]:
    text = str(path)
    patterns = [
        re.compile(
            r"(?P<switch>[^\\/]+)[\\/](?P=switch)[\\/](?P<version>\d+_\d+)[\\/](?P<sub_version>\d+)[\\/][^\\/]+\.jsonl$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?P<switch>[^\\/]+)[\\/](?P<version>\d+_\d+)[\\/](?P<sub_version>\d+)[\\/][^\\/]+\.jsonl$",
            flags=re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return {
                key: normalize_whitespace(value)
                for key, value in match.groupdict().items()
                if normalize_whitespace(value)
            }
    return {}


def normalize_question_text(question: str, context: Dict[str, str]) -> str:
    switch = context.get("switch", "")
    version = context.get("version", "")
    sub_version = context.get("sub_version", "")
    if switch and version and sub_version:
        return clean_question_text(question, switch, version, sub_version)
    return normalize_whitespace(question)


def infer_release_intent(row: Dict[str, object], question: str) -> Optional[str]:
    intent = normalize_whitespace(row.get("intent", ""))
    if intent in RELEASE_LABEL_ORDER:
        return intent
    source_type = str(row.get("source_type", "")).lower()
    q = question.lower()
    if "release_notes_resolved_issues" in source_type:
        if "category" in q:
            return "bug_category"
        if "symptom" in q or "what issue was resolved" in q or "what issue occurs" in q:
            return "bug_symptom"
        if "scenario" in q or "under what scenario" in q:
            return "bug_scenario"
        if "workaround" in q or "how do i fix" in q or "how to fix" in q:
            return "bug_workaround"
        return None
    if "release_notes_caveats" in source_type:
        return "release_caveat"
    return None


def build_release_target_value(row: Dict[str, object], intent: str) -> str:
    answer = normalize_whitespace(row.get("target_value", "")) or normalize_whitespace(row.get("reference", ""))
    if answer:
        return answer
    answer = extract_message_answer(row)
    if answer:
        return answer
    description = normalize_whitespace(row.get("description", ""))
    if description:
        return description
    if intent == "bug_category":
        return normalize_whitespace(row.get("category", ""))
    return ""


def build_release_slots(row: Dict[str, object], intent: str) -> Dict[str, str]:
    slots = {
        "switch": normalize_whitespace(row.get("switch", "")),
        "version": normalize_whitespace(row.get("version", "")),
        "sub_version": normalize_whitespace(row.get("sub_version", "")),
    }
    if intent == "release_caveat":
        slots["feature"] = normalize_whitespace(row.get("feature", ""))
    else:
        slots["bug_id"] = normalize_whitespace(row.get("bug_id", ""))
        slots["category"] = normalize_whitespace(row.get("category", ""))
    return {key: value for key, value in slots.items() if value}


def release_row_to_record(
    row: Dict[str, object],
    source_file: str,
    line_no: int,
    context: Dict[str, str],
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    if not isinstance(row, dict):
        return None, "invalid row"

    question = extract_message_question(row) or normalize_whitespace(row.get("input_text", ""))
    if not question:
        return None, "empty input_text"

    intent = infer_release_intent(row, question)
    if not intent:
        return None, "intent not selected"

    answer = build_release_target_value(row, intent)
    if not answer:
        return None, "empty target_value"

    slots = row.get("slots") if isinstance(row.get("slots"), dict) else {}
    if not slots:
        slots = build_release_slots(row, intent)
    slots = {key: normalize_whitespace(value) for key, value in dict(slots).items() if normalize_whitespace(value)}
    for key in ("switch", "version", "sub_version"):
        if context.get(key) and not slots.get(key):
            slots[key] = context[key]

    return {
        "input_text": normalize_question_text(question, context),
        "intent": intent,
        "slots": slots,
        "target_value": answer,
        "reference": answer,
        "source_file": source_file,
        "line_no": line_no,
        "source_family": "release",
    }, None


def product_row_to_record(
    row: Dict[str, object],
    source_file: str,
    line_no: int,
    context: Dict[str, str],
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    if not isinstance(row, dict):
        return None, "invalid row"

    intent = normalize_whitespace(row.get("intent", ""))
    input_text = normalize_whitespace(row.get("input_text", ""))
    target_value = normalize_whitespace(row.get("target_value", "")) or normalize_whitespace(row.get("reference", ""))
    question = extract_message_question(row) or input_text
    if not question:
        return None, "empty input_text"
    if not intent and row.get("source_type"):
        intent = normalize_whitespace(row.get("source_type", ""))
    if intent not in UNIFIED_LABELS:
        return None, "intent not selected"
    if not target_value:
        target_value = extract_message_answer(row)
    if not target_value:
        return None, "empty target_value"

    slots = row.get("slots") if isinstance(row.get("slots"), dict) else {}
    slots = {key: normalize_whitespace(value) for key, value in dict(slots).items() if normalize_whitespace(value)}
    for key in ("switch", "version", "sub_version"):
        if context.get(key) and not slots.get(key):
            slots[key] = context[key]

    return {
        "input_text": normalize_question_text(question, context),
        "intent": intent,
        "slots": slots,
        "target_value": target_value,
        "reference": target_value,
        "source_file": source_file,
        "line_no": line_no,
        "source_family": "product",
    }, None


def merged_2040_row_to_record(
    row: Dict[str, object],
    source_file: str,
    line_no: int,
    context: Dict[str, str],
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    if not isinstance(row, dict):
        return None, "invalid row"
    intent = normalize_whitespace(row.get("intent", ""))
    input_text = normalize_whitespace(row.get("input_text", ""))
    target_value = normalize_whitespace(row.get("target_value", "")) or normalize_whitespace(row.get("reference", ""))
    if not input_text:
        return None, "empty input_text"
    if intent not in UNIFIED_LABELS:
        return None, "intent not selected"
    if not target_value:
        return None, "empty target_value"
    slots = row.get("slots") if isinstance(row.get("slots"), dict) else {}
    slots = {key: normalize_whitespace(value) for key, value in dict(slots).items() if normalize_whitespace(value)}
    for key in ("switch", "version", "sub_version"):
        if context.get(key) and not slots.get(key):
            slots[key] = context[key]
    return {
        "input_text": normalize_question_text(input_text, context),
        "intent": intent,
        "slots": slots,
        "target_value": target_value,
        "reference": target_value,
        "source_file": source_file,
        "line_no": line_no,
        "source_family": "merged_2040",
    }, None


def collect_records(
    paths: Sequence[Path],
    parser,
) -> Tuple[List[Dict[str, object]], Dict[str, int], int]:
    records: List[Dict[str, object]] = []
    reason_counts: Counter[str] = Counter()
    rows_scanned = 0
    seen = set()

    for path in paths:
        if path.is_file():
            candidate_files = [path]
        else:
            candidate_files = [item for item in sorted(path.rglob("*.jsonl")) if item.is_file()]
        for file_path in candidate_files:
            context = infer_path_context(file_path)
            with file_path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    raw = line.strip()
                    if not raw:
                        continue
                    rows_scanned += 1
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError:
                        reason_counts["invalid_jsonl"] += 1
                        continue
                    record, reason = parser(row, str(file_path), line_no, context)
                    if reason is not None:
                        reason_counts[reason] += 1
                        continue
                    key = (
                        normalize_whitespace(record["input_text"]),
                        normalize_whitespace(record["intent"]),
                        normalize_whitespace(record["target_value"]),
                    )
                    if key in seen:
                        reason_counts["duplicate"] += 1
                        continue
                    seen.add(key)
                    records.append(record)

    return records, dict(reason_counts), rows_scanned


def add_negative_samples(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    augmented = list(records)
    augmented.extend(deepcopy(RELEASE_NEGATIVE_ROWS))
    augmented.extend(deepcopy(PRODUCT_NEGATIVE_ROWS))
    return augmented


def stratified_split(
    records: Sequence[Dict[str, object]],
    seed: int,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    rng = random.Random(seed)
    by_intent: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        by_intent[str(record["intent"])].append(dict(record))

    train: List[Dict[str, object]] = []
    val: List[Dict[str, object]] = []
    test: List[Dict[str, object]] = []

    for intent in UNIFIED_LABELS:
        group = by_intent.get(intent, [])
        rng.shuffle(group)
        n = len(group)
        if n == 0:
            continue
        if n == 1:
            train.extend(group)
            continue
        if n == 2:
            train.append(group[0])
            test.append(group[1])
            continue

        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        n_train = n - n_test - n_val
        while n_train < 1 and (n_val > 1 or n_test > 1):
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            n_train = n - n_test - n_val
        if n_train < 1:
            n_train = 1
        overflow = n_train + n_val + n_test - n
        while overflow > 0 and n_val > 1:
            n_val -= 1
            overflow -= 1
        while overflow > 0 and n_test > 1:
            n_test -= 1
            overflow -= 1
        while overflow > 0 and n_train > 1:
            n_train -= 1
            overflow -= 1

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


class SimpleTokenizer:
    def __init__(self, vocab: Optional[Dict[str, int]] = None) -> None:
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        if vocab is None:
            self.vocab = {self.pad_token: 0, self.unk_token: 1}
        else:
            self.vocab = dict(vocab)
            self.vocab.setdefault(self.pad_token, 0)
            self.vocab.setdefault(self.unk_token, 1)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+|[^\w\s]", text.lower())

    def build_vocab(self, texts: Sequence[str]) -> None:
        next_id = max(self.vocab.values(), default=1) + 1
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = next_id
                    next_id += 1

    def encode(self, text: str, max_length: int) -> List[int]:
        tokens = self.tokenize(text)[:max_length]
        if not tokens:
            tokens = [self.unk_token]
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]


class IntentDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Dict[str, object]],
        tokenizer: SimpleTokenizer,
        label_to_id: Dict[str, int],
        max_length: int,
    ) -> None:
        self.items = list(items)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = self.items[index]
        input_ids = self.tokenizer.encode(str(item["input_text"]), self.max_length)
        label_id = self.label_to_id[str(item["intent"])]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "length": torch.tensor(len(input_ids), dtype=torch.long),
            "label": torch.tensor(label_id, dtype=torch.long),
        }


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    input_ids = []
    labels = []
    for item in batch:
        ids = item["input_ids"][:max_len]
        if ids.numel() < max_len:
            padding = torch.zeros(max_len - ids.numel(), dtype=torch.long)
            ids = torch.cat([ids, padding], dim=0)
        input_ids.append(ids)
        labels.append(item["label"])
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "lengths": lengths,
        "labels": torch.stack(labels, dim=0),
    }


def build_loader(
    records: Sequence[Dict[str, object]],
    tokenizer: SimpleTokenizer,
    label_to_id: Dict[str, int],
    max_length: int,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    dataset = IntentDataset(records, tokenizer, label_to_id, max_length)
    effective_workers = 0 if os.name == "nt" else num_workers
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_batch,
        "num_workers": effective_workers,
        "pin_memory": device.type == "cuda",
    }
    if effective_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


class LSTMIntentModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        num_labels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        features = torch.cat([forward_hidden, backward_hidden], dim=1)
        features = self.dropout(features)
        return self.classifier(features)


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str],
) -> Tuple[Dict[str, object], List[List[int]]]:
    num_labels = len(label_names)
    matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    correct = sum(matrix[i][i] for i in range(num_labels))
    total = len(y_true)

    for idx, name in enumerate(label_names):
        tp = matrix[idx][idx]
        fp = sum(matrix[row][idx] for row in range(num_labels) if row != idx)
        fn = sum(matrix[idx][col] for col in range(num_labels) if col != idx)
        support = sum(matrix[idx])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_precision = sum(item["precision"] for item in per_class.values()) / max(1, num_labels)
    macro_recall = sum(item["recall"] for item in per_class.values()) / max(1, num_labels)
    macro_f1 = sum(item["f1"] for item in per_class.values()) / max(1, num_labels)
    accuracy = correct / max(1, total)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "total": total,
    }, matrix


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    label_names: Sequence[str],
    device: torch.device,
) -> Tuple[float, Dict[str, object], List[List[int]], List[int], List[int], List[float]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    confidences: List[float] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            confidences.extend(probs.max(dim=1).values.cpu().tolist())
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size

    avg_loss = total_loss / max(1, total_items)
    report, matrix = classification_report(y_true, y_pred, label_names)
    return avg_loss, report, matrix, y_true, y_pred, confidences


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
    return total_loss / max(1, total_items)


def predict_with_confidence(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[float]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    confidences: List[float] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, lengths)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            confidences.extend(probs.max(dim=1).values.cpu().tolist())
    return y_true, y_pred, confidences


def build_sample_markdown(samples: Sequence[EvalSample]) -> str:
    if not samples:
        return "# Samples Eval\n\nNo samples available."
    best = sorted(samples, key=lambda item: (item.correct, item.confidence), reverse=True)[:10]
    worst = sorted(samples, key=lambda item: (item.correct, item.confidence))[:10]

    def render_block(title: str, rows: Sequence[EvalSample]) -> List[str]:
        lines = [f"## {title}"]
        if not rows:
            lines.append("No samples available.")
            return lines
        for item in rows:
            lines.extend(
                [
                    f"- source: `{item.source}`",
                    f"  - question: {item.question}",
                    f"  - gold: {item.gold_label}",
                    f"  - prediction: {item.predicted_label}",
                    f"  - confidence: {item.confidence:.4f}",
                    f"  - correct: {str(item.correct).lower()}",
                ]
            )
        return lines

    parts = ["# Samples Eval", ""]
    parts.extend(render_block("10 Best Predictions", best))
    parts.append("")
    parts.extend(render_block("10 Worst Predictions", worst))
    return "\n".join(parts).rstrip() + "\n"


def save_training_metrics(path: Path, history: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return
    headers = list(history[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(history)


def load_external_eval_rows(path: Path, label_field_candidates: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    for row in read_jsonl(path):
        question = normalize_whitespace(row.get("question", "")) or normalize_whitespace(row.get("input_text", ""))
        if not question:
            continue
        label = ""
        for field in label_field_candidates:
            label = normalize_whitespace(row.get(field, ""))
            if label:
                break
        if not label:
            continue
        rows.append(
            {
                "question": question,
                "label": label,
                "source": path.name,
            }
        )
    return rows


def load_broad_questions(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    for row in read_jsonl(path):
        question = normalize_whitespace(row.get("question", ""))
        if not question:
            continue
        rows.append(
            {
                "question": question,
                "label": normalize_whitespace(row.get("predicted_intent", "")),
                "source": path.name,
            }
        )
    return rows


def predict_question(model: nn.Module, tokenizer: SimpleTokenizer, label_names: Sequence[str], question: str, max_length: int, device: torch.device) -> Tuple[str, float]:
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(question, max_length)], dtype=torch.long, device=device)
    lengths = torch.tensor([min(len(tokenizer.encode(question, max_length)), max_length)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids, lengths)
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs).item())
        return label_names[pred_id], float(probs[pred_id].item())


def evaluate_question_rows(
    model: nn.Module,
    tokenizer: SimpleTokenizer,
    label_names: Sequence[str],
    rows: Sequence[Dict[str, str]],
    max_length: int,
    device: torch.device,
    source_name: str,
) -> Tuple[Dict[str, object], List[EvalSample], List[Dict[str, object]]]:
    samples: List[EvalSample] = []
    jsonl_rows: List[Dict[str, object]] = []
    correct = 0
    for row in rows:
        question = row["question"]
        gold = row["label"]
        pred, confidence = predict_question(model, tokenizer, label_names, question, max_length, device)
        is_correct = bool(gold) and pred == gold
        correct += int(is_correct)
        sample = EvalSample(
            question=question,
            gold_label=gold or "n/a",
            predicted_label=pred,
            confidence=confidence,
            correct=is_correct,
            source=source_name,
        )
        samples.append(sample)
        jsonl_rows.append(
            {
                "question": question,
                "gold_label": gold or None,
                "predicted_label": pred,
                "confidence": confidence,
                "correct": is_correct if gold else None,
                "source": source_name,
            }
        )

    total = len(rows)
    report = {
        "source": source_name,
        "count": total,
        "accuracy": (correct / total) if total else 0.0,
        "samples": [
            {
                "question": item.question,
                "gold_label": item.gold_label,
                "predicted_label": item.predicted_label,
                "confidence": item.confidence,
                "correct": item.correct,
            }
            for item in samples[:20]
        ],
    }
    return report, samples, jsonl_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a unified product + release BiLSTM classifier.")
    parser.add_argument("--release_data_dir", type=Path, default=DEFAULT_RELEASE_DATA_DIR)
    parser.add_argument("--product_data_dir", type=Path, default=DEFAULT_PRODUCT_DATA_DIR)
    parser.add_argument("--merged_2040_path", type=Path, default=DEFAULT_MERGED_2040_PATH)
    parser.add_argument("--release_eval_path", type=Path, default=DEFAULT_RELEASE_EVAL_PATH)
    parser.add_argument("--product_eval_path", type=Path, default=DEFAULT_PRODUCT_EVAL_PATH)
    parser.add_argument("--broader_eval_path", type=Path, default=DEFAULT_BROADER_EVAL_PATH)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding_dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "n/a"
    print(f"[DEVICE] {device.type}")
    print(f"[GPU] {gpu_name}")
    print(f"[LABELS] {len(UNIFIED_LABELS)} labels")

    if not args.release_data_dir.exists():
        raise FileNotFoundError(f"Release data directory not found: {args.release_data_dir}")
    if not args.product_data_dir.exists():
        raise FileNotFoundError(f"Product data directory not found: {args.product_data_dir}")
    if not args.merged_2040_path.exists():
        raise FileNotFoundError(f"Merged 2040 data file not found: {args.merged_2040_path}")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    release_records, release_reason_counts, release_scanned = collect_records([args.release_data_dir], release_row_to_record)
    product_records, product_reason_counts, product_scanned = collect_records([args.product_data_dir], product_row_to_record)
    merged_records, merged_reason_counts, merged_scanned = collect_records([args.merged_2040_path], merged_2040_row_to_record)

    release_records = [record for record in release_records if record["intent"] in UNIFIED_LABELS]
    product_records = [record for record in product_records if record["intent"] in UNIFIED_LABELS]
    merged_records = [record for record in merged_records if record["intent"] in UNIFIED_LABELS]

    print(f"[DATA] release rows scanned: {release_scanned}")
    print(f"[DATA] release rows kept: {len(release_records)}")
    print(f"[DATA] release filter reasons: {release_reason_counts}")
    print(f"[DATA] product rows scanned: {product_scanned}")
    print(f"[DATA] product rows kept: {len(product_records)}")
    print(f"[DATA] product filter reasons: {product_reason_counts}")
    print(f"[DATA] merged 2040 rows scanned: {merged_scanned}")
    print(f"[DATA] merged 2040 rows kept: {len(merged_records)}")
    print(f"[DATA] merged 2040 filter reasons: {merged_reason_counts}")

    merged_train, merged_val, merged_test = stratified_split(merged_records, args.seed, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    training_pool = list(release_records) + list(product_records) + list(merged_train)
    training_pool.extend(add_negative_samples([]))

    # Merge in source-level negative samples while keeping the 2040 validation split untouched.
    training_pool = add_negative_samples(training_pool)

    if not training_pool:
        raise ValueError("No clean training rows were found.")

    train_records, val_records, test_records = stratified_split(training_pool, args.seed)
    if not train_records or not val_records:
        raise ValueError("Failed to produce a usable train/validation split.")

    label_names = [label for label in UNIFIED_LABELS if any(str(record["intent"]) == label for record in training_pool)]
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    id_to_label = {str(idx): label for label, idx in label_to_id.items()}

    print(f"[DATA] training pool size: {len(training_pool)}")
    print(f"[DATA] label names: {label_names}")
    print(f"[DATA] split sizes: train={len(train_records)} val={len(val_records)} test={len(test_records)}")
    print(f"[DATA] merged holdout sizes: val={len(merged_val)} test={len(merged_test)}")

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab([str(item["input_text"]) for item in train_records])

    train_loader = build_loader(train_records, tokenizer, label_to_id, args.max_length, args.batch_size, True, device, args.num_workers if device.type == "cuda" else 0)
    val_loader = build_loader(val_records, tokenizer, label_to_id, args.max_length, args.batch_size, False, device, args.num_workers if device.type == "cuda" else 0)
    test_loader = build_loader(test_records, tokenizer, label_to_id, args.max_length, args.batch_size, False, device, args.num_workers if device.type == "cuda" else 0)

    train_counts = Counter(str(record["intent"]) for record in train_records)
    total_train = sum(train_counts.values())
    class_weights = []
    for label in label_names:
        count = max(1, train_counts.get(label, 0))
        class_weights.append(total_train / (len(label_names) * count))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    model = LSTMIntentModel(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_labels=len(label_names),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # Use the non-foreach AdamW path to reduce peak memory pressure on 8 GB GPUs.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, foreach=False)

    best_state = None
    best_val_macro_f1 = -math.inf
    best_epoch = 0
    best_val_report: Dict[str, object] = {}
    best_val_matrix: List[List[int]] = []
    history: List[Dict[str, object]] = []
    patience = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_report, val_matrix, _, _, _ = evaluate(model, val_loader, criterion, label_names, device)
        macro_f1 = float(val_report["macro_f1"])
        val_accuracy = float(val_report["accuracy"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "macro_precision": float(val_report["macro_precision"]),
                "macro_recall": float(val_report["macro_recall"]),
                "macro_f1": macro_f1,
            }
        )

        print(
            f"[EPOCH {epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f} macro_f1={macro_f1:.4f}"
        )
        print(f"[EPOCH {epoch:02d}] per_class_f1={ {label: round(float(val_report['per_class'][label]['f1']), 4) for label in label_names} }")

        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_val_report = val_report
            best_val_matrix = val_matrix
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[EARLY_STOP] no validation Macro F1 improvement for {args.patience} epochs")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model.")

    model.load_state_dict(best_state)

    train_loader_eval = build_loader(train_records, tokenizer, label_to_id, args.max_length, args.batch_size, False, device, args.num_workers if device.type == "cuda" else 0)
    test_loader_eval = build_loader(test_records, tokenizer, label_to_id, args.max_length, args.batch_size, False, device, args.num_workers if device.type == "cuda" else 0)
    train_loss, train_report, _, _, _, _ = evaluate(model, train_loader_eval, criterion, label_names, device)
    val_loss, val_report, _, _, _, _ = evaluate(model, val_loader, criterion, label_names, device)
    test_loss, test_report, test_matrix, _, _, _ = evaluate(model, test_loader_eval, criterion, label_names, device)

    model_path = args.model_dir / "best_model.pt"
    vocab_path = args.model_dir / "vocab_unified_v2.json"
    label_map_path = args.model_dir / "label_map_unified_v2.json"
    training_report_path = args.output_dir / "training_report_unified_v2.json"
    evaluation_report_path = args.output_dir / "evaluation_report_unified_v2.json"
    metrics_csv_path = args.output_dir / "training_metrics_unified_v2.csv"
    confusion_matrix_path = args.output_dir / "confusion_matrix_unified_v2.json"
    samples_path = args.output_dir / "samples_eval_unified_v2.md"

    torch.save(
        {
            "model_state_dict": best_state,
            "config": {
                "model_type": "bilstm",
                "embedding_dim": args.embedding_dim,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "max_length": args.max_length,
                "label_names": label_names,
            },
            "vocab": dict(tokenizer.vocab),
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
        },
        model_path,
    )

    write_json(vocab_path, dict(tokenizer.vocab))
    write_json(
        label_map_path,
        {
            "label_names": label_names,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "label_groups": {
                "release": RELEASE_LABEL_ORDER,
                "product": PRODUCT_LABEL_ORDER,
                "new": NEW_LABEL_ORDER,
            },
        },
    )
    save_training_metrics(metrics_csv_path, history)
    write_json(
        confusion_matrix_path,
        {
            "labels": label_names,
            "matrix": best_val_matrix,
        },
    )

    val_samples: List[EvalSample] = []
    val_true, val_pred, val_conf = predict_with_confidence(model, val_loader, device)
    for record, true_id, pred_id, conf in zip(val_records, val_true, val_pred, val_conf):
        val_samples.append(
            EvalSample(
                question=str(record["input_text"]),
                gold_label=label_names[true_id],
                predicted_label=label_names[pred_id],
                confidence=float(conf),
                correct=bool(true_id == pred_id),
                source="internal_validation",
            )
        )
    samples_path.write_text(build_sample_markdown(val_samples), encoding="utf-8")

    training_report = {
        "data_sources": {
            "release_data_dir": str(args.release_data_dir),
            "product_data_dir": str(args.product_data_dir),
            "merged_2040_path": str(args.merged_2040_path),
        },
        "rows_scanned": {
            "release": release_scanned,
            "product": product_scanned,
            "merged_2040": merged_scanned,
        },
        "rows_kept": {
            "release": len(release_records),
            "product": len(product_records),
            "merged_2040": len(merged_records),
            "train_pool": len(training_pool),
        },
        "split_sizes": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
            "merged_2040_train": len(merged_train),
            "merged_2040_val": len(merged_val),
            "merged_2040_test": len(merged_test),
        },
        "filter_reasons": {
            "release": release_reason_counts,
            "product": product_reason_counts,
            "merged_2040": merged_reason_counts,
        },
        "label_names": label_names,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "validation_metrics": best_val_report,
        "train_metrics": train_report,
        "test_metrics": test_report,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "history": history,
        "device": device.type,
        "gpu_name": gpu_name,
        "artifacts": {
            "model": str(model_path),
            "vocab": str(vocab_path),
            "label_map": str(label_map_path),
            "training_report": str(training_report_path),
            "evaluation_report": str(evaluation_report_path),
            "metrics_csv": str(metrics_csv_path),
            "confusion_matrix": str(confusion_matrix_path),
            "samples_eval": str(samples_path),
        },
        "verdict": "strong" if best_val_macro_f1 >= 0.90 else "good" if best_val_macro_f1 >= 0.80 else "needs review",
    }
    write_json(training_report_path, training_report)

    release_eval_rows = load_external_eval_rows(args.release_eval_path, ("gold_intent", "intent", "predicted_intent"))
    product_eval_rows = load_external_eval_rows(args.product_eval_path, ("intent", "gold_intent", "predicted_intent"))
    broader_eval_rows = load_broad_questions(args.broader_eval_path)

    release_report, release_samples, release_jsonl = evaluate_question_rows(
        model, tokenizer, label_names, release_eval_rows, args.max_length, device, args.release_eval_path.name
    )
    product_report, product_samples, product_jsonl = evaluate_question_rows(
        model, tokenizer, label_names, product_eval_rows, args.max_length, device, args.product_eval_path.name
    )
    broader_report, broader_samples, broader_jsonl = evaluate_question_rows(
        model, tokenizer, label_names, broader_eval_rows, args.max_length, device, args.broader_eval_path.name
    )
    merged_val_rows = [
        {
            "question": str(record["input_text"]),
            "label": str(record["intent"]),
            "source": "merged_2040_val",
        }
        for record in merged_val
    ]
    merged_report, merged_samples, merged_jsonl = evaluate_question_rows(
        model, tokenizer, label_names, merged_val_rows, args.max_length, device, "merged_2040_val"
    )

    external_eval_dir = args.output_dir / "external_eval"
    external_eval_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(external_eval_dir / "release_eval_predictions.jsonl", release_jsonl)
    write_jsonl(external_eval_dir / "product_eval_predictions.jsonl", product_jsonl)
    write_jsonl(external_eval_dir / "broader_14_predictions.jsonl", broader_jsonl)
    write_jsonl(external_eval_dir / "merged_2040_val_predictions.jsonl", merged_jsonl)

    evaluation_report = {
        "release_eval": release_report,
        "product_eval": product_report,
        "broader_14_eval": broader_report,
        "merged_2040_validation_eval": merged_report,
        "notes": {
            "release_eval": "Gold intent labels are loaded from outputs_release_lstm/lookup_eval.jsonl.",
            "product_eval": "Gold intent labels are loaded from outputs_product_question_tests/good_product_questions_30.jsonl.",
            "broader_14_eval": "Reference labels use the previous file's predicted_intent field when present.",
            "merged_2040_validation_eval": "Held-out validation rows are the 10 percent merged_2040 split from the unified corpus.",
        },
        "samples": {
            "release_eval": [
                {
                    "question": item.question,
                    "gold_label": item.gold_label,
                    "predicted_label": item.predicted_label,
                    "confidence": item.confidence,
                    "correct": item.correct,
                }
                for item in release_samples[:20]
            ],
            "product_eval": [
                {
                    "question": item.question,
                    "gold_label": item.gold_label,
                    "predicted_label": item.predicted_label,
                    "confidence": item.confidence,
                    "correct": item.correct,
                }
                for item in product_samples[:20]
            ],
            "broader_14_eval": [
                {
                    "question": item.question,
                    "gold_label": item.gold_label,
                    "predicted_label": item.predicted_label,
                    "confidence": item.confidence,
                    "correct": item.correct,
                }
                for item in broader_samples[:20]
            ],
            "merged_2040_validation_eval": [
                {
                    "question": item.question,
                    "gold_label": item.gold_label,
                    "predicted_label": item.predicted_label,
                    "confidence": item.confidence,
                    "correct": item.correct,
                }
                for item in merged_samples[:20]
            ],
        },
    }
    write_json(evaluation_report_path, evaluation_report)

    print("Training completed")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Macro F1: {best_val_macro_f1:.4f}")
    print(f"Validation accuracy: {float(best_val_report.get('accuracy', 0.0)):.4f}")
    print(f"Model saved at: {model_path}")
    print(f"Vocab saved at: {vocab_path}")
    print(f"Label map saved at: {label_map_path}")
    print(f"Training report: {training_report_path}")
    print(f"Evaluation report: {evaluation_report_path}")
    print(f"Verdict: {training_report['verdict']}")


if __name__ == "__main__":
    main()
