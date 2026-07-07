from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import (  # noqa: E402
    BACKEND_CACHE_DIR,
    DATA_ROOT,
    MODEL_ROOT,
    QWEN_FINALIZE_ALL_RESPONSES,
    PRODUCT_LOOKUP_DATA_PATHS,
    PRODUCT_DOCS_DATA_DIR,
    PRODUCT_LSTM_MODEL_PATH,
    PRODUCT_LSTM_DATA_DIR,
    OLLAMA_BASE_URL,
    QWEN_MODEL_PATH,
    UNIFIED_LSTM_MODEL_PATH,
    RELEASE_AVAILABILITY_PATH,
    RELEASE_BUG_METADATA_PATH,
    RELEASE_LSTM_DATA_DIR,
    RELEASE_LSTM_MODEL_PATH,
    RELEASE_NOTES_DATA_DIR,
    RELEASE_LOOKUP_DATA_PATH,
    RELEASE_LOOKUP_INDEX_PATH,
)
from live_release_chat import answer_question as answer_release_question  # noqa: E402
from lstm_lookup import (  # noqa: E402
    DATA_NOT_AVAILABLE_RESPONSE,
    build_availability_index,
    build_bug_metadata_index,
    build_lookup_entries,
    build_lookup_index,
    check_data_availability,
    extract_slots_from_question,
    load_or_build_availability_index,
    load_or_build_bug_metadata_index,
    normalize_whitespace,
    read_jsonl,
    write_jsonl,
)
from release_notes_qwen_pipeline import (  # noqa: E402
    format_cli_syntax_answer,
    generate_qwen_answer,
    is_command_purpose_question,
    is_cli_syntax_answer,
    load_lstm_support,
    load_lookup_resources,
    load_qwen_model,
    predict_intent,
    validate_qwen_answer,
)


RELEASE_LIKE_INTENTS = {
    "bug_category",
    "bug_scenario",
    "bug_symptom",
    "bug_workaround",
    "release_caveat",
}

PRODUCT_EXACT_INTENTS = {
    "cli_syntax",
    "cli_output",
    "capacity_or_scale",
    "show_command_syntax",
    "show_command_usage",
    "event_id_meaning",
    "event_id_action",
    "version_date",
    "release_date",
    "out_of_domain",
    "data_not_available",
}

PRODUCT_DATANOT_AVAILABLE_RESPONSE = (
    "This particular data is not available in the current Aruba product documentation dataset."
)

PRODUCT_NOT_FOUND_RESPONSE = "I could not find a matching answer in the current Aruba product documentation dataset."
PRODUCT_COMMAND_OUTPUT_RESPONSE = "I found related documentation, but not a reliable exact output match."
PRODUCT_SYNTAX_MATCH_RESPONSE = "I found related documentation, but not a reliable exact syntax match."
PRODUCT_CONTAMINATED_RESPONSE = (
    "I found related documentation, but the retrieved text looks like an index or navigation artifact, so I cannot safely return it as the final answer."
)
PRODUCT_NEEDS_DISAMBIGUATION_RESPONSE = (
    "Multiple possible answers were found. Please provide more detail such as feature, command, version, or sub-version."
)
PRODUCT_SLOT_MISSING_RESPONSE = "I need more detail to answer this, such as the command, topic, version, or sub-version."

PRODUCT_FOLLOWUP_CONTEXT_MISSING_RESPONSE = (
    "I need more context. Please specify the topic you want me to explain."
)

PRODUCT_FOLLOWUP_WORDS = [
    "above bug",
    "that bug",
    "this bug",
    "same bug",
    "above issue",
    "that issue",
    "this issue",
    "above limitation",
    "that limitation",
    "this limitation",
    "explain those",
    "explain that",
    "explain this",
    "explain more",
    "tell me more",
    "what about that",
    "what about this",
    "what does that mean",
    "those two types",
    "elaborate",
    "explain",
]

PRODUCT_STRICT_QUESTION_TYPES = {
    "support_matrix",
    "version_support",
    "capacity_or_scale",
    "cli_syntax",
    "cli_output",
}

PRODUCT_QUESTION_TYPE_TO_INTENTS = {
    "cli_syntax": ["cli_syntax", "show_command_syntax", "product_generic"],
    "cli_output": ["show_command_meaning", "show_command_usage", "concept_explanation", "product_generic"],
    "support_matrix": ["product_requirement", "product_limitation", "concept_explanation", "product_generic"],
    "version_support": ["product_requirement", "product_limitation", "concept_explanation", "product_generic"],
    "procedure": ["configuration_procedure", "concept_explanation", "product_generic"],
    "capacity_or_scale": ["capacity_or_scale", "product_limitation", "product_requirement", "concept_explanation", "product_generic"],
    "limitation": ["product_limitation", "concept_explanation", "product_generic"],
    "requirement": ["product_requirement", "concept_explanation", "product_generic"],
    "concept_explanation": ["concept_explanation", "product_generic"],
    "generic_product_query": ["concept_explanation", "product_generic"],
}

PRODUCT_QUESTION_TYPE_REQUIRED_SLOTS = {
    "cli_syntax": ["command"],
    "cli_output": ["command"],
    "support_matrix": ["feature"],
    "version_support": ["feature"],
    "capacity_or_scale": ["switch"],
}

PRODUCT_TOPIC_FAMILY_TOPICS = {
    "routing_capacity": ["Static routing", "Routing", "IP Routing", "Route Manager"],
    "issu_support": ["ISSU", "Upgrade", "Software update"],
    "vsf_support": ["VSF", "Virtual Switching", "Stacking"],
    "vsx_procedure": ["VSX", "High availability", "Redundancy", "Management Module Failover Overview"],
    "cli_output": ["Routing", "IP Routing", "Static routing", "CLI Reference"],
    "security": ["Security", "AAA", "REST", "Certificates", "PKI"],
    "monitoring": ["Monitoring", "Diagnostics", "Troubleshooting"],
    "limitation": ["Limitations", "Static routing"],
    "requirement": ["Requirements", "Prerequisites"],
    "concept_explanation": [],
    "generic_product_query": [],
}

PRODUCT_FILLER_PREFIXES = (
    "the documented answer is:",
    "according to the documentation:",
    "according to the guide:",
    "the guide says:",
    "the guide states:",
)

PRODUCT_GENERATED_LABEL_PREFIXES = (
    "concept explanation:",
    "product documentation:",
    "answer:",
    "response:",
    "final answer:",
)

PRODUCT_QWEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "please",
    "provide",
    "such",
    "that",
    "the",
    "to",
    "this",
    "was",
    "were",
    "with",
    "you",
}


def _clean(value: object) -> str:
    return normalize_whitespace(value)


def _lower(value: object) -> str:
    return _clean(value).lower()


def _canonical_product_switch(value: object) -> str:
    text = _clean(value)
    upper = text.upper()
    if upper.startswith("CX") and len(text) > 2 and text[2:].isdigit():
        return text[2:]
    return text


def _unique(values: Sequence[str]) -> List[str]:
    seen: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen


def _dominant_answer(answers: Sequence[str]) -> Tuple[str, int, int]:
    counts: Dict[str, int] = {}
    order: List[str] = []
    for answer in answers:
        cleaned = _clean(answer)
        if not cleaned:
            continue
        if cleaned not in counts:
            order.append(cleaned)
        counts[cleaned] = counts.get(cleaned, 0) + 1
    if not counts:
        return "", 0, 0
    ordered = sorted(counts.items(), key=lambda item: (item[1], -order.index(item[0]) if item[0] in order else 0), reverse=True)
    top_answer, top_count = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0
    return top_answer, top_count, runner_up


def _domain_version(version: str, domain: str) -> str:
    version = _clean(version)
    if not version:
        return ""
    if domain == "product":
        return version.replace("_", ".")
    return version


def _product_version_aliases(version: str, sub_version: str = "") -> List[str]:
    version = _clean(version).replace("_", ".")
    sub_version = _clean(sub_version)
    aliases: List[str] = []
    if version and sub_version:
        combined = version if version.endswith(f".{sub_version}") else f"{version}.{sub_version}"
        aliases.append(combined)
        base_parts = version.split(".")
        if len(base_parts) > 2:
            aliases.append(".".join(base_parts[:2]))
        elif len(base_parts) == 2:
            aliases.append(version)
    elif version:
        aliases.append(version)
        base_parts = version.split(".")
        if len(base_parts) > 2:
            aliases.append(".".join(base_parts[:2]))
    return _unique(aliases)


def _product_primary_slot(slots: Dict[str, str]) -> str:
    for key in ("command", "topic", "feature", "section", "category", "event_id", "question_type"):
        value = _clean(slots.get(key, ""))
        if value:
            return value
    return ""


def _product_command_from_question(question: str) -> str:
    text = _clean(question)
    patterns = [
        r"\b(?:what\s+is\s+the\s+output\s+of\s+(?:the\s+)?|output\s+of\s+(?:the\s+)?|show\s+output\s+of\s+(?:the\s+)?|show\s+output\s+for\s+(?:the\s+)?)(?P<command>show\s+.+?)(?:\?|$)",
        r"\b(?:what\s+is\s+the\s+cli\s+syntax\s+for\s+(?:the\s+)?|cli\s+syntax\s+for\s+(?:the\s+)?)(?P<command>.+?)(?:\s+in\s+aos-cx\b|\s+on\s+aos-cx\b|\s+command\b|\?|$)",
        r"\b(?:what\s+command\s+syntax\s+is\s+listed\s+for\s+(?:the\s+)?|what\s+syntax\s+is\s+listed\s+for\s+(?:the\s+)?|what\s+command\s+syntax\s+is\s+documented\s+for\s+(?:the\s+)?|what\s+syntax\s+is\s+documented\s+for\s+(?:the\s+)?)(?P<command>.+?)(?:\s+on\s+\S+|\s+in\s+aos-cx\b|\s+command\b|\?|$)",
        r"\b(?:return\s+the\s+documented\s+syntax\s+for\s+(?:the\s+)?|give\s+the\s+exact\s+syntax\s+of\s+(?:the\s+)?|give\s+the\s+documented\s+syntax\s+for\s+(?:the\s+)?)(?P<command>.+?)(?:\s+on\s+\S+|\s+in\s+aos-cx\b|\s+command\b|\?|$)",
        r"\b(?:could\s+you\s+help\s+me\s+)?find\s+how\s+to\s+configure\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\b(?:could\s+you\s+help\s+me\s+)?find\s+how\s+to\s+use\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\b(?:could\s+you\s+help\s+me\s+)?find\s+how\s+to\s+set\s+up\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\b(?:could\s+you\s+help\s+me\s+)?find\s+(?:the\s+)?syntax\s+for\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+to\s+configure\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+to\s+use\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+to\s+set\s+up\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+do\s+i\s+configure\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+do\s+i\s+use\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+do\s+you\s+configure\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+do\s+you\s+use\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+can\s+i\s+configure\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+can\s+i\s+use\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bwhat\s+is\s+the\s+syntax\s+of\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bwhat\s+is\s+the\s+syntax\s+for\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bwhat\s+is\s+the\s+cli\s+syntax\s+for\s+(?:the\s+)?(?P<command>.+?)(?:\s+in\s+aos-cx\b|\s+command\b|\?|$)",
        r"\bsyntax of (?:the )?(?P<command>.+?) command\b",
        r"\bwhat\s+is\s+the\s+documented\s+syntax\s+for\s+(?:the\s+)?(?P<command>.+?)(?:\s+on\s+\S+|\s+in\s+aos-cx\b|\s+command\b|\?|$)",
        r"\bwhat does (?:the )?(?P<command>.+?) command do\b",
        r"\bwhat is the syntax of (?:the )?(?P<command>.+?) command\b",
        r"\bwhat is the purpose of (?:the )?(?P<command>.+?) command\b",
        r"\bwhat is (?:the )?(?P<command>.+?) command\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            command = _clean(match.group("command"))
            command = command.strip(" ?.")
            return command
    return ""


def _product_sentence_chunks(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'\[])", _clean(text))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _product_structured_segments(text: str) -> List[str]:
    normalized = _cleanup_product_markdown(_clean(text))
    if not normalized:
        return []
    if "\n" in normalized:
        return [line.strip(" -*") for line in normalized.splitlines() if line.strip()]
    if normalized.count(":") < 3 and len(normalized) < 140:
        return []

    segments = [normalized]
    markers = [
        r"(?=\bPOL\d+:\s*)",
        r"(?=\bShowing\b)",
        r"(?=\bSyntax\b)",
        r"(?=\bDescription\b)",
        r"(?=\bExamples?\b)",
        r"(?=\bAttached Access List\b)",
        r"(?=\bAttached Prefix List\b)",
        r"(?=\bPreference Range\b)",
        r"(?=\bApplied on VLAN\b)",
        r"(?=\bApplied on Port\b)",
    ]
    for pattern in markers:
        next_segments: List[str] = []
        for segment in segments:
            if len(segment) < 24:
                next_segments.append(segment)
                continue
            parts = [part.strip(" :") for part in re.split(pattern, segment) if part and part.strip(" :")]
            if len(parts) > 1:
                next_segments.extend(parts)
            else:
                next_segments.append(segment)
        segments = next_segments

    collapsed: List[str] = []
    for segment in segments:
        cleaned = re.sub(r"\s+", " ", segment).strip()
        if cleaned:
            collapsed.append(cleaned)
    return collapsed


def _strip_product_filler_prefix(answer: str) -> str:
    text = _cleanup_product_markdown(answer)
    lower = normalize_whitespace(text).lower()
    for prefix in PRODUCT_FILLER_PREFIXES:
        if lower.startswith(prefix):
            return _cleanup_product_markdown(text[len(prefix) :])
    return text


def _strip_product_generated_label(answer: str) -> str:
    text = _cleanup_product_markdown(answer)
    lower = normalize_whitespace(text).lower()
    for prefix in PRODUCT_GENERATED_LABEL_PREFIXES:
        if lower.startswith(prefix):
            return _cleanup_product_markdown(text[len(prefix) :])
    return text


def _cleanup_product_markdown(answer: str) -> str:
    text = "" if answer is None else str(answer)
    if not text:
        return ""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned_lines: List[str] = []
    in_code_block = False
    blank_run = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            cleaned_lines.append(stripped)
            blank_run = 0
            continue
        if in_code_block:
            cleaned_lines.append(line.rstrip())
            continue
        if not stripped:
            blank_run += 1
            if blank_run <= 2:
                cleaned_lines.append("")
            continue
        blank_run = 0
        stripped = re.sub(r"(?<!`) {2,}(?!`)", " ", stripped)
        stripped = re.sub(r"\.\.(?=\s|$)", ".", stripped)
        cleaned_lines.append(stripped)
    cleaned = "\n".join(cleaned_lines).strip()
    if cleaned.startswith("- "):
        candidate = cleaned[2:].strip()
        if "\n" not in candidate and not re.search(r"^\s*[-*]\s+", candidate, flags=re.MULTILINE):
            cleaned = candidate
    return cleaned


def _prompt_safe_text(text: object) -> str:
    value = "" if text is None else str(text)
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _product_looks_like_command_question(question: str) -> bool:
    text = _clean(question).lower()
    return any(
        phrase in text
        for phrase in [
            "how to configure",
            "how to use",
            "how to set up",
            "how do i configure",
            "how do i use",
            "how do you configure",
            "how do you use",
            "how can i configure",
            "how can i use",
            "help me find how to configure",
        "help me find how to use",
        "help me find the syntax",
        "what is the cli syntax",
        "what is the syntax",
        "what is the syntax for",
        "syntax of",
        "cli syntax for",
        "show syntax",
        "command syntax",
        "how is the command written",
        ]
    )


def _product_is_command_purpose_question(question: str) -> bool:
    return is_command_purpose_question(question)


def _compact_product_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean(value).lower())


def _product_question_type(question: str) -> str:
    text = _clean(question).lower()
    if not text:
        return "generic_product_query"
    if any(
        phrase in text
        for phrase in (
            "what is the syntax",
            "what is the syntax of",
            "what is the cli syntax",
            "syntax of",
            "command syntax",
            "show syntax",
            "cli syntax for",
            "how is the command written",
        )
    ):
        return "cli_syntax"
    if any(
        phrase in text
        for phrase in (
            "what is the output of",
            "what is the output for",
            "output of the",
            "show output of",
            "show output for",
            "display the output of",
        )
    ):
        return "cli_output"
    if any(
        phrase in text
        for phrase in (
            "supported route scale",
            "maximum supported ipv4 route scale",
            "maximum supported ipv6 route scale",
            "maximum route scale",
            "supported scale",
            "route scale",
            "route capacity",
            "capacity",
            "how many routes",
            "maximum number of routes",
            "route scale on",
        )
    ):
        return "capacity_or_scale"
    if "since which version" in text or "since what version" in text or re.search(r"\bversion\b.*\bsupport", text):
        return "version_support"
    if any(
        phrase in text
        for phrase in (
            "which aos-cx switches support",
            "which switch supports",
            "support matrix",
            "does aruba",
            "support vsf",
            "support issu",
            "support vsx",
        )
    ):
        return "support_matrix"
    if any(phrase in text for phrase in ("how can i", "how do i", "how do you", "bring up", "bring it up", "configure", "set up", "enable", "disable")):
        if any(keyword in text for keyword in ("vsx", "vsf", "issu", "redundancy", "high availability", "standalone")):
            return "procedure"
    if any(phrase in text for phrase in ("limitation", "caveat", "restriction", "unsupported", "cannot", "can't")):
        return "limitation"
    if any(phrase in text for phrase in ("requirement", "prerequisite", "must ", "needed", "need to")):
        return "requirement"
    if any(phrase in text for phrase in ("explain", "overview", "what is ", "what does ", "how does ", "tell me about")):
        return "concept_explanation"
    return "generic_product_query"


def _product_support_feature_from_question(question: str) -> str:
    text = _clean(question)
    patterns = [
        r"\b(?:which|what)\s+(?:aos-cx\s+)?switch(?:es)?\s+support\s+(?P<feature>.+?)(?:\?|$)",
        r"\b(?:since\s+which\s+version\s+does\s+(?:the\s+)?(?:\d{4,5}[A-Za-z]?|CX\d{4})\s+support\s+)(?P<feature>.+?)(?:\?|$)",
        r"\b(?:does|do)\s+(?:the\s+)?(?:\d{4,5}[A-Za-z]?|CX\d{4})\s+support\s+(?P<feature>.+?)(?:\?|$)",
        r"\b(?:supported\s+)?(?:feature|capability|protocol|mode)\s+(?P<feature>.+?)(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        feature = _clean(match.group("feature")).strip(" ?.:-")
        feature = re.sub(r"\b(?:support|supports|supported|version|switch|switches)\b.*$", "", feature, flags=re.IGNORECASE).strip(" ?.:-")
        if feature and len(feature.split()) <= 8:
            return feature
    return ""


def _product_support_feature_aliases(feature: str) -> List[str]:
    text = _clean(feature)
    if not text:
        return []
    lower = text.lower()
    aliases = [text]
    if "standalone issu" in lower or "vsf esu" in lower or "vsf issu" in lower or lower == "issu":
        aliases.extend(["VSF ESU", "VSF ISSU", "ISSU"])
    if lower == "vsf" or "virtual switching framework" in lower:
        aliases.extend(["VSF", "Virtual Switching Framework", "show vsf", "VSX"])
    if lower == "vsx" or "virtual switching extension" in lower:
        aliases.extend(["VSX", "VSX mode", "VSX configuration"])
    if "route scale" in lower or "route capacity" in lower or "supported scale" in lower:
        aliases.extend(
            [
                "routes",
                "Routing",
                "IP Routing",
                "show capacities",
                "show capacities-status",
                "show capacities rpvst",
                "Maximum number of routes (IPv4+IPv6)",
                "Maximum number of IPv4 routes",
                "Maximum number of IPv6 routes",
            ]
        )
    return _unique([alias for alias in aliases if _clean(alias)])


def _product_route_type_from_question(question: str) -> str:
    text = _clean(question).lower()
    if "ipv6" in text:
        return "ipv6"
    if "ipv4" in text or re.search(r"\bip\s+route\b", text):
        return "ipv4"
    return ""


def _product_is_route_capacity_question(question: str, slots: Dict[str, str]) -> bool:
    text = _clean(question).lower()
    route_type = _clean(slots.get("route_type", "")).lower()
    if route_type in {"ipv4", "ipv6"}:
        return True
    if any(
        phrase in text
        for phrase in (
            "supported route scale",
            "maximum supported ipv4 route scale",
            "maximum supported ipv6 route scale",
            "maximum route scale",
            "supported scale",
            "route scale",
            "route capacity",
            "route scale on",
            "how many routes",
            "maximum number of routes",
        )
    ):
        return True
    if re.search(r"\bmaximum number of (?:ipv4|ipv6)? routes\b", text):
        return True
    return False


def _product_topic_family(question: str, question_type: str, slots: Dict[str, str]) -> str:
    text = _clean(question).lower()
    feature = _clean(slots.get("feature", "")).lower()
    command = _clean(slots.get("command", "")).lower()
    if question_type == "capacity_or_scale" and _product_is_route_capacity_question(text, slots):
        return "routing_capacity"
    if question_type == "capacity_or_scale":
        return "capacity_or_scale"
    if question_type in {"support_matrix", "version_support"}:
        if "vsx" in text or "vsx" in feature:
            return "vsx_support"
        if "vsf" in text or "vsf" in feature:
            return "vsf_support"
        if "issu" in text or "issu" in feature:
            return "issu_support"
        return "support_matrix"
    if "issu" in text or "issu" in feature:
        return "issu_support" if question_type in {"support_matrix", "version_support"} else "procedure"
    if "vsf" in text or "vsf" in feature:
        return "vsf_support" if question_type in {"support_matrix", "version_support"} else "procedure"
    if "vsx" in text or "redundancy switchover" in text or "failover" in text:
        return "vsx_procedure" if question_type == "procedure" else "vsx_support"
    if question_type == "cli_output" or command.startswith("show "):
        return "cli_output"
    if question_type == "cli_syntax":
        return "cli_syntax"
    if any(keyword in text for keyword in ("rest", "certificate", "pki", "aaa", "security", "snmp")):
        return "security"
    if question_type in {"limitation", "requirement"}:
        return question_type
    if question_type == "concept_explanation":
        return "concept_explanation"
    return "generic_product_query"


def _product_candidate_topics(question: str, question_type: str, topic_family: str, slots: Dict[str, str]) -> List[str]:
    topics: List[str] = []
    for key, value in (("topic", slots.get("topic", "")), ("feature", slots.get("feature", "")), ("category", slots.get("category", ""))):
        cleaned = _clean(value)
        if not cleaned:
            continue
        if key == "topic":
            lower = cleaned.lower()
            if len(cleaned.split()) > 8 or any(
                phrase in lower
                for phrase in (
                    "what is",
                    "what does",
                    "maximum supported",
                    "supported route scale",
                    "output of",
                    "since which",
                    "how can i",
                    "how do i",
                )
            ):
                continue
        topics.append(cleaned)

    text = _clean(question).lower()
    feature = _clean(slots.get("feature", ""))
    for alias in _product_support_feature_aliases(feature):
        if alias and alias not in topics:
            topics.append(alias)
    if topic_family == "routing_capacity":
        if "ipv6" in text:
            topics.extend(
                [
                    "IPv6 routes",
                    "Maximum number of IPv6 routes",
                    "Long Prefix IPv6 route capacity",
                    "Route table",
                    "routes",
                    "Routing",
                    "IP Routing",
                    "show capacities",
                    "show capacities-status",
                    "show capacities rpvst",
                ]
            )
        elif "ipv4" in text:
            topics.extend(
                [
                    "IPv4 routes",
                    "Maximum number of IPv4 routes",
                    "Route table",
                    "routes",
                    "Routing",
                    "IP Routing",
                    "show capacities",
                    "show capacities-status",
                    "show capacities rpvst",
                ]
            )
        else:
            topics.extend(
                [
                    "routes",
                    "Maximum number of routes (IPv4+IPv6)",
                    "Maximum number of IPv4 routes",
                    "Maximum number of IPv6 routes",
                    "Long Prefix IPv6 route capacity",
                    "Route table",
                    "Routing",
                    "IP Routing",
                    "show capacities",
                    "show capacities-status",
                    "show capacities rpvst",
                ]
            )
    elif topic_family == "issu_support":
        topics.extend(["ISSU", "Upgrade", "Software update"])
    elif topic_family == "vsf_support":
        topics.extend(["VSF", "Virtual Switching", "Virtual Switching Framework", "Stacking", "show vsf"])
    elif topic_family == "vsx_procedure":
        topics.extend(["VSX mode", "VSX configuration", "VSX", "High availability", "Redundancy"])
    elif topic_family == "vsx_support":
        topics.extend(["VSX", "VSX mode", "VSX configuration", "High availability", "Redundancy"])
    elif topic_family == "cli_output":
        if feature:
            topics.append(feature)
        topics.extend(["Routing", "IP Routing", "Static routing", "CLI Reference"])
    elif topic_family == "security":
        topics.extend(["Security", "AAA", "REST", "Certificates", "PKI"])
    elif topic_family == "support_matrix":
        topics.extend(["Support matrix", "Supported switches", "Feature support"])
    elif topic_family == "capacity_or_scale":
        topic_hint = _clean(slots.get("topic", ""))
        feature_hint = _clean(slots.get("feature", ""))
        if topic_hint:
            topics.extend(
                [
                    topic_hint,
                    f"{topic_hint} capacity",
                    f"{topic_hint} range",
                    f"{topic_hint} limit",
                ]
            )
        if feature_hint:
            topics.append(feature_hint)
        topics.extend(["capacity", "supported capacity", "maximum supported capacity", "range", "limit"])
    elif topic_family == "monitoring":
        topics.extend(["Monitoring", "Diagnostics", "Troubleshooting"])
    elif topic_family == "limitation":
        topics.extend(["Limitations", "Static routing"])
    elif topic_family == "requirement":
        topics.extend(["Requirements", "Prerequisites"])
    elif topic_family == "concept_explanation":
        if feature:
            topics.append(feature)

    if question_type == "support_matrix":
        topics = _product_support_feature_aliases(feature) + topics
    if question_type == "version_support":
        topics = _product_support_feature_aliases(feature) + topics

    return _unique([topic for topic in topics if topic])


def _product_candidate_intents(question_type: str, predicted_intent: str) -> List[str]:
    intents = list(PRODUCT_QUESTION_TYPE_TO_INTENTS.get(question_type, PRODUCT_QUESTION_TYPE_TO_INTENTS["generic_product_query"]))
    if _clean(predicted_intent):
        intents.append(predicted_intent)
    if question_type == "cli_output":
        intents.append("show_command_meaning")
    return _unique(intents)


def _product_required_slots(question_type: str) -> List[str]:
    return list(PRODUCT_QUESTION_TYPE_REQUIRED_SLOTS.get(question_type, []))


def _product_switch_aliases(switch: str, known_switches: Sequence[str]) -> List[str]:
    base = _canonical_product_switch(switch)
    if not base:
        return []
    aliases = [base]
    base_lower = base.lower()
    if base_lower == "4100":
        aliases.append("4100i")
    elif base_lower == "4100i":
        aliases.append("4100")
    elif base_lower == "6300":
        aliases.append("6300_6400")
    elif base_lower == "6300_6400":
        aliases.append("6300")
    base_compact = _compact_product_key(base)
    if not base_compact:
        return aliases

    scored: List[Tuple[int, str]] = []
    for known in _unique([_canonical_product_switch(value) for value in known_switches if _clean(value)]):
        known_compact = _compact_product_key(known)
        if not known_compact or known.lower() == base.lower():
            continue
        if (
            known_compact.startswith(base_compact)
            or base_compact.startswith(known_compact)
            or base_compact in known_compact
            or known_compact in base_compact
        ):
            scored.append((abs(len(known_compact) - len(base_compact)), known))
    scored.sort(key=lambda item: (item[0], item[1]))
    aliases.extend([item[1] for item in scored])
    return _unique(aliases)


def _product_question_profile(question: str, slots: Dict[str, str], predicted_intent: str) -> Dict[str, object]:
    question_type = _product_question_type(question)
    profile_slots = dict(slots)
    if not profile_slots.get("feature"):
        feature = _product_support_feature_from_question(question)
        if feature:
            profile_slots["feature"] = feature
    if not profile_slots.get("route_type"):
        route_type = _product_route_type_from_question(question)
        if route_type:
            profile_slots["route_type"] = route_type
    if not profile_slots.get("command"):
        command = _product_command_from_question(question)
        if command:
            profile_slots["command"] = command

    topic_family = _product_topic_family(question, question_type, profile_slots)
    candidate_topics = _product_candidate_topics(question, question_type, topic_family, profile_slots)
    candidate_intents = _product_candidate_intents(question_type, predicted_intent)
    required_slots = _product_required_slots(question_type)
    route_type_variants: List[str] = []
    route_type = _clean(profile_slots.get("route_type", ""))
    if route_type:
        route_type_variants.append(route_type)
    elif question_type == "capacity_or_scale":
        route_type_variants.extend(["", "ipv4", "ipv6"])
    normalized_question = _normalize_product_lookup_question(question, profile_slots, predicted_intent)
    query_keywords = [
        token
        for token in re.findall(r"[A-Za-z0-9_]+", _clean(question).lower())
        if token not in PRODUCT_QWEN_STOPWORDS
    ]
    return {
        "question_type": question_type,
        "topic_family": topic_family,
        "candidate_topics": candidate_topics,
        "candidate_intents": candidate_intents,
        "required_slots": required_slots,
        "route_type_variants": _unique(route_type_variants),
        "normalized_question": normalized_question,
        "query_keywords": _unique(query_keywords[:16]),
        "slots": profile_slots,
    }


def _product_missing_required_slots(profile: Dict[str, object], slots: Dict[str, str]) -> List[str]:
    required = [str(item) for item in profile.get("required_slots", []) if _clean(item)]
    missing = [slot for slot in required if not _clean(slots.get(slot, ""))]
    return missing


def _product_clarification_message(question_type: str, missing_slots: Sequence[str]) -> str:
    missing = {slot for slot in missing_slots if slot}
    if question_type == "capacity_or_scale":
        if "switch" in missing and "version" in missing:
            return "Please specify the switch model and AOS-CX version so I can check the supported scale."
        if "switch" in missing:
            return "Please specify the switch model so I can check the supported scale."
    if question_type in {"support_matrix", "version_support"}:
        if "feature" in missing:
            return "Please specify the feature or capability you want to check, such as ISSU, VSF, or VSX."
    if question_type in {"cli_syntax", "cli_output"} and "command" in missing:
        return "Please specify the exact command so I can look up the documented syntax or output."
    return PRODUCT_SLOT_MISSING_RESPONSE


def _product_resolution_is_grounded(profile: Dict[str, object], resolution: Dict[str, object]) -> bool:
    if resolution.get("status") != "found" or not _clean(resolution.get("answer", "")):
        return False
    question_type = _clean(profile.get("question_type", ""))
    lookup_key_used = _clean(resolution.get("lookup_key_used", ""))
    confidence = float(resolution.get("confidence", 0.0) or 0.0)
    answer = _clean(resolution.get("answer", ""))
    if _product_answer_looks_contaminated(answer):
        return False
    if question_type in PRODUCT_STRICT_QUESTION_TYPES:
        if lookup_key_used == "nearest_input_text_similarity":
            return False
        if question_type == "cli_output":
            command = _clean(dict(profile.get("slots", {})).get("command", ""))
            if command and _product_command_has_extended_variant(answer, command):
                return False
            if len(answer.split()) <= 20 and not re.search(r"[\n`|><=]", answer) and not re.search(r"\d", answer):
                return False
        if question_type == "cli_syntax":
            command = _clean(dict(profile.get("slots", {})).get("command", ""))
            if command and command.lower() not in answer.lower():
                return False
        if question_type == "capacity_or_scale":
            route_type = _clean(dict(profile.get("slots", {})).get("route_type", ""))
            answer_route_type = _product_answer_route_type(answer)
            topic_source = " ".join(
                [
                    _clean(profile.get("normalized_question", "")),
                    _clean(dict(profile.get("slots", {})).get("topic", "")),
                    _clean(dict(profile.get("slots", {})).get("feature", "")),
                    _clean(dict(profile.get("slots", {})).get("category", "")),
                ]
            ).lower()
            route_related = bool(route_type) or any(
                term in topic_source
                for term in (
                    "supported route scale",
                    "route scale",
                    "route capacity",
                    "maximum number of routes",
                    "maximum supported",
                    "ipv4 route",
                    "ipv6 route",
                    "long prefix ipv6 route capacity",
                )
            )
            asks_next_hops = bool(re.search(r"\bnext\s+hops?\b", topic_source, flags=re.IGNORECASE))
            if not asks_next_hops and re.search(r"\bnext\s+hops?\b", answer, flags=re.IGNORECASE):
                return False
            if route_related:
                if route_type and answer_route_type and answer_route_type != route_type:
                    return False
                if route_type == "ipv4" and re.search(r"\blong prefix ipv6\b|\bipv6 routes?\b|\bipv6 route\b", answer, flags=re.IGNORECASE):
                    return False
                if route_type == "ipv6" and re.search(r"\bipv4 routes?\b|\bipv4 route\b", answer, flags=re.IGNORECASE):
                    return False
                if route_type == "ipv4" and not re.search(
                    r"\bmaximum number of ipv4 routes\b|\bnumber of routes \(ipv4\+ipv6\)\b|\bipv4 routes?\b",
                    answer,
                    flags=re.IGNORECASE,
                ):
                    return False
                if route_type == "ipv6" and not re.search(
                    r"\bmaximum number of ipv6 routes\b|\bnumber of routes \(ipv4\+ipv6\)\b|\bipv6 routes?\b",
                    answer,
                    flags=re.IGNORECASE,
                ):
                    return False
                if route_type == "" and not re.search(
                    r"\bmaximum number of routes \(ipv4\+ipv6\)\b|\bmaximum number of ipv4 routes\b|\bmaximum number of ipv6 routes\b",
                    answer,
                    flags=re.IGNORECASE,
                ):
                    return False
                if "route" in topic_source and not re.search(r"\broute\b|\bipv4\b|\bipv6\b|\bmaximum\b|\bcapacity\b|\bscale\b", answer, flags=re.IGNORECASE):
                    return False
            else:
                if not re.search(r"\b(capacity|range|limit|supported|maximum|member|entries?|scale)\b", answer, flags=re.IGNORECASE):
                    return False
                topic_tokens = [token for token in re.findall(r"[A-Za-z0-9_]+", topic_source) if token not in PRODUCT_QWEN_STOPWORDS and len(token) > 2]
                if topic_tokens and not any(token in answer.lower() for token in topic_tokens):
                    return False
        if question_type in {"support_matrix", "version_support"}:
            if not re.search(
                r"\b(support|supported|supports|available|version|introduced|compatible|not applicable|not supported|unsupported)\b",
                answer,
                flags=re.IGNORECASE,
            ):
                return False
        return confidence >= 0.62
    return confidence >= 0.5


def _product_lookup_attempts(
    question: str,
    profile: Dict[str, object],
    slots: Dict[str, str],
    known_switches: Sequence[str],
) -> List[Dict[str, object]]:
    profile_slots = dict(profile.get("slots", slots))
    base_slots = dict(slots)
    for key in ("feature", "command", "topic", "category", "route_type", "question_type"):
        value = _clean(profile_slots.get(key, ""))
        if value and not _clean(base_slots.get(key, "")):
            base_slots[key] = value

    candidate_intents = _unique([str(item) for item in profile.get("candidate_intents", []) if _clean(item)])
    candidate_topics = list(profile.get("candidate_topics", []))
    switch_variants = _product_switch_aliases(base_slots.get("switch", ""), known_switches) if base_slots.get("switch") else [""]
    topic_variants = _unique([_clean(base_slots.get("topic", ""))] + candidate_topics)
    feature_variants = _unique([_clean(base_slots.get("feature", ""))] + _product_support_feature_aliases(base_slots.get("feature", ""))) or [""]
    route_type_variants = _unique([_clean(base_slots.get("route_type", ""))] + [str(item) for item in profile.get("route_type_variants", []) if _clean(item)]) or [""]

    attempts: List[Dict[str, object]] = []
    for intent in candidate_intents:
        for switch in switch_variants or [""]:
            for topic in topic_variants or [""]:
                for feature in feature_variants or [""]:
                    for route_type in route_type_variants or [""]:
                        attempt_slots = dict(base_slots)
                        if switch:
                            attempt_slots["switch"] = switch
                        else:
                            attempt_slots.pop("switch", None)
                        if topic:
                            attempt_slots["topic"] = topic
                        if feature:
                            attempt_slots["feature"] = feature
                        if route_type:
                            attempt_slots["route_type"] = route_type
                        else:
                            attempt_slots.pop("route_type", None)
                        attempts.append(
                            {
                                "intent": intent,
                                "slots": attempt_slots,
                                "normalized_question": _normalize_product_lookup_question(question, attempt_slots, intent),
                                "switch_variant": switch,
                                "topic_variant": topic,
                                "feature_variant": feature,
                                "route_type_variant": route_type,
                                "lookup_path": "exact" if switch and topic and feature else "relaxed",
                            }
                        )
    # Deduplicate attempts while preserving order.
    seen: set[Tuple[str, str, str, str]] = set()
    unique_attempts: List[Dict[str, object]] = []
    for attempt in attempts:
        key = (
            _clean(attempt.get("intent", "")),
            _clean(attempt.get("switch_variant", "")),
            _clean(attempt.get("topic_variant", "")),
            _clean(attempt.get("feature_variant", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_attempts.append(attempt)

    def _attempt_specificity(attempt: Dict[str, object]) -> Tuple[int, int, int, int, int]:
        attempt_slots = dict(attempt.get("slots", {}))
        filled = sum(1 for value in attempt_slots.values() if _clean(value))
        has_switch = 1 if _clean(attempt_slots.get("switch", "")) else 0
        has_version = 1 if _clean(attempt_slots.get("version", "")) else 0
        has_topic = 1 if _clean(attempt_slots.get("topic", "")) else 0
        has_feature = 1 if _clean(attempt_slots.get("feature", "")) else 0
        has_route_type = 1 if _clean(attempt_slots.get("route_type", "")) else 0
        return (filled, has_switch, has_version, has_topic + has_feature + has_route_type, 0 if attempt.get("lookup_path") == "exact" else 1)

    unique_attempts.sort(key=_attempt_specificity, reverse=True)
    return unique_attempts


def _product_resolve_with_profile(
    question: str,
    profile: Dict[str, object],
    slots: Dict[str, str],
    entries: Sequence[Any],
    lookup_index: Dict[str, List[int]],
) -> Dict[str, object]:
    attempts = _product_lookup_attempts(question, profile, slots, [entry.switch for entry in entries if _clean(entry.switch)])
    best_resolution: Optional[Dict[str, object]] = None
    best_rank = -1
    status_rank = {"found": 4, "needs_disambiguation": 3, "low_similarity": 2, "not_found": 1, "slot_missing": 0}

    for index, attempt in enumerate(attempts, start=1):
        resolution = _resolve_generic_lookup(
            "product",
            str(attempt.get("normalized_question", question)),
            str(attempt.get("intent", "")),
            dict(attempt.get("slots", {})),
            entries,
            lookup_index,
        )
        resolution["lookup_stage"] = index
        resolution["normalized_question"] = attempt.get("normalized_question", question)
        resolution["attempt_intent"] = attempt.get("intent", "")
        resolution["attempt_switch"] = attempt.get("switch_variant", "")
        resolution["attempt_topic"] = attempt.get("topic_variant", "")
        resolution["attempt_feature"] = attempt.get("feature_variant", "")
        resolution["lookup_path"] = attempt.get("lookup_path", "relaxed")
        if _product_resolution_is_grounded(profile, resolution):
            return {"resolution": resolution, "attempts": attempts}
        rank = status_rank.get(str(resolution.get("status", "error")), -1)
        current_confidence = float(resolution.get("confidence", 0.0) or 0.0)
        best_confidence = float(best_resolution.get("confidence", 0.0) or 0.0) if best_resolution else -1.0
        if rank > best_rank or (rank == best_rank and current_confidence > best_confidence):
            best_rank = rank
            best_resolution = dict(resolution)

    if best_resolution is None:
        best_resolution = {
            "status": "not_found",
            "answer": None,
            "lookup_key_used": None,
            "confidence": 0.0,
            "similarity": 0.0,
            "reason": "no product lookup attempt matched",
            "lookup_stage": 0,
            "normalized_question": question,
            "attempt_intent": "",
            "attempt_switch": "",
            "attempt_topic": "",
            "attempt_feature": "",
            "lookup_path": "",
        }
    return {"resolution": best_resolution, "attempts": attempts}


def _product_answer_looks_like_cli_syntax(answer: str) -> bool:
    raw = _clean(answer)
    text = raw.lower()
    if not text:
        return False
    if re.search(r"\bsyntax\s*:", text) or re.search(r"\bcommand syntax\s*:", text):
        return True
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) <= 3:
        joined = " ".join(lines)
        has_cli_symbols = any(symbol in joined for symbol in ("<", ">", "[", "]", "{", "}", "|"))
        starts_like_command = re.match(
            r"^(no\s+)?(show|clear|ip|ipv6|interface|vlan|bfd|redundancy|apply|aaa|erps|mdns-sd)\b",
            joined.lower(),
        )
        if has_cli_symbols and starts_like_command and len(joined.split()) <= 40:
            return True
    return False


def _product_answer_looks_contaminated(answer: str) -> bool:
    raw = _cleanup_product_markdown(_clean(answer))
    text = raw.lower()
    if not text:
        return False
    if "table of contents" in text or text.startswith("contents"):
        return True
    if "chapter 1 about this document" in text and "applicable products" in text:
        return True

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    pipe_count = raw.count("|")
    page_number_hits = len(re.findall(r"\|\s*\d+\s*\|", raw))
    heading_hits = sum(
        1
        for marker in ("chapter", "contents", "appendix", "about this document", "applicable products", "support and other resources")
        if marker in text
    )
    if pipe_count >= 20 and (heading_hits >= 2 or page_number_hits >= 8):
        return True
    if len(lines) < 4:
        return False

    dotted_leader_lines = sum(1 for line in lines if re.search(r"\.{6,}", line))
    page_number_lines = sum(
        1 for line in lines if re.match(r"^\s*(?:\d+|[ivxlcdm]+)\s+.+", line, flags=re.IGNORECASE)
    )
    navigation_lines = sum(
        1
        for line in lines
        if re.search(r"\bchapter\b|\bappendix\b|\bcontents\b|\bsection\b", line, flags=re.IGNORECASE)
    )
    pipe_dense_lines = sum(1 for line in lines if line.count("|") >= 2)
    command_like_lines = sum(
        1
        for line in lines
        if re.match(r"^(no\s+)?(show|clear|ip|ipv6|interface|vlan|bfd|redundancy|aaa|erps|apply|mdns-sd)\b", line.lower())
    )

    if dotted_leader_lines and page_number_lines:
        return True
    if dotted_leader_lines >= 2:
        return True
    if navigation_lines >= 2:
        return True
    if len(lines) >= 6 and (pipe_dense_lines >= 2 or command_like_lines >= 3):
        return True
    if len(raw) > 700 and (dotted_leader_lines or page_number_lines or pipe_dense_lines or pipe_count >= 12):
        return True
    return False


def _product_command_has_extended_variant(answer: str, command: str) -> bool:
    answer_text = _clean(answer).lower()
    command_text = _clean(command).lower()
    if not answer_text or not command_text:
        return False
    pattern = rf"(?<!\w){re.escape(command_text)}\s+[a-z0-9_-]+"
    return bool(re.search(pattern, answer_text))


def _product_answer_route_type(answer: str) -> str:
    text = _clean(answer).lower()
    if not text:
        return ""
    has_ipv4 = bool(re.search(r"\bipv4\b", text))
    has_ipv6 = bool(re.search(r"\bipv6\b", text) or re.search(r"long prefix ipv6", text))
    if has_ipv4 and not has_ipv6:
        return "ipv4"
    if has_ipv6 and not has_ipv4:
        return "ipv6"
    return ""


def _normalize_product_lookup_question(question: str, slots: Dict[str, str], predicted_intent: str) -> str:
    text = _clean(question)
    command = _clean(slots.get("command", ""))
    topic = _clean(slots.get("topic", ""))
    feature = _clean(slots.get("feature", ""))
    category = _clean(slots.get("category", ""))
    event_id = _clean(slots.get("event_id", ""))
    question_type = _clean(slots.get("question_type", "")) or _clean(predicted_intent)
    intent = _clean(predicted_intent)
    switch = _clean(slots.get("switch", ""))
    version = _clean(slots.get("version", ""))
    route_type = _clean(slots.get("route_type", ""))

    if command and _product_looks_like_command_question(text):
        return f"What is the syntax of {command} command?"
    if event_id:
        return f"What does event {event_id} mean?"
    if question_type == "capacity_or_scale":
        parts = ["What is the supported"]
        if route_type:
            parts.append(f"{route_type.upper()}")
        parts.append("route scale")
        if switch:
            parts.append(f"for Aruba {switch}")
        if version:
            parts.append(f"running AOS-CX {version}")
        return " ".join(parts).replace("  ", " ").strip() + "?"
    if question_type == "version_support" and feature:
        if switch:
            return f"Since which version does Aruba {switch} support {feature}?"
        return f"Since which version does AOS-CX support {feature}?"
    if question_type == "support_matrix" and feature:
        return f"Which AOS-CX switches support {feature}?"
    if feature and category:
        return f"What is {category} {feature}?"
    if feature:
        return f"What is {feature}?"
    if topic:
        return f"What is {topic}?"

    lower = text.lower()
    if _product_looks_like_command_question(text):
        command_match = _product_command_from_question(text)
        if command_match:
            return f"What is the syntax of {command_match} command?"
    if intent in PRODUCT_EXACT_INTENTS:
        return text
    return text


def _product_intent_override(question: str, slots: Dict[str, str], predicted_intent: str) -> str:
    text = _clean(question).lower()
    if slots.get("command") and _product_looks_like_command_question(text):
        return "cli_syntax"
    if slots.get("command") and _product_is_command_purpose_question(text) and predicted_intent in {"cli_syntax", "show_command_syntax"}:
        return "concept_explanation"
    return predicted_intent


def _product_topic_from_question(question: str) -> str:
    text = _clean(question)
    patterns = [
        r"\bwhat\s+does\s+(?:the\s+)?(?:guide|documentation|docs|manual)\s+say\s+about\s+(?P<topic>.+?)(?:\?|$)",
        r"\bwhat\s+does\s+(?:the\s+)?(?:guide|documentation|docs|manual)\s+explain\s+about\s+(?P<topic>.+?)(?:\?|$)",
        r"\bwhat\s+can\s+you\s+tell\s+me\s+about\s+(?P<topic>.+?)(?:\?|$)",
        r"\btell\s+me\s+about\s+(?P<topic>.+?)(?:\?|$)",
        r"\bwhat\s+is\s+(?P<topic>.+?)(?:\?|$)",
        r"\bexplain\s+(?P<topic>.+?)(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        topic = _clean(match.group("topic")).strip(" ?.")
        if not topic or re.search(r"\bcommand\b", topic, flags=re.IGNORECASE):
            continue
        if re.search(r"\bsyntax\b|\boutput\b", topic, flags=re.IGNORECASE):
            continue
        if re.search(r"\b(this|that|those|these|it)\b", topic, flags=re.IGNORECASE) and len(topic.split()) <= 6:
            continue
        return topic
    return ""


def _product_capacity_topic_from_question(question: str) -> str:
    text = _clean(question)
    lower = text.lower()
    route_scale_markers = (
        "supported route scale",
        "maximum supported ipv4 route scale",
        "maximum supported ipv6 route scale",
        "maximum route scale",
        "supported scale",
        "route scale",
        "route capacity",
        "route scale on",
        "maximum supported",
    )
    if any(marker in lower for marker in route_scale_markers):
        if "ipv4" in lower:
            return "IPv4 route scale"
        if "ipv6" in lower:
            return "IPv6 route scale"
        return "route scale"
    patterns = [
        r"\bwhat\s+is\s+the\s+supported\s+capacity\s+for\s+(?P<topic>.+?)(?:\?|$)",
        r"\bwhat\s+is\s+the\s+maximum\s+supported\s+(?P<topic>.+?)(?:\?|$)",
        r"\bwhat\s+is\s+the\s+supported\s+route\s+scale\s+for\s+(?P<topic>.+?)(?:\?|$)",
        r"\bwhat\s+is\s+the\s+maximum\s+route\s+scale\s+for\s+(?P<topic>.+?)(?:\?|$)",
        r"\bwhat\s+is\s+the\s+supported\s+capacity\s+of\s+(?P<topic>.+?)(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        topic = _clean(match.group("topic")).strip(" ?.:-")
        if not topic:
            continue
        topic = re.sub(r"^(?:for|on|in|of|with|about|regarding)\s+", "", topic, flags=re.IGNORECASE)
        topic = re.sub(r"\b(?:aruba\s+)?(?:aos-cx\s+)?\d{4,5}(?:_\d{4})?\b.*$", "", topic, flags=re.IGNORECASE).strip(" ?.:-")
        topic = re.sub(r"\b(?:switch|series|running|version)\b.*$", "", topic, flags=re.IGNORECASE).strip(" ?.:-")
        if topic:
            lower_topic = topic.lower()
            if "ipv4" in lower_topic:
                return "IPv4 route scale"
            if "ipv6" in lower_topic:
                return "IPv6 route scale"
            if "route scale" in lower_topic:
                return "route scale"
            return topic
    topic = _product_topic_from_question(text)
    if topic:
        topic = re.sub(
            r"^(?:the\s+)?(?:supported\s+)?(?:capacity|route\s+scale)(?:\s+for)?\s+",
            "",
            topic,
            flags=re.IGNORECASE,
        )
        topic = re.sub(r"^(?:for|on|in|of|with|about|regarding)\s+", "", topic, flags=re.IGNORECASE)
        if re.search(r"\bipv4\b", topic, flags=re.IGNORECASE):
            return "IPv4 route scale"
        if re.search(r"\bipv6\b", topic, flags=re.IGNORECASE):
            return "IPv6 route scale"
        if re.search(r"\broute\s+scale\b", topic, flags=re.IGNORECASE):
            return "route scale"
    return _clean(topic)


def _product_event_id_from_question(question: str) -> str:
    text = _clean(question)
    match = re.search(r"\b(?:event\s+id|event)\s*(?:is\s*)?(?P<event_id>\d{3,7})\b", text, flags=re.IGNORECASE)
    return match.group("event_id") if match else ""


def _product_slots_from_question(question: str) -> Dict[str, str]:
    slots = extract_slots_from_question(question)
    text = _clean(question)
    question_type = _product_question_type(text)
    switch_match = re.search(
        r"\b(?:For\s+)?(?:an?\s+|the\s+)?(?P<switch>(?:CX\d{4}|\d{4,5}[A-Za-z]?))\s+(?:Switch\s+Series\s+)?(?:running\s+)?AOS-CX\s+(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<sub>\d+))?\b",
        text,
        flags=re.IGNORECASE,
    )
    if switch_match:
        slots["switch"] = _canonical_product_switch(switch_match.group("switch"))
        slots["version"] = f"{switch_match.group('major')}.{switch_match.group('minor')}"
        if switch_match.group("sub"):
            slots["sub_version"] = switch_match.group("sub")
    elif not slots.get("switch") and re.search(r"\b(?:support|scale|version|configure|syntax|output|bring up|issue|caveat|limitation)\b", text, flags=re.IGNORECASE):
        generic_switch_match = re.search(
            r"\b(?:for|on|in|since|with)\s+(?:an?\s+|the\s+)?(?:Aruba\s+)?(?P<switch>(?:CX\d{4}|\d{4,5}[A-Za-z]?))\b",
            text,
            flags=re.IGNORECASE,
        )
        if generic_switch_match:
            slots["switch"] = _canonical_product_switch(generic_switch_match.group("switch"))
        else:
            broad_switch_match = re.search(
                r"\b(?:Aruba\s+)?(?P<switch>(?:CX\d{4}|\d{4,5}[A-Za-z]?))\b",
                text,
                flags=re.IGNORECASE,
            )
            if broad_switch_match:
                slots["switch"] = _canonical_product_switch(broad_switch_match.group("switch"))
    command = _product_command_from_question(text)
    if command:
        slots["command"] = command
    feature = _product_support_feature_from_question(text)
    if feature and not slots.get("feature"):
        slots["feature"] = feature
    topic = _product_topic_from_question(text)
    if question_type == "capacity_or_scale":
        capacity_topic = _product_capacity_topic_from_question(text)
        if capacity_topic:
            slots["topic"] = capacity_topic
    elif topic and not slots.get("topic") and question_type not in {"cli_syntax", "cli_output", "capacity_or_scale", "support_matrix", "version_support"}:
        slots["topic"] = topic
    event_id = _product_event_id_from_question(text)
    if event_id:
        slots["event_id"] = event_id
    route_type = _product_route_type_from_question(text)
    if route_type:
        slots["route_type"] = route_type
    return slots


def _merge_context_slots(
    slots: Dict[str, str],
    session_context: Dict[str, Optional[str]],
    selected_context: Dict[str, str],
    *,
    use_session_context: bool,
) -> Dict[str, str]:
    effective = {key: _clean(value) for key, value in slots.items() if _clean(value)}
    for key in ("switch", "version", "sub_version", "feature", "category", "bug_id", "command", "topic", "event_id"):
        value = _clean(selected_context.get(key, ""))
        if key == "switch":
            value = _canonical_product_switch(value)
        if value and not effective.get(key):
            effective[key] = value
    if not use_session_context:
        return effective
    for key in ("last_bug_id", "last_switch", "last_version", "last_sub_version", "last_feature", "last_category", "last_command", "last_topic", "last_event_id"):
        value = _clean(session_context.get(key))
        if not value:
            continue
        target = key.replace("last_", "")
        if target == "switch":
            value = _canonical_product_switch(value)
        if target == "version" and value:
            value = _domain_version(value, "product")
        if not effective.get(target):
            effective[target] = value
    return effective


def _build_product_lookup_index(entries) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = defaultdict(list)
    for entry in entries:
        slots = dict(entry.slots)
        slots.setdefault("switch", entry.switch)
        slots.setdefault("version", entry.version)
        slots.setdefault("sub_version", entry.sub_version)
        slots.setdefault("command", _clean(slots.get("command", "")))
        slots.setdefault("topic", _clean(slots.get("topic", "")))
        slots.setdefault("feature", _clean(slots.get("feature", "")))
        slots.setdefault("category", _clean(slots.get("category", "")))
        slots.setdefault("section", _clean(slots.get("section", "")))
        slots.setdefault("event_id", _clean(slots.get("event_id", "")))
        slots.setdefault("question_type", _clean(slots.get("question_type", "")))

        switch = _canonical_product_switch(slots.get("switch", ""))
        version = _clean(slots.get("version", "")).replace("_", ".")
        sub_version = _clean(slots.get("sub_version", ""))
        primary = _product_primary_slot(slots)

        candidates: List[str] = []
        version_aliases = _product_version_aliases(version, sub_version)
        for version_alias in version_aliases:
            if switch and version_alias and sub_version and primary:
                candidates.append("|".join([entry.intent, switch, version_alias, sub_version, primary]))
            if switch and version_alias and primary:
                candidates.append("|".join([entry.intent, switch, version_alias, primary]))
        if switch and primary:
            candidates.append("|".join([entry.intent, switch, primary]))
        if primary:
            candidates.append("|".join([entry.intent, primary]))
        for version_alias in version_aliases:
            if switch and version_alias and sub_version:
                candidates.append("|".join([entry.intent, switch, version_alias, sub_version]))
            if switch and version_alias:
                candidates.append("|".join([entry.intent, switch, version_alias]))
        if switch:
            candidates.append("|".join([entry.intent, switch]))
        candidates.append(entry.intent)

        for key in _unique(candidates):
            if entry.entry_id not in index[key]:
                index[key].append(entry.entry_id)
    return dict(index)


def _build_product_availability_index(entries) -> Dict[str, object]:
    product_docs: Dict[str, Dict[str, Dict[str, set[str]]]] = defaultdict(lambda: {"versions": defaultdict(set)})
    for entry in entries:
        switch = _canonical_product_switch(entry.switch)
        version = _clean(entry.version).replace("_", ".")
        sub_version = _clean(entry.sub_version)
        if switch and version:
            for version_alias in _product_version_aliases(version, sub_version):
                product_docs[switch]["versions"][version_alias].add(sub_version)
    normalized: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for switch, payload in product_docs.items():
        versions = payload.get("versions", {})
        normalized[switch] = {
            "versions": {version: sorted(value for value in values if value) for version, values in versions.items()}
        }
    return {"release_notes": {}, "product_docs": normalized}


def _build_product_bug_metadata_index(entries) -> Dict[str, List[Dict[str, str]]]:
    index: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for entry in entries:
        bug_id = _clean(entry.bug_id)
        if not bug_id:
            continue
        index[bug_id].append(
            {
                "switch": _clean(entry.switch),
                "version": _domain_version(entry.version, "product"),
                "sub_version": _clean(entry.sub_version),
                "intent": _clean(entry.intent),
                "feature": _clean(entry.feature),
                "category": _clean(entry.category),
                "question_type": _clean(entry.question_type),
            }
        )
    return dict(index)


def _select_primary_answer(answers: Sequence[str]) -> str:
    unique = _unique([_clean(answer) for answer in answers if _clean(answer)])
    return unique[0] if unique else ""


def _build_candidate_keys(domain: str, intent: str, slots: Dict[str, str]) -> List[str]:
    intent = _clean(intent)
    if not intent:
        return []
    switch = _clean(slots.get("switch", ""))
    version = _domain_version(slots.get("version", ""), domain)
    sub_version = _clean(slots.get("sub_version", ""))
    bug_id = _clean(slots.get("bug_id", ""))
    feature = _clean(slots.get("feature", ""))
    category = _clean(slots.get("category", ""))
    command = _clean(slots.get("command", ""))
    topic = _clean(slots.get("topic", ""))
    section = _clean(slots.get("section", ""))
    event_id = _clean(slots.get("event_id", ""))
    question_type = _clean(slots.get("question_type", ""))
    primary = _product_primary_slot(slots)

    candidates: List[str] = []
    if domain == "release":
        if intent == "release_caveat":
            if switch and version and sub_version and feature and question_type:
                candidates.append("|".join([intent, switch, version, sub_version, feature, question_type]))
            if switch and version and sub_version and feature:
                candidates.append("|".join([intent, switch, version, sub_version, feature]))
            if switch and version and feature and question_type:
                candidates.append("|".join([intent, switch, version, feature, question_type]))
            if switch and version and feature:
                candidates.append("|".join([intent, switch, version, feature]))
            if feature and question_type:
                candidates.append("|".join([intent, feature, question_type]))
            if feature:
                candidates.append("|".join([intent, feature]))
            return _unique(candidates)
        if intent.startswith("bug_"):
            if switch and version and sub_version and bug_id:
                candidates.append("|".join([intent, switch, version, sub_version, bug_id]))
            if bug_id:
                candidates.append("|".join([intent, bug_id]))
            if switch and version and sub_version and category and bug_id:
                candidates.append("|".join([intent, switch, version, sub_version, category, bug_id]))
            if category and bug_id:
                candidates.append("|".join([intent, category, bug_id]))
            return _unique(candidates)
        return _unique(candidates)

    # product lookups
    version_aliases = _product_version_aliases(version, sub_version)
    for version_alias in version_aliases:
        if switch and version_alias and sub_version and primary:
            candidates.append("|".join([intent, switch, version_alias, sub_version, primary]))
        if switch and version_alias and primary:
            candidates.append("|".join([intent, switch, version_alias, primary]))
    if switch and primary:
        candidates.append("|".join([intent, switch, primary]))
    if primary:
        candidates.append("|".join([intent, primary]))
    for version_alias in version_aliases:
        if switch and version_alias and sub_version:
            candidates.append("|".join([intent, switch, version_alias, sub_version]))
        if switch and version_alias:
            candidates.append("|".join([intent, switch, version_alias]))
    if switch:
        candidates.append("|".join([intent, switch]))
    if command:
        candidates.append("|".join([intent, command]))
    if topic:
        candidates.append("|".join([intent, topic]))
    if feature:
        candidates.append("|".join([intent, feature]))
    if section:
        candidates.append("|".join([intent, section]))
    if event_id:
        candidates.append("|".join([intent, event_id]))
    if category:
        candidates.append("|".join([intent, category]))
    if question_type:
        candidates.append("|".join([intent, question_type]))
    candidates.append(intent)
    return _unique(candidates)


def _tokenize(text: object) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", _clean(text).lower())


def _jaccard(left: object, right: object) -> float:
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _score_entry(domain: str, question: str, slots: Dict[str, str], entry) -> float:
    q_norm = _clean(question).lower()
    e_norm = _clean(entry.input_text).lower()
    seq = SequenceMatcher(None, q_norm, e_norm).ratio()
    tok = _jaccard(q_norm, e_norm)
    score = 0.58 * seq + 0.25 * tok

    for key in ("bug_id", "switch", "version", "sub_version", "feature", "category", "command", "topic", "section", "event_id", "question_type"):
        slot_value = _clean(slots.get(key, ""))
        entry_value = _clean(getattr(entry, key, ""))
        if not slot_value or not entry_value:
            continue
        if key == "version":
            slot_value = _domain_version(slot_value, domain)
            entry_value = _domain_version(entry_value, domain)
        if slot_value.lower() == entry_value.lower():
            if key in {"bug_id", "switch", "command", "topic"}:
                score += 0.15
            elif key in {"version", "sub_version", "feature", "category", "section", "event_id"}:
                score += 0.08
            else:
                score += 0.05

    command_value = _clean(slots.get("command", "")).lower()
    if command_value:
        answer_text = _clean(getattr(entry, "answer", "")).lower()
        input_text = _clean(getattr(entry, "input_text", "")).lower()
        command_root = command_value.split()[0] if command_value else ""
        if command_value and command_value in answer_text:
            score += 0.3
        elif command_root and command_root in answer_text:
            score += 0.12
        elif command_value and command_value in input_text:
            score += 0.12
        elif command_root and command_root in input_text:
            score += 0.05

    if _clean(slots.get("question_type", "")) == "cli_output" and command_value:
        answer_text = _clean(getattr(entry, "answer", "")).lower()
        input_text = _clean(getattr(entry, "input_text", "")).lower()
        if command_value in input_text:
            score += 0.18
        if command_value in answer_text:
            score += 0.08
        if _product_command_has_extended_variant(answer_text, command_value):
            score -= 0.22

    question_type = _clean(slots.get("question_type", ""))
    if question_type in {"support_matrix", "version_support"}:
        feature_value = _clean(slots.get("feature", "")).lower()
        answer_text = _clean(getattr(entry, "answer", "")).lower()
        input_text = _clean(getattr(entry, "input_text", "")).lower()
        if feature_value and feature_value in answer_text:
            score += 0.22
        elif feature_value and feature_value in input_text:
            score += 0.14
        if re.search(r"\b(support|supported|supports|available|introduced|compatible)\b", answer_text):
            score += 0.10
        if re.search(r"\bnot applicable\b|\bnot supported\b|\bunsupported\b", answer_text):
            score += 0.08
        if re.search(r"\b(support-log|support-files|support-file)\b", input_text):
            score -= 0.18
        if _clean(slots.get("switch", "")) and _clean(slots.get("switch", "")).lower() in input_text:
            score += 0.08
        if _clean(slots.get("version", "")) and _clean(slots.get("version", "")).replace("_", ".") in input_text:
            score += 0.08
    elif question_type == "capacity_or_scale":
        answer_text = _clean(getattr(entry, "answer", "")).lower()
        input_text = _clean(getattr(entry, "input_text", "")).lower()
        route_type = _clean(slots.get("route_type", "")).lower()
        question_text = q_norm
        route_related = bool(route_type) or bool(
            re.search(
                r"\b(route scale|route capacity|supported route scale|maximum supported|maximum number of (?:ipv4|ipv6)? routes|how many routes)\b",
                question_text,
            )
        )
        mentions_next_hop = bool(re.search(r"\bnext\s+hops?\b|\bnext-hop\b", question_text))
        mentions_ipv4 = "ipv4" in question_text
        mentions_ipv6 = "ipv6" in question_text
        mentions_rip = bool(re.search(r"\bripv?2\b|\bripng\b|\brip\b", question_text))
        if route_related:
            route_capacity_terms = (
                "maximum number of routes",
                "maximum number of ipv4 routes",
                "maximum number of ipv6 routes",
                "long prefix ipv6 route capacity",
                "show resources",
                "show capacities",
                "show capacities-status",
                "route scale",
                "route capacity",
            )
            if re.search(r"\bmaximum number of routes\b|\bnumber of routes\b|\bmaximum number of ipv4 routes\b|\bmaximum number of ipv6 routes\b", answer_text):
                score += 0.22
            if re.search(r"\bmaximum number of routes\b|\bnumber of routes\b|\bmaximum number of ipv4 routes\b|\bmaximum number of ipv6 routes\b", input_text):
                score += 0.14
            if "show capacities rpvst" in input_text or "show capacities-status" in input_text:
                score += 0.12
            if re.search(r"\b(route|routes|scale|capacity|maximum|supported)\b", answer_text):
                score += 0.14
            if re.search(r"\b(route|routes|scale|capacity|maximum|supported)\b", input_text):
                score += 0.08
            if re.search(r"\b\d{2,5}\b", answer_text):
                score += 0.05
            if any(term in answer_text for term in route_capacity_terms):
                score += 0.18
            if any(term in input_text for term in route_capacity_terms):
                score += 0.12
            if "long prefix ipv6 route capacity" in answer_text or "long prefix ipv6 route capacity" in input_text:
                score += 0.20
            if "show resources" in answer_text or "show resources" in input_text:
                score += 0.10
            if mentions_ipv4 and "ipv4" in answer_text:
                score += 0.10
            if mentions_ipv6 and "ipv6" in answer_text:
                score += 0.10
            if route_type == "ipv4" and "ipv4" in answer_text:
                score += 0.12
            if route_type == "ipv6" and "ipv6" in answer_text:
                score += 0.12
            if not mentions_next_hop and re.search(r"\bnext\s+hops?\b|\bnext-hop\b", answer_text):
                score -= 0.30
            if not mentions_next_hop and re.search(r"\bnext\s+hops?\b|\bnext-hop\b", input_text):
                score -= 0.22
            if not mentions_rip and re.search(r"\bripv?2\b|\bripng\b|\brip\b", answer_text):
                score -= 0.20
            if not mentions_rip and re.search(r"\bripv?2\b|\bripng\b|\brip\b", input_text):
                score -= 0.12
            if mentions_ipv4 and "ipv6" in answer_text and "ipv4" not in answer_text:
                score -= 0.08
            if mentions_ipv6 and "ipv4" in answer_text and "ipv6" not in answer_text:
                score -= 0.08
        else:
            topic_source = " ".join(
                [
                    _clean(slots.get("topic", "")),
                    _clean(slots.get("feature", "")),
                    _clean(slots.get("category", "")),
                ]
            ).lower()
            topic_tokens = [token for token in re.findall(r"[A-Za-z0-9_]+", topic_source) if token not in PRODUCT_QWEN_STOPWORDS and len(token) > 2]
            generic_capacity_markers = (
                "capacity",
                "supported",
                "maximum",
                "range",
                "limit",
                "member",
                "entries",
                "entry",
                "scale",
            )
            if any(marker in answer_text for marker in generic_capacity_markers):
                score += 0.12
            if any(marker in input_text for marker in generic_capacity_markers):
                score += 0.06
            if re.search(r"\b\d+\s*(?:to|-|through|/)\s*\d+\b|\b\d{2,5}\b", answer_text):
                score += 0.08
            if re.search(r"\b\d+\s*(?:to|-|through|/)\s*\d+\b|\b\d{2,5}\b", input_text):
                score += 0.04
            if topic_tokens and any(token in answer_text for token in topic_tokens):
                score += 0.10
            if topic_tokens and any(token in input_text for token in topic_tokens):
                score += 0.06

    return min(1.0, score)


def _product_answer_matches_exact_query(question_type: str, slots: Dict[str, str], answer: str) -> bool:
    text = _clean(answer).lower()
    if not text:
        return False
    if question_type == "capacity_or_scale":
        route_type = _clean(slots.get("route_type", "")).lower()
        route_related = bool(route_type) or any(
            term in text
            for term in (
                "supported route scale",
                "route scale",
                "route capacity",
                "maximum number of routes",
                "maximum number of ipv4 routes",
                "maximum number of ipv6 routes",
                "long prefix ipv6 route capacity",
                "show capacities",
                "show capacities-status",
                "show resources",
            )
        )
        if route_related:
            core_route_terms = (
                "route scale",
                "route capacity",
                "maximum number of routes",
                "maximum number of ipv4 routes",
                "maximum number of ipv6 routes",
                "long prefix ipv6 route capacity",
            )
            route_terms = core_route_terms + (
                "show capacities",
                "show capacities-status",
                "show resources",
            )
            if route_type == "ipv4":
                core_route_terms = core_route_terms + ("ipv4 routes", "maximum number of ipv4 routes")
                route_terms = route_terms + ("ipv4", "ipv4 routes", "maximum number of ipv4 routes")
            elif route_type == "ipv6":
                core_route_terms = core_route_terms + ("ipv6 routes", "maximum number of ipv6 routes")
                route_terms = route_terms + ("ipv6", "ipv6 routes", "maximum number of ipv6 routes")
            if not any(term in text for term in core_route_terms):
                return False
            if any(term in text for term in route_terms):
                return True
            if route_type and re.search(rf"\b{route_type}\b.*\broute\b", text):
                return True
            return False
        topic_source = " ".join(
            [
                _clean(slots.get("topic", "")),
                _clean(slots.get("feature", "")),
                _clean(slots.get("category", "")),
            ]
        ).lower()
        topic_tokens = [token for token in re.findall(r"[A-Za-z0-9_]+", topic_source) if token not in PRODUCT_QWEN_STOPWORDS and len(token) > 2]
        if topic_tokens and not any(token in text for token in topic_tokens):
            return False
        if not re.search(r"\b(capacity|range|limit|supported|maximum|member|entries?|scale)\b", text):
            return False
        return True
    if question_type in {"support_matrix", "version_support"}:
        feature = _clean(slots.get("feature", ""))
        aliases = _product_support_feature_aliases(feature)
        support_terms = ("support", "supported", "supports", "available", "introduced", "not supported", "unsupported")
        if any(alias.lower() in text for alias in aliases if alias) and any(term in text for term in support_terms):
            return True
        if feature and feature.lower() in text and ("not supported" in text or "unsupported" in text):
            return True
        return False
    if question_type == "cli_syntax":
        command = _clean(slots.get("command", "")).lower()
        if command and command in text:
            return True
        if any(symbol in text for symbol in ("<", ">", "[", "]", "{", "}", "|")):
            return True
        return False
    if question_type == "cli_output":
        command = _clean(slots.get("command", "")).lower()
        if command and command in text:
            return True
        if "show " in text:
            return True
        return False
    return True


def _rank_entries(domain: str, question: str, slots: Dict[str, str], candidates: Sequence[Any]) -> List[Tuple[Any, float]]:
    ranked = [(entry, _score_entry(domain, question, slots, entry)) for entry in candidates]
    ranked.sort(key=lambda item: (item[1], getattr(item[0], "entry_id", 0)), reverse=True)
    return ranked


def _resolve_generic_lookup(
    domain: str,
    question: str,
    intent: str,
    slots: Dict[str, str],
    entries,
    lookup_index: Dict[str, List[int]],
) -> Dict[str, object]:
    candidates = _build_candidate_keys(domain, intent, slots)
    question_type = _clean(slots.get("question_type", ""))
    exact_product_intent = domain == "product" and (
        intent in PRODUCT_EXACT_INTENTS or question_type in {"cli_output", "capacity_or_scale"}
    )
    matching_entries: List[Any] = []
    seen_ids: set[int] = set()
    for key in candidates:
        key_entry_ids = lookup_index.get(key, [])
        key_entries: List[Any] = []
        for entry_id in key_entry_ids:
            if entry_id in seen_ids or not (0 <= entry_id < len(entries)):
                continue
            seen_ids.add(entry_id)
            entry = entries[entry_id]
            key_entries.append(entry)
        if not key_entries:
            continue

        if exact_product_intent and question_type != "cli_syntax":
            key_answers = [_clean(entry.answer) for entry in key_entries if _clean(entry.answer)]
            unique_key_answers = _unique(key_answers)
            if len(unique_key_answers) == 1 and _product_answer_matches_exact_query(question_type, slots, unique_key_answers[0]):
                return {
                    "status": "found",
                    "answer": unique_key_answers[0],
                    "lookup_key_used": key,
                    "confidence": 0.99,
                    "similarity": 0.99,
                    "reason": "exact product key",
                }

        matching_entries.extend(key_entries)

    if matching_entries:
        if exact_product_intent and question_type == "cli_syntax":
            command = _clean(slots.get("command", "")).lower()
            if command:
                command_entries = [
                    entry
                    for entry in matching_entries
                    if command in _clean(getattr(entry, "answer", "")).lower()
                    or command in _clean(getattr(entry, "input_text", "")).lower()
                ]
                if command_entries:
                    matching_entries = command_entries
                else:
                    return {
                        "status": "not_found",
                        "answer": None,
                        "lookup_key_used": candidates[0] if candidates else None,
                        "confidence": 0.0,
                        "similarity": 0.0,
                        "reason": "no exact command syntax match",
                    }
        answers = [_clean(entry.answer) for entry in matching_entries if _clean(entry.answer)]
        unique_answers = _unique(answers)
        if len(unique_answers) == 1:
            return {
                "status": "found",
                "answer": unique_answers[0],
                "lookup_key_used": candidates[0] if candidates else None,
                "confidence": 0.98,
                "similarity": 0.98,
                "reason": None,
            }
        ranked = _rank_entries(domain, question, slots, matching_entries)
        if not ranked:
            return {
                "status": "not_found",
                "answer": None,
                "lookup_key_used": candidates[0] if candidates else None,
                "confidence": 0.0,
                "similarity": 0.0,
                "reason": "no ranked entries",
            }
        best_entry, best_score = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if exact_product_intent:
            if best_score < 0.5:
                return {
                    "status": "low_similarity",
                    "answer": None,
                    "lookup_key_used": candidates[0] if candidates else None,
                    "confidence": best_score,
                    "similarity": best_score,
                    "reason": "best similarity below exact-product threshold",
                }
            if not _product_answer_matches_exact_query(question_type, slots, _clean(best_entry.answer)):
                return {
                    "status": "low_similarity",
                    "answer": None,
                    "lookup_key_used": candidates[0] if candidates else None,
                    "confidence": best_score,
                    "similarity": best_score,
                    "reason": "exact answer failed topical sanity check",
                }
            return {
                "status": "found",
                "answer": _clean(best_entry.answer),
                "lookup_key_used": candidates[0] if candidates else None,
                "confidence": best_score,
                "similarity": best_score,
                "reason": "exact product intent",
            }
        if best_score < 0.56:
            return {
                "status": "low_similarity",
                "answer": None,
                "lookup_key_used": candidates[0] if candidates else None,
                "confidence": best_score,
                "similarity": best_score,
                "reason": "best similarity below threshold",
            }
        if domain == "product" and intent == "concept_explanation":
            if question_type in {"support_matrix", "version_support", "capacity_or_scale"}:
                if best_score < 0.5:
                    return {
                        "status": "low_similarity",
                        "answer": None,
                        "lookup_key_used": candidates[0] if candidates else None,
                        "confidence": best_score,
                        "similarity": best_score,
                        "reason": "best similarity below strict product threshold",
                    }
                if best_score - runner_up_score < 0.05:
                    return {
                        "status": "needs_disambiguation",
                        "answer": None,
                        "lookup_key_used": candidates[0] if candidates else None,
                        "confidence": best_score,
                        "similarity": best_score,
                        "reason": "multiple close answers",
                    }
            return {
                "status": "found",
                "answer": _clean(best_entry.answer),
                "lookup_key_used": candidates[0] if candidates else None,
                "confidence": best_score,
                "similarity": best_score,
                "reason": "best product explanation match",
            }
        if best_score - runner_up_score < 0.05:
            return {
                "status": "needs_disambiguation",
                "answer": None,
                "lookup_key_used": candidates[0] if candidates else None,
                "confidence": best_score,
                "similarity": best_score,
                "reason": "multiple close answers",
            }
        return {
            "status": "found",
            "answer": _clean(best_entry.answer),
            "lookup_key_used": candidates[0] if candidates else None,
            "confidence": best_score,
            "similarity": best_score,
            "reason": None,
        }

    if exact_product_intent:
        return {
            "status": "not_found",
            "answer": None,
            "lookup_key_used": candidates[0] if candidates else None,
            "confidence": 0.0,
            "similarity": 0.0,
            "reason": "no exact product syntax match",
        }

    intent_candidates = [entry for entry in entries if _clean(entry.intent) == _clean(intent)]
    ranked_pool = intent_candidates or list(entries)
    ranked = _rank_entries(domain, question, slots, ranked_pool)
    if not ranked:
        return {
            "status": "not_found",
            "answer": None,
            "lookup_key_used": candidates[0] if candidates else None,
            "confidence": 0.0,
            "similarity": 0.0,
            "reason": "no candidates",
        }
    best_entry, best_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score < 0.56:
        return {
            "status": "low_similarity",
            "answer": None,
            "lookup_key_used": candidates[0] if candidates else None,
            "confidence": best_score,
            "similarity": best_score,
            "reason": "best similarity below threshold",
        }
    if best_score - runner_up_score < 0.05 and len(ranked) > 1:
        return {
            "status": "needs_disambiguation",
            "answer": None,
            "lookup_key_used": candidates[0] if candidates else None,
            "confidence": best_score,
            "similarity": best_score,
            "reason": "similar candidates",
        }
    return {
        "status": "found",
        "answer": _clean(best_entry.answer),
        "lookup_key_used": candidates[0] if candidates else None,
        "confidence": best_score,
        "similarity": best_score,
        "reason": None,
    }


def _is_no_workaround(answer: str) -> bool:
    text = _clean(answer).lower()
    return "no workaround is documented" in text or text == "no workaround is documented in the release notes."


def _should_use_qwen(domain: str, intent: str, lookup_answer: str) -> bool:
    answer = _clean(lookup_answer)
    if not answer:
        return False
    if is_cli_syntax_answer(answer, intent):
        return False
    if _is_no_workaround(answer):
        return False
    if domain == "release" and intent in {"bug_category", "version_date", "release_date", "event_id", "cli_syntax", "show_command_syntax"}:
        return False
    if domain == "product":
        if intent in PRODUCT_EXACT_INTENTS:
            return False
        if intent == "concept_explanation":
            return True
        return len(answer.split()) > 8
    if len(answer.split()) < 8:
        return False
    return True


def _build_qwen_prompt(
    domain: str,
    question: str,
    predicted_intent: str,
    slots: Dict[str, str],
    lookup_answer: str,
    previous_context: Optional[Dict[str, str]] = None,
) -> str:
    title = "release-note" if domain == "release" else "product documentation"
    source_type = predicted_intent or "response_formatter"
    data_family = "release_notes" if domain == "release" else "product_documentation"
    if domain == "product":
        extra_guidance = (
            "For product documentation, keep the full grounded meaning and make it easier to read.\n"
            "Do not truncate the answer.\n"
            "If the grounded answer contains multiple facts, types, methods, goals, requirements, conditions, or steps, format them as bullet points or numbered steps.\n"
            "If the grounded answer contains commands, wrap the command text in backticks.\n"
            "If the user asks what a command does, explain the purpose only if the grounded answer already includes that purpose.\n"
            "If the grounded answer only contains syntax, say that only syntax was found.\n"
            "Avoid filler prefixes like 'The documented answer is' or 'According to the documentation'.\n"
            "Write concise bullet points that preserve every factual detail from the grounded answer.\n"
            "Do not add any facts that are not already grounded.\n"
        )
    else:
        extra_guidance = (
            "Keep the answer grounded and precise.\n"
            "Use short headings, bullets, or numbered steps only when they improve readability.\n"
            "Prefer bullets for multiple factual items.\n"
        )
    if domain == "product" and previous_context:
        previous_question = _clean(previous_context.get("last_question", ""))
        previous_lookup_answer = _clean(previous_context.get("last_lookup_answer", ""))
        previous_final_answer = _clean(previous_context.get("last_final_answer", ""))
        if previous_question and (previous_lookup_answer or previous_final_answer):
            return (
                "You are an HPE Aruba AOS-CX product documentation response formatter.\n\n"
                "Facts come only from the previous retrieved answer and previous final answer.\n"
                "You must not answer from your own knowledge.\n"
                "Your job is to answer the follow-up clearly and conversationally using only the previous context.\n"
                "Do not add new facts.\n"
                "Do not truncate the answer.\n"
                "If the previous answer explains multiple types or conditions, keep all of them and format them as bullets when helpful.\n"
                "If the previous context is missing, ask the user to specify the topic.\n\n"
                f"Previous question:\n{previous_question}\n\n"
                f"Previous retrieved answer:\n{previous_lookup_answer}\n\n"
                f"Previous final answer:\n{previous_final_answer}\n\n"
                f"Follow-up question:\n{_clean(question)}\n\n"
                "Task:\n"
                "Answer the follow-up using only the previous retrieved answer and previous final answer.\n"
                "Use bullets if it helps explain multiple documented items.\n"
                "Return only the final formatted answer."
            )
    return (
        f"You are an HPE Aruba AOS-CX {title} assistant.\n\n"
        "Use only the grounded answer provided below.\n"
        "Do not invent facts.\n"
        "Do not change Bug IDs, categories, versions, commands, workarounds, symptoms, scenarios, caveats, or feature names.\n"
        "If the grounded answer says no workaround is documented, preserve that meaning exactly.\n\n"
        f"{extra_guidance}\n"
        f"Question:\n{_clean(question)}\n\n"
        f"Predicted intent:\n{_clean(predicted_intent)}\n\n"
        f"Slots:\n{json.dumps(slots, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Metadata:\nSwitch: {_clean(slots.get('switch', ''))}\nVersion: {_clean(slots.get('version', ''))}\nSub-version: {_clean(slots.get('sub_version', ''))}\nSource type: {source_type}\nData family: {data_family}\n\n"
        f"Retrieved answer:\n{_prompt_safe_text(lookup_answer)}\n\n"
        "Task:\nAnswer the user naturally using only the grounded answer.\n"
        "Write a neat, complete response.\n"
        "If the grounded answer lists multiple documented items, format them as bullets.\n"
        "If the grounded answer is short, restate it clearly and do not add unsupported facts.\n"
        "Do not truncate the answer."
    )


FINAL_RESPONSE_SYSTEM_PROMPT = (
    "You are the final answer generation model for an Aruba AOS-CX QA assistant.\n"
    "Use only the provided grounded data.\n"
    "Do not invent facts.\n"
    "If the answer is not grounded, explain that the current dataset does not contain a reliable exact answer.\n"
    "If details are missing, ask a clarification question.\n"
    "If related context is provided, summarize it carefully and mention that it is related, not an exact match.\n"
    "Do not create new switch support, version support, route scale, CLI syntax, command output, or bug facts from memory.\n"
)


QUESTION_UNDERSTANDING_SYSTEM_PROMPT = (
    "You are a question-understanding assistant for an Aruba AOS-CX RAG backend.\n"
    "Your job is to understand the user question and return strict JSON only.\n"
    "Do not answer the question.\n"
    "Do not add markdown.\n"
    "Do not add explanatory text.\n"
    "Return only valid JSON with the requested fields.\n"
)


def _extract_json_object(text: str) -> Dict[str, object]:
    raw = _clean(text)
    if not raw:
        return {}
    candidates: List[str] = []
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())
    brace = re.search(r"\{[\s\S]*\}", raw)
    if brace:
        candidates.append(brace.group(0).strip())
    candidates.append(raw)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            try:
                payload = ast.literal_eval(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
    return {}


def _string_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [_clean(item) for item in value if _clean(item)]
    if isinstance(value, str):
        return [_clean(item) for item in re.split(r"[,\n;]+", value) if _clean(item)]
    return []


def _llm_understand_question(
    domain_hint: str,
    question: str,
    session_context: Dict[str, Optional[str]],
    selected_context: Dict[str, str],
    qwen_bundle: Optional[QwenBundle],
    device: Optional[torch.device],
) -> Dict[str, object]:
    if qwen_bundle is None or not qwen_bundle.loaded or qwen_bundle.model is None or qwen_bundle.tokenizer is None or device is None:
        return {}
    question_text = _clean(question)
    if not question_text:
        return {}
    previous_question = _clean(session_context.get("last_question", ""))
    previous_answer = _clean(session_context.get("last_final_answer", "")) or _clean(session_context.get("last_lookup_answer", ""))
    prompt = (
        f"{QUESTION_UNDERSTANDING_SYSTEM_PROMPT}\n"
        "Return a JSON object with these keys:\n"
        "{\n"
        '  "domain": "product|release|unified",\n'
        '  "intent": "string",\n'
        '  "question_type": "string",\n'
        '  "topic_family": "string",\n'
        '  "candidate_topics": ["string"],\n'
        '  "candidate_intents": ["string"],\n'
        '  "required_slots": ["string"],\n'
        '  "needs_clarification": true,\n'
        '  "clarification_question": "string",\n'
        '  "lookup_query": "string",\n'
        '  "is_command_purpose": false,\n'
        '  "slots": {"switch":"", "version":"", "sub_version":"", "feature":"", "category":"", "topic":"", "command":"", "route_type":"", "bug_id":"", "event_id":""}\n'
        "}\n"
        "Use only the user question and the conversation context.\n"
        "If the question is a follow-up, infer the missing detail from the previous context when possible.\n"
        "If clarification is needed, set needs_clarification to true and provide a short natural clarification_question.\n"
        "If the question is clear, set needs_clarification to false.\n"
        "Use concise values. Return only JSON.\n\n"
        f"Domain hint: {_clean(domain_hint)}\n\n"
        f"Conversation history:\nPrevious question: {previous_question}\nPrevious answer: {previous_answer}\nSelected context: {json.dumps(selected_context, ensure_ascii=False)}\n\n"
        f"Current question:\n{question_text}"
    )
    try:
        response = generate_qwen_answer(
            qwen_bundle.tokenizer,
            qwen_bundle.model,
            prompt,
            domain_hint or "product",
            device,
            data_family="product_documentation" if _clean(domain_hint).lower() != "release" else "release_notes",
            system_prompt=QUESTION_UNDERSTANDING_SYSTEM_PROMPT,
        )
    except Exception:
        return {}
    payload = _extract_json_object(response)
    if not payload:
        return {}
    slots = payload.get("slots")
    if isinstance(slots, dict):
        normalized_slots: Dict[str, str] = {}
        for key in ("switch", "version", "sub_version", "feature", "category", "topic", "command", "route_type", "bug_id", "event_id"):
            value = _clean(slots.get(key, ""))
            if value:
                normalized_slots[key] = value
        payload["slots"] = normalized_slots
    else:
        payload["slots"] = {}
    payload["candidate_topics"] = _string_list(payload.get("candidate_topics"))
    payload["candidate_intents"] = _string_list(payload.get("candidate_intents"))
    payload["required_slots"] = _string_list(payload.get("required_slots"))
    payload["domain"] = _clean(payload.get("domain", domain_hint)).lower()
    payload["intent"] = _clean(payload.get("intent", ""))
    payload["question_type"] = _clean(payload.get("question_type", ""))
    payload["topic_family"] = _clean(payload.get("topic_family", ""))
    payload["lookup_query"] = _clean(payload.get("lookup_query", question_text))
    payload["clarification_question"] = _clean(payload.get("clarification_question", ""))
    payload["needs_clarification"] = bool(payload.get("needs_clarification", False))
    payload["is_command_purpose"] = bool(payload.get("is_command_purpose", False))
    return payload


def _qwen_finalization_safe_fallback(
    domain: str,
    lookup_status: str,
    question_type: str = "",
    contamination_detected: bool = False,
) -> str:
    domain_key = _clean(domain).lower()
    status = _clean(lookup_status).lower()
    question_kind = _clean(question_type).lower()
    if domain_key == "release":
        if status == "not_found":
            return "No matching answer was found in the current Aruba AOS-CX dataset."
        if status == "low_similarity":
            return "I found related documentation, but not a reliable exact match."
        if status == "needs_disambiguation":
            return "Multiple possible answers were found. Please provide more detail such as feature, bug ID, command, version, or sub-version."
        if status == "slot_missing":
            return "I need more detail to answer this, such as the bug ID, feature, command, version, or sub-version."
        return "Unable to answer from the current release-note dataset."
    if status == "data_not_available":
        return PRODUCT_DATANOT_AVAILABLE_RESPONSE
    if status == "not_found":
        if contamination_detected:
            if question_kind == "cli_syntax":
                return PRODUCT_SYNTAX_MATCH_RESPONSE
            if question_kind == "cli_output":
                return PRODUCT_COMMAND_OUTPUT_RESPONSE
            return PRODUCT_CONTAMINATED_RESPONSE
        return PRODUCT_NOT_FOUND_RESPONSE
    if status == "low_similarity":
        return "I found related documentation, but not a reliable exact match."
    if status == "needs_disambiguation":
        return PRODUCT_NEEDS_DISAMBIGUATION_RESPONSE
    if status == "slot_missing":
        return PRODUCT_SLOT_MISSING_RESPONSE
    return _format_deterministic("product", status)


def _qwen_finalization_no_fact_answer_is_safe(
    lookup_status: str,
    qwen_answer: str,
    contamination_detected: bool = False,
) -> bool:
    text = _clean(qwen_answer).lower()
    if not text:
        return False
    status = _clean(lookup_status).lower()
    if contamination_detected:
        return any(
            marker in text
            for marker in (
                "cannot safely",
                "index",
                "artifact",
                "related documentation",
                "cannot use",
            )
        )
    if status == "not_found":
        return any(marker in text for marker in ("not available", "no matching answer", "current dataset", "could not find"))
    if status == "slot_missing":
        return any(
            marker in text
            for marker in (
                "please specify",
                "need more detail",
                "which switch",
                "which version",
                "current dataset",
                "not available",
            )
        )
    if status == "needs_disambiguation":
        return any(marker in text for marker in ("multiple possible answers", "provide more detail", "not enough detail"))
    if status == "low_similarity":
        return any(marker in text for marker in ("related documentation", "not a reliable exact match", "not an exact match"))
    if status == "data_not_available":
        return any(marker in text for marker in ("not available", "current dataset"))
    return bool(text)


def _build_final_response_qwen_prompt(
    question: str,
    intent: str,
    slots: Dict[str, str],
    lookup_status: str,
    target_value: Optional[str] = None,
    related_context: Optional[str] = None,
    rejection_reason: Optional[str] = None,
    contamination_detected: bool = False,
    data_family: str = "product_documentation",
    source_type: str = "",
    question_type: str = "",
) -> str:
    return (
        "Question:\n"
        f"{_clean(question)}\n\n"
        "Predicted intent:\n"
        f"{_clean(intent)}\n\n"
        "Extracted slots:\n"
        f"{json.dumps(slots or {}, ensure_ascii=False, sort_keys=True)}\n\n"
        "Lookup status:\n"
        f"{_clean(lookup_status)}\n\n"
        "Grounded target value:\n"
        f"{_prompt_safe_text(target_value or 'NONE')}\n\n"
        "Related context:\n"
        f"{_prompt_safe_text(related_context or 'NONE')}\n\n"
        "Rejection reason:\n"
        f"{_prompt_safe_text(rejection_reason or 'NONE')}\n\n"
        "Contamination detected:\n"
        f"{'true' if contamination_detected else 'false'}\n\n"
        "Metadata:\n"
        f"Source type: {_clean(source_type)}\n"
        f"Data family: {_clean(data_family)}\n"
        f"Question type: {_clean(question_type)}\n\n"
        "Task:\n"
        "Return the final user-facing answer only.\n"
        "Use only the grounded information above.\n"
        "If the lookup status is found, explain the grounded answer clearly.\n"
        "If the lookup status is low_similarity, say that the answer is related but not exact.\n"
        "If the lookup status is not_found, say the current dataset does not contain a reliable exact answer.\n"
        "If the lookup status is slot_missing, ask for the missing detail or say the current dataset does not contain a reliable exact answer and request the missing slot.\n"
        "If the retrieved text looks contaminated or unsafe, do not use it as facts.\n"
        "Do not invent switch support, version support, route scale, CLI syntax, command output, bug facts, or workaround facts."
    )


def finalize_answer_with_qwen(
    question: str,
    intent: str,
    slots: Dict[str, str],
    lookup_status: str,
    target_value: Optional[str] = None,
    related_context: Optional[str] = None,
    rejection_reason: Optional[str] = None,
    contamination_detected: bool = False,
    *,
    qwen_bundle: Optional[QwenBundle] = None,
    device: Optional[torch.device] = None,
    data_family: str = "product_documentation",
    source_type: str = "",
    question_type: str = "",
    fallback_answer: str = "",
) -> Dict[str, object]:
    fallback_text = _cleanup_product_markdown(fallback_answer)
    if not fallback_text:
        fallback_text = _qwen_finalization_safe_fallback(
            "release" if _clean(data_family).lower() == "release_notes" else "product",
            lookup_status,
            question_type=question_type,
            contamination_detected=contamination_detected,
        )

    qwen_enabled = (
        QWEN_FINALIZE_ALL_RESPONSES
        and qwen_bundle is not None
        and qwen_bundle.loaded
        and qwen_bundle.model is not None
        and qwen_bundle.tokenizer is not None
        and device is not None
    )
    if not qwen_enabled:
        return {
            "final_answer": fallback_text,
            "qwen_used": False,
            "qwen_answer": None,
            "qwen_validation_passed": False,
            "answer_source": "lookup_fallback",
            "validation_reason": "qwen finalization disabled or unavailable",
        }

    grounding_text = ""
    if target_value and not contamination_detected:
        grounding_text = _clean(target_value)
    elif related_context and not contamination_detected:
        grounding_text = _clean(related_context)

    prompt = _build_final_response_qwen_prompt(
        question=question,
        intent=intent,
        slots=slots,
        lookup_status=lookup_status,
        target_value=target_value,
        related_context=related_context,
        rejection_reason=rejection_reason,
        contamination_detected=contamination_detected,
        data_family=data_family,
        source_type=source_type,
        question_type=question_type,
    )
    try:
        qwen_answer = generate_qwen_answer(
            qwen_bundle.tokenizer,
            qwen_bundle.model,
            prompt,
            intent,
            device,
            data_family=data_family,
            system_prompt=FINAL_RESPONSE_SYSTEM_PROMPT,
        )
        if grounding_text:
            qwen_validation_passed, validation_reason = validate_qwen_answer(
                intent,
                slots,
                grounding_text,
                qwen_answer,
                data_family=data_family,
            )
            if contamination_detected:
                qwen_validation_passed = False
                validation_reason = "contaminated retrieved text"
        else:
            qwen_validation_passed = _qwen_finalization_no_fact_answer_is_safe(
                lookup_status,
                qwen_answer,
                contamination_detected=contamination_detected,
            )
            validation_reason = "safe fallback" if qwen_validation_passed else "unsafe fallback"

        if qwen_validation_passed:
            return {
                "final_answer": qwen_answer,
                "qwen_used": True,
                "qwen_answer": qwen_answer,
                "qwen_validation_passed": True,
                "answer_source": "qwen_finalized",
                "validation_reason": validation_reason,
            }
        return {
            "final_answer": fallback_text,
            "qwen_used": True,
            "qwen_answer": qwen_answer,
            "qwen_validation_passed": False,
            "answer_source": "lookup_fallback",
            "validation_reason": validation_reason,
        }
    except Exception as exc:  # pragma: no cover - runtime/model dependent
        return {
            "final_answer": fallback_text,
            "qwen_used": True,
            "qwen_answer": None,
            "qwen_validation_passed": False,
            "answer_source": "lookup_fallback",
            "validation_reason": str(exc),
        }


def _finalize_answer_payload(
    payload: Dict[str, object],
    *,
    question: str,
    intent: str,
    slots: Dict[str, str],
    lookup_status: str,
    target_value: Optional[str] = None,
    related_context: Optional[str] = None,
    rejection_reason: Optional[str] = None,
    contamination_detected: bool = False,
    qwen_bundle: Optional[QwenBundle] = None,
    device: Optional[torch.device] = None,
    data_family: str = "product_documentation",
    source_type: str = "",
    question_type: str = "",
) -> Dict[str, object]:
    finalized = finalize_answer_with_qwen(
        question,
        intent,
        slots,
        lookup_status,
        target_value=target_value,
        related_context=related_context,
        rejection_reason=rejection_reason,
        contamination_detected=contamination_detected,
        qwen_bundle=qwen_bundle,
        device=device,
        data_family=data_family,
        source_type=source_type,
        question_type=question_type,
        fallback_answer=_clean(payload.get("final_answer", "")),
    )
    payload = dict(payload)
    payload["final_answer"] = finalized["final_answer"]
    payload["qwen_used"] = finalized["qwen_used"]
    payload["qwen_answer"] = finalized["qwen_answer"]
    payload["qwen_validation_passed"] = finalized["qwen_validation_passed"]
    payload["answer_source"] = finalized["answer_source"]
    debug = payload.get("debug")
    if not isinstance(debug, dict):
        debug = {}
    debug["qwen_finalization"] = {
        "enabled": QWEN_FINALIZE_ALL_RESPONSES,
        "lookup_status": lookup_status,
        "validation_reason": finalized.get("validation_reason"),
        "used": finalized["qwen_used"],
    }
    payload["debug"] = debug
    return payload


def _session_template() -> Dict[str, Optional[str]]:
    return {
        "last_question": None,
        "last_final_answer": None,
        "last_lookup_answer": None,
        "last_source_type": None,
        "last_data_family": None,
        "last_bug_id": None,
        "last_valid_bug_id": None,
        "last_switch": None,
        "last_version": None,
        "last_sub_version": None,
        "last_feature": None,
        "last_category": None,
        "last_command": None,
        "last_topic": None,
        "last_event_id": None,
        "last_intent": None,
        "last_domain": None,
    }


def _update_session_context(
    session_context: Dict[str, Optional[str]],
    question: str,
    slots: Dict[str, str],
    predicted_intent: str,
    lookup_answer: str,
    final_answer: str,
    source_type: str,
    data_family: str,
) -> None:
    session_context["last_question"] = _clean(question)
    for key in ["bug_id", "switch", "version", "sub_version", "feature", "category", "command", "topic", "event_id"]:
        if slots.get(key):
            session_context[f"last_{key}"] = slots[key]
    if slots.get("bug_id"):
        session_context["last_bug_id"] = slots["bug_id"]
        session_context["last_valid_bug_id"] = slots["bug_id"]
    session_context["last_intent"] = predicted_intent
    session_context["last_lookup_answer"] = lookup_answer
    session_context["last_final_answer"] = final_answer
    session_context["last_source_type"] = source_type
    session_context["last_data_family"] = data_family


def _reuse_session_slots(slots: Dict[str, str], session_context: Dict[str, Optional[str]]) -> Dict[str, str]:
    effective = dict(slots)
    if not effective.get("bug_id") and _clean(session_context.get("last_bug_id")):
        effective["bug_id"] = _clean(session_context.get("last_bug_id"))
    for key in ["switch", "version", "sub_version", "feature", "category", "command", "topic", "event_id"]:
        if not effective.get(key) and _clean(session_context.get(f"last_{key}")):
            value = _clean(session_context.get(f"last_{key}"))
            if key == "version":
                value = _domain_version(value, "product")
            effective[key] = value
    return effective


def _format_deterministic(domain: str, status: str) -> str:
    if domain == "release":
        if status == "not_found":
            return "No matching answer was found in the current Aruba AOS-CX dataset."
        if status == "low_similarity":
            return "I found related documentation, but not a reliable exact match."
        if status == "needs_disambiguation":
            return "Multiple possible answers were found. Please provide more detail such as feature, bug ID, command, version, or sub-version."
        if status == "slot_missing":
            return "I need more detail to answer this, such as the bug ID, feature, command, version, or sub-version."
        return "Unable to answer from the current release-note dataset."
    if status == "not_found":
        return PRODUCT_NOT_FOUND_RESPONSE
    if status == "low_similarity":
        return "I found related documentation, but not a reliable exact match."
    if status == "needs_disambiguation":
        return PRODUCT_NEEDS_DISAMBIGUATION_RESPONSE
    if status == "slot_missing":
        return PRODUCT_SLOT_MISSING_RESPONSE
    return "Unable to answer from the current product documentation dataset."


def _polish_product_answer(answer: str, intent: str, slots: Optional[Dict[str, str]] = None) -> str:
    text = _cleanup_product_markdown(_strip_product_filler_prefix(answer))
    if not text:
        return text
    if intent in PRODUCT_EXACT_INTENTS or _is_no_workaround(text):
        return text
    if intent == "concept_explanation":
        return _format_product_concept_answer(text, slots or {})
    if text[-1] not in ".!?":
        return f"{text}."
    return text


def _looks_like_followup_question(question: str) -> bool:
    text = _clean(question).lower()
    if any(phrase in text for phrase in PRODUCT_FOLLOWUP_WORDS):
        return True
    if text in {"this", "that", "it", "explain", "elaborate"}:
        return True
    if len(text.split()) <= 10 and re.search(r"\b(this|that|those|these|it)\b", text):
        return True
    return False


def _is_product_followup(question: str) -> bool:
    return _looks_like_followup_question(question)


def _product_meaningful_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", _clean(text).lower())
    return [token for token in tokens if token not in PRODUCT_QWEN_STOPWORDS]


def _product_qwen_is_too_drifty(lookup_answer: str, qwen_answer: str) -> bool:
    candidate = _clean(qwen_answer)
    if not candidate:
        return True
    if re.search(r"\b[a-z0-9_-]+\(config\)#", candidate, flags=re.IGNORECASE):
        return True
    if "config)#" in candidate.lower():
        return True
    original_tokens = set(_product_meaningful_tokens(lookup_answer))
    candidate_tokens = set(_product_meaningful_tokens(candidate))
    if not original_tokens or not candidate_tokens:
        return True
    overlap = len(original_tokens & candidate_tokens) / max(1, len(original_tokens))
    return overlap < 0.6


def _product_relevant_snippet(answer: str, keywords: Sequence[str], limit: int = 8) -> str:
    text = _cleanup_product_markdown(_strip_product_filler_prefix(answer))
    if not text:
        return ""
    lowered_keywords = [_clean(keyword).lower() for keyword in keywords if _clean(keyword)]
    if not lowered_keywords:
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    scored_lines: List[Tuple[int, int, str]] = []
    seen: set[str] = set()
    for line in lines:
        low = line.lower()
        hit_count = sum(1 for keyword in lowered_keywords if keyword in low)
        if hit_count and line not in seen:
            priority = 0
            if "not applicable" in low or "not supported" in low or "unsupported" in low:
                priority += 3
            if "maximum number" in low or "show capacities" in low or "show capacities-status" in low:
                priority += 2
            if "support" in low or "supported" in low:
                priority += 1
            if len(line) > 300:
                sentence_hits: List[Tuple[int, int, str]] = []
                for sentence in re.split(r"(?<=[.!?])\s+", line):
                    cleaned_sentence = sentence.strip()
                    if not cleaned_sentence:
                        continue
                    sentence_low = cleaned_sentence.lower()
                    sentence_hit_count = sum(1 for keyword in lowered_keywords if keyword in sentence_low)
                    if not sentence_hit_count:
                        continue
                    sentence_priority = 0
                    if "not applicable" in sentence_low or "not supported" in sentence_low or "unsupported" in sentence_low:
                        sentence_priority += 3
                    if "maximum number" in sentence_low or "show capacities" in sentence_low or "show capacities-status" in sentence_low:
                        sentence_priority += 2
                    if "support" in sentence_low or "supported" in sentence_low:
                        sentence_priority += 1
                    sentence_hits.append((sentence_priority + sentence_hit_count, -len(cleaned_sentence), cleaned_sentence))
                if sentence_hits:
                    sentence_hits.sort(reverse=True)
                    for item in sentence_hits[: max(1, min(limit, 3))]:
                        candidate = item[2]
                        if candidate not in seen:
                            seen.add(candidate)
                            scored_lines.append(item)
                    continue
            seen.add(line)
            scored_lines.append((priority + hit_count, -len(line), line))
    if scored_lines:
        scored_lines.sort(reverse=True)
        selected = "\n".join(line for _score, _len, line in scored_lines[: max(1, min(limit, 3))])
        if len(selected) > 700 and "\n" not in selected:
            sentences = re.split(r"(?<=[.!?])\s+", selected)
            sentence_scored: List[Tuple[int, int, str]] = []
            for sentence in sentences:
                cleaned_sentence = sentence.strip()
                if not cleaned_sentence:
                    continue
                sentence_low = cleaned_sentence.lower()
                hit_count = sum(1 for keyword in lowered_keywords if keyword in sentence_low)
                if not hit_count:
                    continue
                sentence_priority = 0
                if "not applicable" in sentence_low or "not supported" in sentence_low or "unsupported" in sentence_low:
                    sentence_priority += 3
                if "maximum number" in sentence_low or "show capacities" in sentence_low or "show capacities-status" in sentence_low:
                    sentence_priority += 2
                if "support" in sentence_low or "supported" in sentence_low:
                    sentence_priority += 1
                sentence_scored.append((sentence_priority + hit_count, -len(cleaned_sentence), cleaned_sentence))
            if sentence_scored:
                sentence_scored.sort(reverse=True)
                return " ".join(sentence for _score, _len, sentence in sentence_scored[: max(1, min(limit, 3))])
        return selected

    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored_sentences: List[Tuple[int, int, str]] = []
    seen_sentences: set[str] = set()
    for sentence in sentences:
        cleaned_sentence = sentence.strip()
        if not cleaned_sentence:
            continue
        low = cleaned_sentence.lower()
        hit_count = sum(1 for keyword in lowered_keywords if keyword in low)
        if hit_count and cleaned_sentence not in seen_sentences:
            seen_sentences.add(cleaned_sentence)
            priority = 0
            if "not applicable" in low or "not supported" in low or "unsupported" in low:
                priority += 3
            if "maximum number" in low or "show capacities" in low or "show capacities-status" in low:
                priority += 2
            if "support" in low or "supported" in low:
                priority += 1
            scored_sentences.append((priority + hit_count, -len(cleaned_sentence), cleaned_sentence))
    if scored_sentences:
        scored_sentences.sort(reverse=True)
        return " ".join(sentence for _score, _len, sentence in scored_sentences[: max(1, min(limit, 3))])
    return text


def _product_strip_toc_artifacts(text: str) -> str:
    cleaned_lines: List[str] = []
    for line in [segment.strip() for segment in _clean(text).splitlines()]:
        if not line:
            continue
        low = line.lower()
        if "table of contents" in low:
            continue
        if re.search(r"\.{6,}\s*\d+\s*$", line):
            continue
        if re.match(r"^\d+\s*$", line):
            continue
        cleaned_lines.append(line)
    if cleaned_lines:
        return "\n".join(cleaned_lines)
    return _clean(text)


def _format_product_concept_answer(answer: str, slots: Dict[str, str]) -> str:
    text = _cleanup_product_markdown(_strip_product_filler_prefix(answer))
    if not text:
        return ""
    question_type = _clean(slots.get("question_type", ""))
    feature_keywords = _product_support_feature_aliases(_clean(slots.get("feature", "")))
    topic_keywords = _unique([
        _clean(slots.get("topic", "")),
        _clean(slots.get("category", "")),
        _clean(slots.get("command", "")),
    ])
    if question_type in {"support_matrix", "version_support", "capacity_or_scale"}:
        if question_type == "capacity_or_scale":
            topic_hint = _clean(slots.get("topic", ""))
            feature_hint = _clean(slots.get("feature", ""))
            keywords = [
                "capacity",
                "supported",
                "maximum",
                "range",
                "limit",
                "scale",
                "member",
                "entries",
                "route scale",
                "route capacity",
                "maximum number of routes",
                "maximum number of ipv4 routes",
                "maximum number of ipv6 routes",
                "show capacities",
                "show capacities-status",
                "show capacities rpvst",
                "show resources",
                "long prefix ipv6 route capacity",
                "route table",
            ]
            route_type = _clean(slots.get("route_type", "")).lower()
            if route_type == "ipv4":
                keywords.extend(["ipv4 route scale", "ipv4 routes", "maximum number of ipv4 routes"])
            elif route_type == "ipv6":
                keywords.extend(["ipv6 route scale", "ipv6 routes", "maximum number of ipv6 routes", "long prefix ipv6 route capacity"])
            else:
                keywords.extend(["ipv4 routes", "ipv6 routes", "long prefix ipv6 route capacity"])
            if topic_hint:
                keywords.extend([topic_hint, f"{topic_hint} capacity", f"{topic_hint} range", f"{topic_hint} limit"])
            if feature_hint:
                keywords.append(feature_hint)
            focus_limit = 4
        else:
            keywords = [
                "version",
                "introduced",
                "compatible",
                "not applicable",
                "not supported",
                "unsupported",
                "support",
                "supported",
            ]
            focus_limit = 3
        focus = _product_relevant_snippet(text, _unique(keywords + feature_keywords + topic_keywords), limit=focus_limit)
        if focus:
            text = _product_strip_toc_artifacts(focus)
    if text.startswith(("**", "-", "1.", "*")) or "\n" in text:
        return text

    structured_segments = _product_structured_segments(text)
    if len(structured_segments) > 1:
        topic = _clean(slots.get("topic", "") or slots.get("feature", "") or slots.get("category", ""))
        lines: List[str] = []
        if topic:
            lines.append(f"**{topic}**")
            lines.append("")
        lines.extend(f"- {segment}" for segment in structured_segments)
        return _cleanup_product_markdown("\n".join(lines))

    sentences = _product_sentence_chunks(text)
    if len(sentences) <= 1:
        return text if text.endswith((".", "!", "?")) else f"{text}."

    topic = _clean(slots.get("topic", "") or slots.get("feature", "") or slots.get("category", ""))
    lead_sentence = sentences[0]
    list_like = bool(
        re.search(r"\b(two types|three types|four types|includes|consists|contains|requirements|conditions|methods|goals|steps|overview)\b", lead_sentence, flags=re.IGNORECASE)
        or ":" in lead_sentence
        or any(sentence[:1].isupper() and len(sentence.split()) <= 18 for sentence in sentences[1:])
    )

    if not list_like:
        return text if text.endswith((".", "!", "?")) else f"{text}."

    lines: List[str] = []
    if topic:
        lines.append(f"**{topic}**")
        lines.append("")
    if len(sentences) == 2 and re.search(r"\b(types?|methods?|steps?|requirements?|conditions?|goals?)\b", lead_sentence, flags=re.IGNORECASE):
        lines.append(lead_sentence)
        lines.append("")
        lines.extend(f"- {sentence}" for sentence in sentences[1:])
    else:
        lines.extend(f"- {sentence}" for sentence in sentences)
    return _cleanup_product_markdown("\n".join(line for line in lines if line is not None))


@dataclass
class QwenBundle:
    tokenizer: Any = None
    model: Any = None
    requested_path: str = ""
    resolved_path: str = ""
    model_kind: str = "disabled"
    resolution_reason: str = ""
    base_model_name: str = ""
    loaded: bool = False
    error: str = ""


@dataclass
class LstmBundle:
    model: Any = None
    tokenizer: Any = None
    config: Dict[str, object] = field(default_factory=dict)
    requested_path: str = ""
    resolved_path: str = ""
    loaded: bool = False
    error: str = ""


@dataclass
class ReleaseRuntime:
    model_path: Path
    lookup_data_path: Path
    lookup_index_path: Path
    availability_path: Path
    bug_metadata_path: Path
    device: torch.device
    qwen: QwenBundle
    lstm_model: Any = field(init=False)
    lstm_tokenizer: Any = field(init=False)
    lstm_config: Dict[str, object] = field(init=False)
    lookup_entries: List[Any] = field(init=False)
    lookup_index: Dict[str, List[int]] = field(init=False)
    availability_index: Dict[str, object] = field(init=False)
    bug_metadata_index: Dict[str, List[Dict[str, str]]] = field(init=False)

    def __post_init__(self) -> None:
        self.lookup_entries, self.lookup_index = load_lookup_resources(self.lookup_index_path, self.lookup_data_path)
        self.availability_index = load_or_build_availability_index(self.availability_path, self.lookup_entries)
        self.bug_metadata_index = load_or_build_bug_metadata_index(self.bug_metadata_path, self.lookup_entries)
        self.lstm_model, self.lstm_tokenizer, self.lstm_config = load_lstm_support(self.model_path, self.device)

    def answer(
        self,
        question: str,
        session_context: Dict[str, Optional[str]],
        selected_context: Dict[str, str],
        show_debug: bool = False,
        override_intent: Optional[str] = None,
    ) -> Dict[str, object]:
        result = answer_release_question(
            question,
            self.lstm_model,
            self.lstm_tokenizer,
            self.lstm_config,
            self.lookup_entries,
            self.lookup_index,
            self.availability_index,
            self.bug_metadata_index,
            self.qwen.tokenizer,
            self.qwen.model,
            self.device,
            session_context,
            override_intent=override_intent,
        )
        result = _finalize_answer_payload(
            result,
            question=result.get("question", _clean(question)),
            intent=str(result.get("predicted_intent") or ""),
            slots=result.get("slots", {}) if isinstance(result.get("slots"), dict) else {},
            lookup_status=str(result.get("lookup_status") or "error"),
            target_value=_clean(result.get("lookup_answer", "")) or None,
            related_context=_clean(result.get("lookup_answer", "")) if str(result.get("lookup_status") or "") in {"low_similarity", "needs_disambiguation"} else None,
            rejection_reason=str(result.get("validation_reason") or result.get("lookup_status") or ""),
            contamination_detected=False,
            qwen_bundle=self.qwen,
            device=self.device,
            data_family="release_notes",
            source_type=str(result.get("source_type") or ""),
            question_type=str(result.get("predicted_intent") or ""),
        )
        return {
            "domain": "release",
            "question": result.get("question", _clean(question)),
            "predicted_intent": result.get("predicted_intent"),
            "raw_lstm_intent": result.get("raw_lstm_intent"),
            "slots": result.get("slots", {}),
            "lookup_status": result.get("lookup_status"),
            "lookup_key_used": result.get("lookup_key_used"),
            "lookup_answer": result.get("lookup_answer"),
            "qwen_used": result.get("qwen_used"),
            "qwen_answer": result.get("qwen_answer"),
            "qwen_validation_passed": result.get("qwen_validation_passed"),
            "final_answer": result.get("final_answer"),
            "answer_source": result.get("answer_source"),
            "source_type": result.get("source_type"),
            "data_family": result.get("data_family"),
            "confidence": result.get("availability_check", {}).get("available", True) if isinstance(result.get("availability_check"), dict) else None,
            "similarity": None,
            "debug": {
                "availability_check": result.get("availability_check"),
                "continuation_used": result.get("continuation_used"),
                "pending_intent_used": result.get("pending_intent_used"),
                "resolved_bug_id": result.get("resolved_bug_id"),
                "validation_reason": result.get("validation_reason"),
                "qwen_finalization": result.get("debug", {}).get("qwen_finalization") if isinstance(result.get("debug"), dict) else None,
            },
        }


@dataclass
class ProductRuntime:
    model_path: Path
    data_paths: Sequence[Path]
    device: torch.device
    qwen: QwenBundle
    cache_dir: Path = BACKEND_CACHE_DIR
    lstm_model: Any = field(init=False)
    lstm_tokenizer: Any = field(init=False)
    lstm_config: Dict[str, object] = field(init=False)
    lookup_entries: List[Any] = field(init=False)
    lookup_index: Dict[str, List[int]] = field(init=False)
    availability_index: Dict[str, object] = field(init=False)
    bug_metadata_index: Dict[str, List[Dict[str, str]]] = field(init=False)

    def __post_init__(self) -> None:
        records: List[Dict[str, object]] = []
        for path in self.data_paths:
            if path.exists():
                records.extend(read_jsonl(path))
        self.lookup_entries = build_lookup_entries(records)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lookup_cache = self.cache_dir / "product_lookup_index.json"
        metadata_cache = self.cache_dir / "product_metadata_index.json"
        availability_cache = self.cache_dir / "product_availability_index.json"
        self.lookup_index = _build_product_lookup_index(self.lookup_entries)
        with lookup_cache.open("w", encoding="utf-8") as handle:
            json.dump(self.lookup_index, handle, indent=2, ensure_ascii=False)
        self.bug_metadata_index = _build_product_bug_metadata_index(self.lookup_entries)
        with metadata_cache.open("w", encoding="utf-8") as handle:
            json.dump(self.bug_metadata_index, handle, indent=2, ensure_ascii=False)
        self.availability_index = _build_product_availability_index(self.lookup_entries)
        with availability_cache.open("w", encoding="utf-8") as handle:
            json.dump(self.availability_index, handle, indent=2, ensure_ascii=False)
        self.lstm_model, self.lstm_tokenizer, self.lstm_config = load_lstm_support(self.model_path, self.device)

    def _availability_check(self, slots: Dict[str, str]) -> Dict[str, object]:
        release_notes = self.availability_index.get("product_docs", {})
        if not isinstance(release_notes, dict):
            release_notes = {}
        switch = _canonical_product_switch(slots.get("switch", ""))
        version = _clean(slots.get("version", "")).replace("_", ".")
        sub_version = _clean(slots.get("sub_version", ""))
        bug_id = _clean(slots.get("bug_id", ""))
        known_switches = [entry.switch for entry in self.lookup_entries if _clean(entry.switch)]
        switch_variants = _product_switch_aliases(switch, known_switches) if switch else []
        matching_switch = next((candidate for candidate in switch_variants if candidate in release_notes), "")

        if switch and not matching_switch:
            return {"available": False, "status": "data_not_available", "reason": f"switch {switch} not in product availability"}
        if switch and version:
            payload = release_notes.get(matching_switch, {}) if matching_switch else {}
            versions = payload.get("versions", {}) if isinstance(payload, dict) else {}
            version_aliases = _product_version_aliases(version, sub_version)
            matched_version = next((candidate for candidate in version_aliases if candidate in versions), "")
            if not matched_version:
                return {"available": False, "status": "data_not_available", "reason": f"version {version} not available for switch {matching_switch or switch}"}
            if sub_version:
                available_sub_versions = versions.get(matched_version, [])
                if available_sub_versions and sub_version not in available_sub_versions:
                    return {
                        "available": False,
                        "status": "data_not_available",
                        "reason": f"sub-version {sub_version} not available for switch {matching_switch or switch} version {version}",
                    }
        if bug_id and bug_id not in self.bug_metadata_index:
            return {"available": False, "status": "data_not_available", "reason": f"bug {bug_id} not found in product metadata"}
        return {"available": True, "status": "available", "reason": None}

    def answer(
        self,
        question: str,
        session_context: Dict[str, Optional[str]],
        selected_context: Dict[str, str],
        show_debug: bool = False,
        override_intent: Optional[str] = None,
    ) -> Dict[str, object]:
        cleaned_question = _clean(question)
        raw_lstm_intent = predict_intent(cleaned_question, self.lstm_model, self.lstm_tokenizer, self.lstm_config, self.device)
        override_intent = _clean(override_intent)
        if override_intent:
            raw_lstm_intent = override_intent
        llm_analysis = _llm_understand_question("product", cleaned_question, session_context, selected_context, self.qwen, self.device)
        llm_slots = llm_analysis.get("slots", {}) if isinstance(llm_analysis.get("slots"), dict) else {}
        slots = _merge_context_slots(llm_slots, session_context, selected_context, use_session_context=True)
        slots["switch"] = _canonical_product_switch(slots.get("switch", ""))
        predicted_intent = _clean(llm_analysis.get("intent", "")) or raw_lstm_intent
        question_type_hint = _clean(llm_analysis.get("question_type", "")) or _clean(predicted_intent)
        slots["question_type"] = question_type_hint
        slots["topic_family"] = _clean(llm_analysis.get("topic_family", ""))
        is_syntax_question = question_type_hint in {"cli_syntax", "show_command_syntax"}
        is_command_purpose = bool(llm_analysis.get("is_command_purpose", False))
        lookup_question = _clean(llm_analysis.get("lookup_query", "")) or cleaned_question
        is_followup = False
        used_previous_context = False
        followup_context: Dict[str, str] = {}
        if _clean(llm_analysis.get("clarification_question", "")) and bool(llm_analysis.get("needs_clarification", False)):
            clarification = _clean(llm_analysis.get("clarification_question", "")) or PRODUCT_SLOT_MISSING_RESPONSE
            return _finalize_answer_payload(
                {
                    "domain": "product",
                    "question": cleaned_question,
                    "predicted_intent": predicted_intent,
                    "raw_lstm_intent": raw_lstm_intent,
                    "slots": slots,
                    "lookup_status": "needs_disambiguation",
                    "lookup_key_used": None,
                    "lookup_answer": None,
                    "qwen_used": False,
                    "qwen_answer": None,
                    "qwen_validation_passed": False,
                    "final_answer": clarification,
                    "answer_source": "llm_clarification",
                    "source_type": predicted_intent,
                    "data_family": "product_documentation",
                    "confidence": 0.0,
                    "similarity": 0.0,
                    "debug": {
                        "llm_analysis": llm_analysis,
                        "availability_check": {"available": True, "status": "available", "reason": None},
                    },
                },
                question=cleaned_question,
                intent=predicted_intent,
                slots=slots,
                lookup_status="needs_disambiguation",
                target_value=None,
                related_context=None,
                rejection_reason=clarification,
                contamination_detected=False,
                qwen_bundle=self.qwen,
                device=self.device,
                data_family="product_documentation",
                source_type="llm_clarification",
                question_type=question_type_hint,
            )
        question_profile = {
            "question_type": question_type_hint or "generic_product_query",
            "topic_family": _clean(llm_analysis.get("topic_family", "")),
            "candidate_topics": _string_list(llm_analysis.get("candidate_topics"))[:8],
            "candidate_intents": _string_list(llm_analysis.get("candidate_intents"))[:8] or [predicted_intent] if predicted_intent else [],
            "required_slots": [],
            "route_type_variants": [],
            "normalized_question": lookup_question,
            "query_keywords": _string_list(llm_analysis.get("query_keywords"))[:16],
            "slots": slots,
        }
        profile_question_type = _clean(question_profile.get("question_type", ""))
        profile_topic_family = _clean(question_profile.get("topic_family", ""))
        strict_question = False

        availability_check = self._availability_check(slots)
        if not availability_check.get("available", True):
            if strict_question:
                return _finalize_answer_payload(
                    {
                    "domain": "product",
                    "question": cleaned_question,
                    "predicted_intent": predicted_intent,
                    "raw_lstm_intent": raw_lstm_intent,
                    "slots": slots,
                    "lookup_status": "data_not_available",
                    "lookup_key_used": None,
                    "lookup_answer": None,
                    "qwen_used": False,
                    "qwen_answer": None,
                    "qwen_validation_passed": False,
                    "final_answer": PRODUCT_DATANOT_AVAILABLE_RESPONSE,
                    "answer_source": "deterministic_availability",
                    "source_type": predicted_intent,
                    "data_family": "product_documentation",
                    "confidence": 0.0,
                    "similarity": 0.0,
                    "debug": {
                        "availability_check": availability_check,
                        "question_profile": question_profile,
                    },
                    },
                    question=cleaned_question,
                    intent=predicted_intent,
                    slots=slots,
                    lookup_status="data_not_available",
                    target_value=None,
                    related_context=None,
                    rejection_reason=availability_check.get("reason"),
                    contamination_detected=False,
                    qwen_bundle=self.qwen,
                    device=self.device,
                    data_family="product_documentation",
                    source_type="deterministic_availability",
                    question_type=profile_question_type,
                )
            common_slots = dict(slots)
            common_slots.pop("switch", None)
            common_slots.pop("version", None)
            common_slots.pop("sub_version", None)
            common_resolution = _resolve_generic_lookup(
                "product",
                lookup_question,
                predicted_intent,
                common_slots,
                self.lookup_entries,
                self.lookup_index,
            )
            if common_resolution.get("status") == "found" and common_resolution.get("answer"):
                availability_check = {"available": True, "status": "available", "reason": "common product lookup fallback"}
                slots = common_slots
        if not availability_check.get("available", True):
            return _finalize_answer_payload(
                {
                "domain": "product",
                "question": cleaned_question,
                "predicted_intent": "data_not_available",
                "raw_lstm_intent": "data_not_available",
                "slots": slots,
                "lookup_status": "data_not_available",
                "lookup_key_used": None,
                "lookup_answer": None,
                "qwen_answer": None,
                "qwen_validation_passed": False,
                "final_answer": PRODUCT_DATANOT_AVAILABLE_RESPONSE,
                "answer_source": "deterministic_availability",
                "confidence": 0.0,
                "similarity": 0.0,
                "debug": {
                    "availability_check": availability_check,
                    "question_profile": question_profile,
                },
                },
                question=cleaned_question,
                intent="data_not_available",
                slots=slots,
                lookup_status="data_not_available",
                target_value=None,
                related_context=None,
                rejection_reason=availability_check.get("reason"),
                contamination_detected=False,
                qwen_bundle=self.qwen,
                device=self.device,
                data_family="product_documentation",
                source_type="deterministic_availability",
                question_type=profile_question_type,
            )

        if False and _is_product_followup(cleaned_question) and not slots.get("command") and not slots.get("topic"):
            if _clean(session_context.get("last_command")):
                slots["command"] = _clean(session_context.get("last_command"))
            if _clean(session_context.get("last_topic")):
                slots["topic"] = _clean(session_context.get("last_topic"))
            if _clean(session_context.get("last_feature")) and not slots.get("feature"):
                slots["feature"] = _clean(session_context.get("last_feature"))
            if _clean(session_context.get("last_category")) and not slots.get("category"):
                slots["category"] = _clean(session_context.get("last_category"))

        resolution_bundle = _product_resolve_with_profile(cleaned_question, question_profile, slots, self.lookup_entries, self.lookup_index)
        resolution = dict(resolution_bundle.get("resolution", {}))
        lookup_status = str(resolution.get("status", "error"))
        lookup_answer = _clean(resolution.get("answer", "")) if resolution.get("answer") else ""
        lookup_key_used = resolution.get("lookup_key_used")
        confidence = float(resolution.get("confidence", 0.0) or 0.0)
        similarity = float(resolution.get("similarity", 0.0) or 0.0)
        lookup_stage = resolution.get("lookup_stage")
        grounded = _product_resolution_is_grounded(question_profile, resolution)
        if lookup_status == "found" and profile_question_type == "capacity_or_scale":
            topic_source = " ".join(
                [
                    _clean(question_profile.get("normalized_question", "")),
                    _clean(slots.get("topic", "")),
                    _clean(slots.get("feature", "")),
                    _clean(slots.get("category", "")),
                ]
            ).lower()
            if "vsf" in topic_source and not re.search(r"\bvsf\b|\bmember\b|\brange\b|\bstack\b", lookup_answer, flags=re.IGNORECASE):
                grounded = False
                lookup_status = "low_similarity"
            elif "vsx" in topic_source and not re.search(r"\bvsx\b|\bredundanc|failover|standby|active\b", lookup_answer, flags=re.IGNORECASE):
                grounded = False
                lookup_status = "low_similarity"
            elif "route" in topic_source and not re.search(r"\broute\b|\bipv4\b|\bipv6\b|\bmaximum\b|\bcapacity\b|\bscale\b", lookup_answer, flags=re.IGNORECASE):
                grounded = False
                lookup_status = "low_similarity"
        strict_fallback_message = ""
        if lookup_status == "found" and lookup_answer and _product_answer_looks_contaminated(lookup_answer):
            if profile_question_type == "cli_syntax":
                strict_fallback_message = PRODUCT_SYNTAX_MATCH_RESPONSE
            elif profile_question_type == "cli_output":
                strict_fallback_message = PRODUCT_COMMAND_OUTPUT_RESPONSE
        elif strict_question and not grounded:
            if profile_question_type == "cli_syntax":
                strict_fallback_message = PRODUCT_SYNTAX_MATCH_RESPONSE
            elif profile_question_type == "cli_output":
                strict_fallback_message = PRODUCT_COMMAND_OUTPUT_RESPONSE
        if strict_fallback_message:
            return _finalize_answer_payload(
                {
                "domain": "product",
                "question": cleaned_question,
                "predicted_intent": predicted_intent,
                "raw_lstm_intent": raw_lstm_intent,
                "slots": slots,
                "lookup_status": "not_found",
                "lookup_key_used": lookup_key_used,
                "lookup_answer": lookup_answer or None,
                "qwen_used": False,
                "qwen_answer": None,
                "qwen_validation_passed": False,
                "final_answer": strict_fallback_message,
                "answer_source": "lookup_fallback",
                "source_type": predicted_intent,
                "data_family": "product_documentation",
                "confidence": confidence,
                "similarity": similarity,
                "debug": {
                    "availability_check": availability_check,
                    "question_profile": question_profile,
                    "lookup_stage": lookup_stage,
                    "lookup_resolution": resolution,
                    "strict_fallback_message": strict_fallback_message,
                    "grounded": grounded,
                },
                },
                question=cleaned_question,
                intent=predicted_intent,
                slots=slots,
                lookup_status="not_found",
                target_value=lookup_answer or None,
                related_context=None,
                rejection_reason=strict_fallback_message,
                contamination_detected=bool(lookup_answer and _product_answer_looks_contaminated(lookup_answer)),
                qwen_bundle=self.qwen,
                device=self.device,
                data_family="product_documentation",
                source_type="lookup_fallback",
                question_type=profile_question_type,
            )
        if strict_question and not grounded:
            if lookup_status == "found":
                lookup_status = "needs_disambiguation" if lookup_key_used else "low_similarity"
            if lookup_status == "found":
                lookup_status = "low_similarity"
        if (
            not _product_resolution_is_grounded(question_profile, resolution)
            and strict_question
            and profile_question_type == "capacity_or_scale"
            and not _clean(slots.get("version", ""))
            and lookup_status != "found"
        ):
            clarification = "Please specify the AOS-CX version so I can check the supported route scale for that switch."
            return _finalize_answer_payload(
                {
                "domain": "product",
                "question": cleaned_question,
                "predicted_intent": predicted_intent,
                "raw_lstm_intent": raw_lstm_intent,
                "slots": slots,
                "lookup_status": "needs_disambiguation",
                "lookup_key_used": lookup_key_used,
                "lookup_answer": lookup_answer or None,
                "qwen_used": False,
                "qwen_answer": None,
                "qwen_validation_passed": False,
                "final_answer": clarification,
                "answer_source": "clarification",
                "source_type": predicted_intent,
                "data_family": "product_documentation",
                "confidence": confidence,
                "similarity": similarity,
                "debug": {
                    "availability_check": availability_check,
                    "question_profile": question_profile,
                    "lookup_stage": lookup_stage,
                    "lookup_resolution": resolution,
                },
                },
                question=cleaned_question,
                intent=predicted_intent,
                slots=slots,
                lookup_status="needs_disambiguation",
                target_value=None,
                related_context=None,
                rejection_reason=clarification,
                contamination_detected=False,
                qwen_bundle=self.qwen,
                device=self.device,
                data_family="product_documentation",
                source_type="clarification",
                question_type=profile_question_type,
            )
        if (
            not _product_resolution_is_grounded(question_profile, resolution)
            and strict_question
            and profile_question_type == "capacity_or_scale"
            and _clean(slots.get("switch", ""))
            and _clean(slots.get("version", ""))
            and not _clean(slots.get("route_type", ""))
            and lookup_status != "found"
        ):
            clarification = "Please specify whether you mean IPv4 or IPv6 route scale for that switch."
            return _finalize_answer_payload(
                {
                "domain": "product",
                "question": cleaned_question,
                "predicted_intent": predicted_intent,
                "raw_lstm_intent": raw_lstm_intent,
                "slots": slots,
                "lookup_status": "needs_disambiguation",
                "lookup_key_used": lookup_key_used,
                "lookup_answer": lookup_answer or None,
                "qwen_used": False,
                "qwen_answer": None,
                "qwen_validation_passed": False,
                "final_answer": clarification,
                "answer_source": "clarification",
                "source_type": predicted_intent,
                "data_family": "product_documentation",
                "confidence": confidence,
                "similarity": similarity,
                "debug": {
                    "availability_check": availability_check,
                    "question_profile": question_profile,
                    "lookup_stage": lookup_stage,
                    "lookup_resolution": resolution,
                },
                },
                question=cleaned_question,
                intent=predicted_intent,
                slots=slots,
                lookup_status="needs_disambiguation",
                target_value=None,
                related_context=None,
                rejection_reason=clarification,
                contamination_detected=False,
                qwen_bundle=self.qwen,
                device=self.device,
                data_family="product_documentation",
                source_type="clarification",
                question_type=profile_question_type,
            )

        previous_context_answer = ""
        if used_previous_context:
            previous_context_answer = _clean(session_context.get("last_lookup_answer")) or _clean(session_context.get("last_final_answer"))
            if previous_context_answer and lookup_status != "found" and not strict_question:
                lookup_status = "found"
                lookup_answer = previous_context_answer
                lookup_key_used = "session_context"
                confidence = max(confidence, 0.9)
                similarity = max(similarity, 0.9)

        qwen_answer = None
        qwen_validation_passed = False
        qwen_used = False
        answer_source = "lookup_fallback"
        final_answer = _format_deterministic("product", lookup_status)
        source_type = predicted_intent
        data_family = "product_documentation"

        if lookup_status == "found" and lookup_answer:
            formatter_lookup_answer = lookup_answer
            if used_previous_context and previous_context_answer:
                formatter_lookup_answer = previous_context_answer

            if is_syntax_question or is_command_purpose or _product_answer_looks_like_cli_syntax(formatter_lookup_answer):
                final_answer = format_cli_syntax_answer(
                    cleaned_question,
                    lookup_answer,
                    {"intent": predicted_intent, **slots},
                )
                answer_source = "deterministic_command_formatter" if is_command_purpose and not is_syntax_question else "deterministic_cli_syntax"
            else:
                if profile_question_type in {"support_matrix", "version_support", "capacity_or_scale"}:
                    final_answer = _format_product_concept_answer(lookup_answer, {**slots, "question_type": profile_question_type})
                    if not final_answer:
                        final_answer = _polish_product_answer(lookup_answer, predicted_intent, slots)
                else:
                    final_answer = _polish_product_answer(lookup_answer, predicted_intent, slots)
                answer_source = "deterministic_lookup"

            use_qwen_for_product = self.qwen.loaded and _should_use_qwen("product", predicted_intent, lookup_answer) and not QWEN_FINALIZE_ALL_RESPONSES
            if is_syntax_question or is_command_purpose or profile_question_type == "cli_output":
                use_qwen_for_product = False
            if used_previous_context and len(formatter_lookup_answer.split()) < 15:
                use_qwen_for_product = False

            if use_qwen_for_product:
                prompt = _build_qwen_prompt(
                    "product",
                    cleaned_question,
                    predicted_intent,
                    slots,
                    formatter_lookup_answer,
                    followup_context if used_previous_context else None,
                )
                try:
                    qwen_used = True
                    qwen_answer = generate_qwen_answer(
                        self.qwen.tokenizer,
                        self.qwen.model,
                        prompt,
                        predicted_intent,
                        self.device,
                        data_family="product_documentation",
                    )
                    qwen_validation_passed, _reason = validate_qwen_answer(
                        predicted_intent,
                        slots,
                        formatter_lookup_answer,
                        qwen_answer,
                        data_family="product_documentation",
                    )
                    if qwen_validation_passed and not _product_qwen_is_too_drifty(formatter_lookup_answer, qwen_answer):
                        final_answer = qwen_answer
                        answer_source = "qwen_grounded"
                    else:
                        final_answer = _polish_product_answer(formatter_lookup_answer, predicted_intent, slots)
                        answer_source = "lookup_fallback"
                except Exception:
                    qwen_answer = None
                    qwen_validation_passed = False
        elif used_previous_context and previous_context_answer:
            final_answer = _polish_product_answer(previous_context_answer, predicted_intent, slots)
            answer_source = "session_context"
            lookup_status = "found"
            lookup_answer = previous_context_answer
            lookup_key_used = "session_context"

        final_answer = _cleanup_product_markdown(_strip_product_generated_label(final_answer))

        finalization_fallback = final_answer
        if not (lookup_status == "found" and grounded):
            finalization_fallback = _qwen_finalization_safe_fallback(
                "product",
                lookup_status,
                question_type=profile_question_type,
                contamination_detected=bool(lookup_answer and _product_answer_looks_contaminated(lookup_answer)),
            )

        finalization = finalize_answer_with_qwen(
            cleaned_question,
            predicted_intent,
            slots,
            lookup_status,
            target_value=lookup_answer if lookup_status == "found" and lookup_answer else None,
            related_context=lookup_answer if lookup_status in {"low_similarity", "needs_disambiguation"} and lookup_answer else None,
            rejection_reason=str(strict_fallback_message or resolution.get("reason") or lookup_stage or lookup_status),
            contamination_detected=bool(lookup_answer and _product_answer_looks_contaminated(lookup_answer)),
            qwen_bundle=self.qwen,
            device=self.device,
            data_family="product_documentation",
            source_type=source_type,
            question_type=profile_question_type,
            fallback_answer=finalization_fallback,
        )
        final_answer = str(finalization["final_answer"])
        qwen_used = bool(finalization["qwen_used"])
        qwen_answer = finalization["qwen_answer"] if finalization["qwen_used"] else qwen_answer
        qwen_validation_passed = bool(finalization["qwen_validation_passed"])
        answer_source = str(finalization.get("answer_source", answer_source))

        if lookup_status == "found" and lookup_answer:
            formatter_lookup_answer = lookup_answer
            if used_previous_context and previous_context_answer:
                formatter_lookup_answer = previous_context_answer
            _update_session_context(
                session_context,
                cleaned_question,
                slots,
                predicted_intent,
                formatter_lookup_answer,
                final_answer,
                source_type,
                data_family,
            )

        if show_debug:
            print(f"[FORMATTER] question: {cleaned_question}")
            print(f"[FORMATTER] normalized_question: {lookup_question}")
            print(f"[FORMATTER] predicted_intent: {predicted_intent}")
            print(f"[FORMATTER] question_type: {profile_question_type}")
            print(f"[FORMATTER] topic_family: {_clean(question_profile.get('topic_family', ''))}")
            print(f"[FORMATTER] candidate_topics: {question_profile.get('candidate_topics', [])}")
            print(f"[FORMATTER] candidate_intents: {question_profile.get('candidate_intents', [])}")
            print(f"[FORMATTER] normalized_query: {lookup_question}")
            print(f"[FORMATTER] lookup_status: {lookup_status}")
            print(f"[FORMATTER] lookup_stage: {lookup_stage}")
            print(f"[FORMATTER] lookup_path: {resolution.get('lookup_path', '')}")
            print(f"[FORMATTER] lookup_answer_length: {len(lookup_answer or '')}")
            print(f"[FORMATTER] final_answer_length: {len(final_answer or '')}")
            print(f"[FORMATTER] is_cli_syntax: {is_syntax_question}")
            print(f"[FORMATTER] is_command_purpose: {is_command_purpose}")
            print(f"[FORMATTER] qwen_used: {qwen_used}")
            print(f"[FORMATTER] validation_passed: {qwen_validation_passed}")
            print(f"[FOLLOWUP] is_followup: {is_followup}")
            print(f"[FOLLOWUP] used_previous_context: {used_previous_context}")

        return {
            "domain": "product",
            "question": cleaned_question,
            "predicted_intent": predicted_intent,
            "raw_lstm_intent": raw_lstm_intent,
            "slots": slots,
            "lookup_status": lookup_status,
            "lookup_key_used": lookup_key_used,
            "lookup_stage": lookup_stage,
            "lookup_answer": lookup_answer or None,
            "qwen_used": qwen_used,
            "qwen_answer": qwen_answer,
            "qwen_validation_passed": qwen_validation_passed,
            "final_answer": final_answer,
            "answer_source": answer_source,
            "source_type": source_type,
            "data_family": data_family,
            "confidence": confidence,
            "similarity": similarity,
            "debug": {
                "availability_check": availability_check,
                "is_followup": is_followup,
                "used_previous_context": used_previous_context,
                "question_profile": question_profile,
                "lookup_resolution": resolution,
                "normalized_question": lookup_question,
                "qwen_finalization": {
                    "enabled": QWEN_FINALIZE_ALL_RESPONSES,
                    "lookup_status": lookup_status,
                    "used": qwen_used,
                    "validation_passed": qwen_validation_passed,
                },
            },
        }


@dataclass
class AnswerService:
    device: torch.device
    release: ReleaseRuntime
    product: ProductRuntime
    unified_lstm: LstmBundle
    qwen: QwenBundle
    sessions: Dict[str, Dict[str, Dict[str, Optional[str]]]] = field(default_factory=dict)

    @classmethod
    def create(cls, device: Optional[torch.device] = None) -> "AnswerService":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qwen_bundle = QwenBundle()
        unified_bundle = LstmBundle()
        try:
            tokenizer, model, meta = load_qwen_model(QWEN_MODEL_PATH, device)
            qwen_bundle = QwenBundle(
                tokenizer=tokenizer,
                model=model,
                requested_path=meta.get("requested_path", str(QWEN_MODEL_PATH)),
                resolved_path=meta.get("resolved_path", ""),
                model_kind=meta.get("model_kind", "adapter"),
                resolution_reason=meta.get("resolution_reason", ""),
                base_model_name=meta.get("base_model_name", ""),
                loaded=True,
            )
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            qwen_bundle = QwenBundle(
                requested_path=str(QWEN_MODEL_PATH),
                loaded=False,
                error=str(exc),
            )
        try:
            unified_model, unified_tokenizer, unified_config = load_lstm_support(UNIFIED_LSTM_MODEL_PATH, device)
            unified_bundle = LstmBundle(
                model=unified_model,
                tokenizer=unified_tokenizer,
                config=unified_config,
                requested_path=str(UNIFIED_LSTM_MODEL_PATH),
                resolved_path=str(UNIFIED_LSTM_MODEL_PATH),
                loaded=True,
            )
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            unified_bundle = LstmBundle(
                requested_path=str(UNIFIED_LSTM_MODEL_PATH),
                loaded=False,
                error=str(exc),
            )

        release = ReleaseRuntime(
            model_path=RELEASE_LSTM_MODEL_PATH,
            lookup_data_path=RELEASE_LOOKUP_DATA_PATH,
            lookup_index_path=RELEASE_LOOKUP_INDEX_PATH,
            availability_path=RELEASE_AVAILABILITY_PATH,
            bug_metadata_path=RELEASE_BUG_METADATA_PATH,
            device=device,
            qwen=qwen_bundle,
        )
        product = ProductRuntime(
            model_path=PRODUCT_LSTM_MODEL_PATH,
            data_paths=PRODUCT_LOOKUP_DATA_PATHS,
            device=device,
            qwen=qwen_bundle,
        )
        return cls(device=device, release=release, product=product, unified_lstm=unified_bundle, qwen=qwen_bundle)

    def new_session_id(self) -> str:
        return uuid4().hex

    def _session(self, session_id: str) -> Dict[str, Dict[str, Optional[str]]]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "release": _session_template(),
                "product": _session_template(),
                "unified": _session_template(),
            }
        else:
            self.sessions[session_id].setdefault("release", _session_template())
            self.sessions[session_id].setdefault("product", _session_template())
            self.sessions[session_id].setdefault("unified", _session_template())
        return self.sessions[session_id]

    def resolve_domain(self, requested_domain: str, question: str, session: Optional[Dict[str, Dict[str, Optional[str]]]] = None) -> str:
        domain = _clean(requested_domain).lower()
        if domain in {"release", "product", "unified"}:
            return domain
        session = session or {}
        if self.qwen.loaded:
            llm_analysis = _llm_understand_question(
                "auto",
                question,
                session.get("product", {}),
                {},
                self.qwen,
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            llm_domain = _clean(llm_analysis.get("domain", "")).lower()
            if llm_domain in {"release", "product", "unified"}:
                return llm_domain
        # Fallback for when the LLM is unavailable or does not produce a usable classification.
        text = _clean(question).lower()
        if any(keyword in text for keyword in ["bug ", "bug id", "workaround", "scenario", "symptom", "release note", "caveat"]):
            return "release"
        return "product"

    def chat(
        self,
        question: str,
        session_id: Optional[str] = None,
        domain: str = "auto",
        selected_switch: str = "",
        selected_version: str = "",
        selected_sub_version: str = "",
        show_debug: bool = False,
    ) -> Dict[str, object]:
        session_id = session_id or self.new_session_id()
        session = self._session(session_id)
        resolved_domain = self.resolve_domain(domain, question, session)
        selected_context = {
            "switch": selected_switch,
            "version": selected_version,
            "sub_version": selected_sub_version,
        }
        if resolved_domain == "unified":
            unified_intent = ""
            if self.unified_lstm.loaded and self.unified_lstm.model is not None and self.unified_lstm.tokenizer is not None and self.unified_lstm.config:
                try:
                    unified_intent = predict_intent(question, self.unified_lstm.model, self.unified_lstm.tokenizer, self.unified_lstm.config, self.device)
                except Exception:
                    unified_intent = ""
            route_domain = "release" if unified_intent in RELEASE_LIKE_INTENTS else self.resolve_domain("auto", question, session)
            runtime = self.release if route_domain == "release" else self.product
            session_context = session["unified"]
            result = runtime.answer(
                question,
                session_context,
                selected_context,
                show_debug=show_debug,
                override_intent=unified_intent or None,
            )
            result["domain"] = "unified"
            debug = result.get("debug")
            if isinstance(debug, dict):
                debug["unified_route_domain"] = route_domain
                debug["unified_intent"] = unified_intent
                debug["unified_model_loaded"] = self.unified_lstm.loaded
                if self.unified_lstm.error:
                    debug["unified_model_error"] = self.unified_lstm.error
        else:
            runtime = self.release if resolved_domain == "release" else self.product
            session_context = session[resolved_domain]
            result = runtime.answer(question, session_context, selected_context, show_debug=show_debug)

        return {
            "session_id": session_id,
            "domain": resolved_domain,
            "question": result.get("question"),
            "predicted_intent": result.get("predicted_intent"),
            "raw_lstm_intent": result.get("raw_lstm_intent"),
            "slots": result.get("slots", {}),
            "lookup_status": result.get("lookup_status"),
            "lookup_key_used": result.get("lookup_key_used"),
            "lookup_answer": result.get("lookup_answer"),
            "qwen_used": result.get("qwen_used"),
            "qwen_answer": result.get("qwen_answer"),
            "qwen_validation_passed": result.get("qwen_validation_passed"),
            "final_answer": result.get("final_answer"),
            "answer_source": result.get("answer_source"),
            "source_type": result.get("source_type"),
            "data_family": result.get("data_family"),
            "confidence": result.get("confidence"),
            "similarity": result.get("similarity"),
            "debug": result.get("debug", {}),
            "qwen_loaded": self.qwen.loaded,
            "qwen_model_path": self.qwen.resolved_path or self.qwen.requested_path,
            "qwen_model_kind": self.qwen.model_kind,
            "qwen_error": self.qwen.error or None,
        }

    def health(self) -> Dict[str, object]:
        return {
            "device": str(self.device),
            "qwen_loaded": self.qwen.loaded,
            "qwen_model_path": self.qwen.resolved_path or self.qwen.requested_path,
            "qwen_model_kind": self.qwen.model_kind,
            "qwen_error": self.qwen.error or None,
            "qwen_finalize_all_responses": QWEN_FINALIZE_ALL_RESPONSES,
            "unified_lstm_loaded": self.unified_lstm.loaded,
            "unified_lstm_path": self.unified_lstm.resolved_path or self.unified_lstm.requested_path,
            "unified_lstm_error": self.unified_lstm.error or None,
            "release_lstm_path": str(RELEASE_LSTM_MODEL_PATH),
            "product_lstm_path": str(PRODUCT_LSTM_MODEL_PATH),
            "unified_lstm_model_path": str(UNIFIED_LSTM_MODEL_PATH),
            "model_root": str(MODEL_ROOT),
            "data_root": str(DATA_ROOT),
            "release_notes_data_dir": str(RELEASE_NOTES_DATA_DIR),
            "product_docs_data_dir": str(PRODUCT_DOCS_DATA_DIR),
            "release_lstm_data_dir": str(RELEASE_LSTM_DATA_DIR),
            "product_lstm_data_dir": str(PRODUCT_LSTM_DATA_DIR),
            "release_lookup_path": str(RELEASE_LOOKUP_INDEX_PATH),
            "release_bug_metadata_path": str(RELEASE_BUG_METADATA_PATH),
            "release_availability_path": str(RELEASE_AVAILABILITY_PATH),
            "product_lookup_data_paths": [str(path) for path in PRODUCT_LOOKUP_DATA_PATHS],
            "backend_cache_dir": str(BACKEND_CACHE_DIR),
            "ollama_base_url": OLLAMA_BASE_URL or None,
            "release_runtime": {
                "lookup_entries": len(self.release.lookup_entries),
                "lookup_keys": len(self.release.lookup_index),
                "availability_switches": len(self.release.availability_index.get("release_notes", {})),
            },
            "product_runtime": {
                "lookup_entries": len(self.product.lookup_entries),
                "lookup_keys": len(self.product.lookup_index),
                "availability_switches": len(self.product.availability_index.get("product_docs", {})),
            },
        }
