"""Classify UltraFeedback prompts into subjective domain categories."""

import re

# Keywords/patterns for domain classification
_CODING_PATTERNS = re.compile(
    r"\b(code|function|program|script|algorithm|debug|implement|class\s|"
    r"def\s|import\s|return\s|variable|compile|syntax|API|database|SQL|"
    r"python|javascript|java\b|html|css|regex)\b",
    re.IGNORECASE,
)

_MATH_PATTERNS = re.compile(
    r"\b(calculate|compute|solve|equation|formula|integral|derivative|"
    r"probability|proof|theorem|algebra|geometry|arithmetic|fraction|"
    r"percentage|ratio|matrix|vector)\b",
    re.IGNORECASE,
)

_QA_PATTERNS = re.compile(
    r"\b(who is|who was|what is|what are|when did|when was|where is|"
    r"where was|how many|how much|which country|which city|capital of|"
    r"define\b|explain what)\b",
    re.IGNORECASE,
)

_CREATIVE_PATTERNS = re.compile(
    r"\b(write a (story|poem|song|essay|letter|script|dialogue|narrative)|"
    r"creative writing|fiction|imagine|compose|draft a|storytelling|"
    r"fairy tale|short story|haiku|limerick|screenplay)\b",
    re.IGNORECASE,
)

_OPINION_PATTERNS = re.compile(
    r"\b(what do you think|opinion|debate|argue|pros and cons|"
    r"should I|is it better|compare.*vs|ethical|moral|controversial|"
    r"agree or disagree|perspective on)\b",
    re.IGNORECASE,
)

# UltraFeedback source mappings (known source tendencies)
_SOURCE_DOMAIN_MAP = {
    "evol_instruct": None,  # Mixed, needs keyword classification
    "sharegpt": None,
    "flan_v2_niv2": None,
    "flan_v2_cot": "math",
    "false_qa": "qa",
    "ultrachat": None,
}


def classify_domain(instruction: str, source: str = "") -> str:
    """Classify an instruction into a subjective domain.

    Priority order: coding > math > qa > creative > opinion > advice (default).

    Args:
        instruction: The prompt text.
        source: UltraFeedback source identifier.

    Returns:
        One of: 'coding', 'math', 'qa', 'creative', 'opinion', 'advice'.
    """
    # Check source-level hints first
    mapped = _SOURCE_DOMAIN_MAP.get(source)
    if mapped:
        return mapped

    # Keyword-based classification (priority order)
    if _CODING_PATTERNS.search(instruction):
        return "coding"
    if _MATH_PATTERNS.search(instruction):
        return "math"
    if _QA_PATTERNS.search(instruction):
        return "qa"
    if _CREATIVE_PATTERNS.search(instruction):
        return "creative"
    if _OPINION_PATTERNS.search(instruction):
        return "opinion"

    # Default to advice for remaining conversational prompts
    return "advice"
