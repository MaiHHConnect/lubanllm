"""Local username masking; curated licensed lexicon, no external moderation call."""
import json
import re
import unicodedata
from pathlib import Path


def compact(text):
    return "".join(c.lower() for c in unicodedata.normalize("NFKC", text)
                   if c == "*" or not unicodedata.category(c).startswith(("P", "Z")))


WORDS = frozenset(compact(word) for word in json.loads(Path(__file__).with_name("name_words.json").read_text()))
LENGTHS = sorted({len(word) for word in WORDS}, reverse=True)


def mask_name(text):
    text = unicodedata.normalize("NFKC", text)
    key_chars, positions = [], []
    for i, c in enumerate(text):
        if c == "*" or not unicodedata.category(c).startswith(("P", "Z")):
            for letter in c.lower():
                key_chars.append(letter)
                positions.append(i)
    key = "".join(key_chars)
    spans, offset = [], 0
    while offset < len(key):
        length = next((n for n in LENGTHS if offset + n <= len(key) and key[offset:offset+n] in WORDS), 0)
        if length:
            spans.append((positions[offset], positions[offset + length - 1] + 1))
            offset += length
        else:
            offset += 1
    parts, start = [], 0
    for begin, end in spans:
        parts.extend((text[start:begin], "***"))
        start = end
    return re.sub(r"(?:\*\*\*){2,}", "***", "".join(parts) + text[start:])
