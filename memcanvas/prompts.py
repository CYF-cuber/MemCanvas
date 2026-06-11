"""Prompt templates used by MemCanvas evaluation scripts."""

SCIENCEQA_PROMPT = """Study the reference canvases above. Each shows a solved example.
{hint}

Question: {question}
{choices}
Think step by step, then answer with just the letter:"""

OKVQA_PROMPT = """Study the reference canvases. Answer the question about the last image.
Question: {question}
Answer concisely:"""

MMQA_PROMPT_WITH_MEMORY = """Below are memory canvases from similar questions. Study them.
---
{context}
Question: {question}
Answer concisely:"""

MMQA_PROMPT_NO_MEMORY = """{context}
Question: {question}
Answer concisely:"""

HOTPOTQA_MEMORY_INSTRUCTION = """Below are memory canvases from previously solved similar questions. Each canvas shows: relevant context passages, the question, and the correct answer (marked with ✓). Study these canvases carefully."""

HOTPOTQA_PROMPT = """{memory_instruction}

Now answer the following new question using the context below.

{context}

Question: {question}
Answer concisely:"""

COMPRESSION_PROMPT = """Compress the interaction into a concise memory record. Keep entities, numbers, relations, user intent, answer-relevant visual details, and cross-modal links. Remove filler and redundant wording."""


def format_choices(choices: list[str]) -> str:
    return "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices))
