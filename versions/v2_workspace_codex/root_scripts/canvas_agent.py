#!/usr/bin/env python3
"""
Canvas Manager Agent — lightweight LLM agent for intelligent canvas management.

The agent makes two types of decisions:
1. Write decisions: How to organize new knowledge into canvases
2. Read decisions: Which canvases to retrieve for a given query

The agent is a text-only LLM (Qwen3-4B) that sees canvas metadata (not images).
The VLM reader (Qwen2.5-VL-7B) stays frozen and only reads canvas images.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    model_path: str = "/home/cyf/Qwen3-4B"
    lora_path: Optional[str] = None  # Path to LoRA adapter (SFT/RL trained)
    device: str = "cuda"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    # For RL: whether to sample or greedy
    do_sample: bool = True


# ---------------------------------------------------------------------------
# Canvas metadata (what the agent sees — no images)
# ---------------------------------------------------------------------------
@dataclass
class CanvasMeta:
    """Lightweight metadata for a canvas — this is what the agent sees."""
    canvas_id: int
    topic: str  # short topic description
    num_items: int  # number of knowledge items stored
    num_patches: int  # number of 640x640 patches used
    max_patches: int  # maximum patches allowed
    created_at: float = 0.0  # timestamp
    last_accessed: float = 0.0
    access_count: int = 0
    text_summary: str = ""  # brief content summary

    @property
    def utilization(self) -> float:
        return self.num_patches / max(self.max_patches, 1)

    @property
    def is_full(self) -> bool:
        return self.num_patches >= self.max_patches

    def to_display(self) -> str:
        fill = f"{self.num_patches}/{self.max_patches}"
        return (
            f"[Canvas #{self.canvas_id}] "
            f"Topic: \"{self.topic}\" | "
            f"Patches: {fill} | "
            f"Items: {self.num_items} | "
            f"Accesses: {self.access_count}"
        )


# ---------------------------------------------------------------------------
# Agent action definitions
# ---------------------------------------------------------------------------
@dataclass
class WriteAction:
    action: str  # CREATE, APPEND, MERGE, COMPRESS, NOOP
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class ReadAction:
    action: str  # RETRIEVE, DIRECT_ANSWER
    canvas_ids: List[int] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
WRITE_SYSTEM_PROMPT = """\
You are a memory canvas manager. Your job is to organize knowledge into visual canvases efficiently.

Given the current canvas library and new knowledge, decide the best action:
- CREATE: Create a new canvas with a descriptive topic title (when knowledge is about a new topic)
- APPEND <canvas_id>: Add to an existing canvas (when knowledge fits an existing topic)
- NOOP: Skip storage (when knowledge is trivial, duplicate, or not worth storing)

Rules:
1. Each canvas should have a coherent topic
2. Prefer APPEND over CREATE to keep canvases consolidated
3. Use NOOP for trivial or duplicate information
4. Keep topic titles concise (3-8 words)

Respond with a single JSON object on one line:
{"action": "...", "canvas_id": <int or null>, "topic": "<for CREATE only>", "reason": "..."}"""

READ_SYSTEM_PROMPT = """\
You are a memory retrieval agent. Given a question and candidate canvases ranked by similarity, select which canvases to retrieve.

Rules:
1. Select 1-3 most relevant canvases
2. Consider both topic relevance and content summary
3. If the question can be answered without memory, use DIRECT_ANSWER

Respond with a single JSON object on one line:
{"action": "RETRIEVE", "canvas_ids": [<int>, ...], "reason": "..."}
or
{"action": "DIRECT_ANSWER", "reason": "..."}"""


def build_write_prompt(canvas_library: List[CanvasMeta], new_knowledge: str) -> str:
    """Build the write decision prompt."""
    if canvas_library:
        lib_str = "\n".join(c.to_display() for c in canvas_library)
    else:
        lib_str = "(empty — no canvases yet)"

    return (
        f"<canvas_library>\n{lib_str}\n</canvas_library>\n\n"
        f"<new_knowledge>\n{new_knowledge}\n</new_knowledge>\n\n"
        f"Decide how to store this knowledge."
    )


def build_read_prompt(
    question: str,
    candidates: List[Tuple[CanvasMeta, float]],  # (meta, similarity_score)
) -> str:
    """Build the read decision prompt."""
    cand_lines = []
    for meta, score in candidates:
        summary = meta.text_summary[:200] if meta.text_summary else "No summary"
        cand_lines.append(
            f"[Canvas #{meta.canvas_id}] Topic: \"{meta.topic}\" | "
            f"Similarity: {score:.2f}\n"
            f"  Summary: {summary}"
        )
    cand_str = "\n".join(cand_lines)

    return (
        f"<question>{question}</question>\n\n"
        f"<candidates>\n{cand_str}\n</candidates>\n\n"
        f"Which canvases should be retrieved to answer the question?"
    )


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------
def parse_json_response(text: str) -> Optional[Dict]:
    """Extract JSON from model response, handling common issues."""
    # Try to find JSON in the response
    # First try: direct parse
    text = text.strip()

    # Remove thinking tags if present (Qwen3 thinking mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try to find JSON object
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return None


# ---------------------------------------------------------------------------
# Canvas Manager Agent
# ---------------------------------------------------------------------------
class CanvasManagerAgent:
    """Lightweight LLM agent for canvas management decisions."""

    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        """Load the agent model, optionally with LoRA adapter."""
        if self._loaded:
            return

        print(f"Loading Canvas Manager Agent from {self.config.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.config.device,
            trust_remote_code=True,
        )

        # Apply LoRA adapter if specified
        if self.config.lora_path:
            print(f"  Loading LoRA adapter from {self.config.lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, self.config.lora_path)
            self.model = self.model.merge_and_unload()
            print(f"  LoRA merged successfully")

        self.model.eval()
        self._loaded = True
        print(f"  Agent loaded on {self.config.device}")

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from the agent."""
        if not self._loaded:
            self.load()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3: disable thinking for speed
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.do_sample else 1.0,
                top_p=self.config.top_p if self.config.do_sample else 1.0,
                do_sample=self.config.do_sample,
            )

        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()

    def write_decision(
        self,
        canvas_library: List[CanvasMeta],
        new_knowledge: str,
    ) -> WriteAction:
        """Decide how to store new knowledge."""
        user_prompt = build_write_prompt(canvas_library, new_knowledge)
        response = self._generate(WRITE_SYSTEM_PROMPT, user_prompt)

        parsed = parse_json_response(response)
        if parsed is None:
            # Fallback: CREATE if no canvases, else APPEND to first
            if not canvas_library:
                return WriteAction(action="CREATE", params={"topic": "General"}, reason="parse_failed")
            return WriteAction(action="APPEND", params={"canvas_id": 0}, reason="parse_failed")

        action = parsed.get("action", "NOOP").upper()

        if action == "CREATE":
            topic = parsed.get("topic", "Untitled")
            return WriteAction(action="CREATE", params={"topic": topic}, reason=parsed.get("reason", ""))
        elif action == "APPEND":
            cid = parsed.get("canvas_id", 0)
            return WriteAction(action="APPEND", params={"canvas_id": cid}, reason=parsed.get("reason", ""))
        elif action == "NOOP":
            return WriteAction(action="NOOP", reason=parsed.get("reason", ""))
        else:
            return WriteAction(action="NOOP", reason=f"unknown_action_{action}")

    def read_decision(
        self,
        question: str,
        candidates: List[Tuple[CanvasMeta, float]],
    ) -> ReadAction:
        """Decide which canvases to retrieve."""
        if not candidates:
            return ReadAction(action="DIRECT_ANSWER", reason="no_candidates")

        user_prompt = build_read_prompt(question, candidates)
        response = self._generate(READ_SYSTEM_PROMPT, user_prompt)

        parsed = parse_json_response(response)
        if parsed is None:
            # Fallback: return top candidate
            return ReadAction(
                action="RETRIEVE",
                canvas_ids=[candidates[0][0].canvas_id],
                reason="parse_failed",
            )

        action = parsed.get("action", "RETRIEVE").upper()

        if action == "DIRECT_ANSWER":
            return ReadAction(action="DIRECT_ANSWER", reason=parsed.get("reason", ""))
        else:
            ids = parsed.get("canvas_ids", [])
            # Validate IDs
            valid_ids = [c.canvas_id for c, _ in candidates]
            ids = [i for i in ids if i in valid_ids]
            if not ids:
                ids = [candidates[0][0].canvas_id]
            return ReadAction(action="RETRIEVE", canvas_ids=ids, reason=parsed.get("reason", ""))

    def unload(self):
        """Free GPU memory."""
        if self._loaded:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._loaded = False
            torch.cuda.empty_cache()
            print("Agent unloaded")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Canvas Manager Agent — Quick Test")
    print("=" * 60)

    agent = CanvasManagerAgent(AgentConfig(
        model_path="/home/cyf/Qwen3-4B",
        do_sample=False,  # greedy for deterministic test
    ))
    agent.load()

    # Test 1: Write decision — empty library
    print("\n--- Test 1: Write to empty library ---")
    action = agent.write_decision(
        canvas_library=[],
        new_knowledge="The Eiffel Tower was built in 1889 for the World's Fair in Paris. It stands 330 meters tall.",
    )
    print(f"  Action: {action.action}")
    print(f"  Params: {action.params}")
    print(f"  Reason: {action.reason}")

    # Test 2: Write decision — existing canvases
    print("\n--- Test 2: Write to existing library ---")
    library = [
        CanvasMeta(canvas_id=0, topic="Paris Landmarks", num_items=3, num_patches=1, max_patches=4),
        CanvasMeta(canvas_id=1, topic="French Cuisine", num_items=2, num_patches=1, max_patches=4),
    ]
    action = agent.write_decision(
        canvas_library=library,
        new_knowledge="The Louvre Museum houses the Mona Lisa and over 380,000 artworks.",
    )
    print(f"  Action: {action.action}")
    print(f"  Params: {action.params}")
    print(f"  Reason: {action.reason}")

    # Test 3: Read decision
    print("\n--- Test 3: Read decision ---")
    candidates = [
        (CanvasMeta(canvas_id=0, topic="Paris Landmarks", num_items=3, num_patches=1, max_patches=4,
                    text_summary="Eiffel Tower (1889, 330m), Arc de Triomphe, Notre-Dame Cathedral"),
         0.92),
        (CanvasMeta(canvas_id=1, topic="French Cuisine", num_items=2, num_patches=1, max_patches=4,
                    text_summary="Croissants, Coq au Vin, French wine regions"),
         0.35),
        (CanvasMeta(canvas_id=2, topic="World Skyscrapers", num_items=5, num_patches=2, max_patches=4,
                    text_summary="Burj Khalifa (828m), Shanghai Tower, Empire State Building"),
         0.71),
    ]
    read = agent.read_decision(
        question="How tall is the Eiffel Tower?",
        candidates=candidates,
    )
    print(f"  Action: {read.action}")
    print(f"  Canvas IDs: {read.canvas_ids}")
    print(f"  Reason: {read.reason}")

    # Test 4: NOOP case
    print("\n--- Test 4: Trivial knowledge ---")
    action = agent.write_decision(
        canvas_library=library,
        new_knowledge="Hello, how are you today?",
    )
    print(f"  Action: {action.action}")
    print(f"  Reason: {action.reason}")

    agent.unload()
    print("\nAll tests passed!")
