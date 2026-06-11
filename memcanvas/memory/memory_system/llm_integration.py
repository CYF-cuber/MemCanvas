"""
LLM Integration - textLLMtext

textLLMtext。

text：
1. VLMtext：text（textQwen2-VL）
2. Embeddingtext：textvision tokenstextLLM embeddingtext
3. text：textVLMtext，textLLM
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Literal
from pathlib import Path
import numpy as np
from PIL import Image


@dataclass
class LLMIntegrationConfig:
    """LLMtext"""
    # text: vlm, embedding_inject, text_convert
    mode: str = "vlm"
    # VLMtext
    vlm_model: str = "Qwen/Qwen2-VL-7B-Instruct"
    # textLLMtext
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    # text
    device: str = "cuda"
    # maximum number of memories
    max_memories: int = 5
    # texttokentext
    max_tokens: int = 4096
    # text（textVLMtext）
    save_canvas: bool = True


@dataclass
class MemoryContext:
    """text，textLLM"""
    # memory IDtext
    memory_ids: List[str]
    # text（VLMtext）
    images: Optional[List[Image.Image]] = None
    # Vision tokenstext（embeddingtext）
    vision_tokens: Optional[List[np.ndarray]] = None
    # text（text）
    text_descriptions: Optional[List[str]] = None
    # text
    metadata: List[Dict[str, Any]] = field(default_factory=list)


class MemoryToLLM:
    """
    textLLMtext

    textLLMtext。

    text：
    ```python
    # text1: VLM（text，textQwen2-VL）
    bridge = MemoryToLLM(mode="vlm")
    context = bridge.prepare_context(retrieval_results)
    response = bridge.query_vlm("text", context)

    # text2: text（textLLMtext）
    bridge = MemoryToLLM(mode="text_convert")
    context = bridge.prepare_context(retrieval_results)
    response = bridge.query_llm("text", context)
    ```
    """

    def __init__(self, config: Optional[LLMIntegrationConfig] = None):
        self.config = config or LLMIntegrationConfig()
        self._vlm_model = None
        self._vlm_processor = None
        self._llm_model = None
        self._llm_tokenizer = None

    def prepare_context(
        self,
        retrieval_results: List,
        include_images: bool = True,
        include_tokens: bool = False,
        generate_descriptions: bool = False
    ) -> MemoryContext:
        """
        text

        Args:
            retrieval_results: text（RetrievalResult）
            include_images: text
            include_tokens: textvision tokens
            generate_descriptions: text

        Returns:
            MemoryContext
        """
        memory_ids = []
        images = [] if include_images else None
        vision_tokens = [] if include_tokens else None
        text_descriptions = [] if generate_descriptions else None
        metadata = []

        for result in retrieval_results[:self.config.max_memories]:
            memory = result.memory
            memory_ids.append(result.memory_id)

            # text
            meta_info = {
                "memory_id": result.memory_id,
                "score": result.score,
                "created_at": memory.meta.created_at.isoformat(),
                "modalities": memory.meta.modalities,
                "source": memory.meta.source
            }
            metadata.append(meta_info)

            # textvision tokens
            if include_tokens:
                vision_tokens.append(memory.tokens)

            # text，text
            # text：text
            if include_images:
                canvas_image = self._load_canvas_image(result.memory_id)
                if canvas_image:
                    images.append(canvas_image)

        # text
        if generate_descriptions:
            text_descriptions = self._generate_descriptions(images or [], metadata)

        return MemoryContext(
            memory_ids=memory_ids,
            images=images,
            vision_tokens=vision_tokens,
            text_descriptions=text_descriptions,
            metadata=metadata
        )

    def _load_canvas_image(self, memory_id: str) -> Optional[Image.Image]:
        """text（text）"""
        # text
        # textNone，text
        return None

    def _generate_descriptions(
        self,
        images: List[Image.Image],
        metadata: List[Dict]
    ) -> List[str]:
        """textVLMtext"""
        if not images:
            return [f"text {m['memory_id']}，text {m['created_at']}" for m in metadata]

        descriptions = []
        for img, meta in zip(images, metadata):
            desc = self._describe_image(img)
            descriptions.append(f"[text {meta['memory_id']}]\n{desc}")

        return descriptions

    def _describe_image(self, image: Image.Image) -> str:
        """textVLMtext"""
        self._init_vlm()

        if self._vlm_model is None:
            return "（text）"

        # textQwen2-VLtext
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "text，text、text、text。"}
                ]
            }
        ]

        try:
            text = self._vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._vlm_processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            ).to(self.config.device)

            output_ids = self._vlm_model.generate(**inputs, max_new_tokens=512)
            output_text = self._vlm_processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]

            return output_text
        except Exception as e:
            return f"（text: {e}）"

    # ==================== VLM text ====================

    def query_vlm(
        self,
        question: str,
        context: MemoryContext,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        textVLM（textQwen2-VL）text

        Args:
            question: text
            context: text
            system_prompt: text

        Returns:
            text
        """
        self._init_vlm()

        if self._vlm_model is None:
            return "VLMtext"

        # textmessage
        content = []

        # text
        if context.images:
            for i, img in enumerate(context.images):
                content.append({"type": "image", "image": img})
                content.append({
                    "type": "text",
                    "text": f"[text {i+1}，text: {context.metadata[i]['score']:.2f}]"
                })

        # text
        content.append({"type": "text", "text": f"\ntext，text：{question}"})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        try:
            text = self._vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._vlm_processor(
                text=[text],
                images=context.images,
                return_tensors="pt"
            ).to(self.config.device)

            output_ids = self._vlm_model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens
            )
            response = self._vlm_processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]

            return response
        except Exception as e:
            return f"text: {e}"

    # ==================== textLLMtext ====================

    def query_llm(
        self,
        question: str,
        context: MemoryContext,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        textLLMtext

        text。

        Args:
            question: text
            context: text（texttext_descriptions）
            system_prompt: text

        Returns:
            text
        """
        self._init_llm()

        if self._llm_model is None:
            return "LLMtext"

        # text
        memory_text = ""
        if context.text_descriptions:
            for i, desc in enumerate(context.text_descriptions):
                memory_text += f"\n--- text {i+1} ---\n{desc}\n"
        else:
            # text
            for i, meta in enumerate(context.metadata):
                memory_text += f"\n--- text {i+1} ---\n"
                memory_text += f"ID: {meta['memory_id']}\n"
                memory_text += f"text: {meta['created_at']}\n"
                memory_text += f"text: {meta['score']:.2f}\n"

        prompt = f"""text：
{memory_text}

text，text：{question}"""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            text = self._llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._llm_tokenizer(text, return_tensors="pt").to(self.config.device)

            output_ids = self._llm_model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens
            )
            response = self._llm_tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )

            return response
        except Exception as e:
            return f"text: {e}"

    # ==================== text ====================

    def _init_vlm(self):
        """textVLM"""
        if self._vlm_model is not None:
            return

        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch

            print(f"textVLM: {self.config.vlm_model}")
            self._vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.vlm_model,
                torch_dtype=torch.bfloat16,
                device_map=self.config.device
            )
            self._vlm_processor = AutoProcessor.from_pretrained(self.config.vlm_model)

        except Exception as e:
            print(f"VLMtext: {e}")
            self._vlm_model = None

    def _init_llm(self):
        """textLLM"""
        if self._llm_model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"textLLM: {self.config.llm_model}")
            self._llm_model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model,
                torch_dtype=torch.bfloat16,
                device_map=self.config.device
            )
            self._llm_tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)

        except Exception as e:
            print(f"LLMtext: {e}")
            self._llm_model = None


# ==================== text ====================

def create_memory_bridge(
    mode: Literal["vlm", "text_convert"] = "vlm",
    model_name: Optional[str] = None,
    device: str = "cuda"
) -> MemoryToLLM:
    """
    text

    Args:
        mode: text
            - "vlm": text（textQwen2-VL）
            - "text_convert": textLLM
        model_name: text
        device: text

    Returns:
        MemoryToLLMtext
    """
    config = LLMIntegrationConfig(
        mode=mode,
        device=device
    )

    if model_name:
        if mode == "vlm":
            config.vlm_model = model_name
        else:
            config.llm_model = model_name

    return MemoryToLLM(config)


# ==================== text ====================

class MemoryAugmentedQA:
    """
    text

    text：
    1. text
    2. textvector
    3. text
    4. textLLM
    5. text

    text：
    ```python
    qa = MemoryAugmentedQA(manager, mode="vlm")
    answer = qa.ask("text？")
    ```
    """

    def __init__(
        self,
        memory_manager,
        mode: str = "vlm",
        device: str = "cuda"
    ):
        self.manager = memory_manager
        self.mode = mode
        self.device = device

        # text
        from .text_query import TextQueryEncoder, TextQueryConfig
        self.text_encoder = TextQueryEncoder(TextQueryConfig(
            encode_mode="clip_text",
            device=device
        ))

        # text
        self.bridge = MemoryToLLM(LLMIntegrationConfig(
            mode=mode,
            device=device
        ))

    def ask(
        self,
        question: str,
        top_k: int = 3,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        text

        Args:
            question: text
            top_k: text
            system_prompt: text

        Returns:
            text
        """
        # 1. text
        query_vector = self.text_encoder.encode(question)

        # 2. text
        results = self.manager.retrieve(
            query_vector=query_vector,
            top_k=top_k
        )

        if not results:
            return {
                "answer": "text。",
                "memories_used": 0,
                "retrieval_results": []
            }

        # 3. text
        context = self.bridge.prepare_context(
            results,
            include_images=(self.mode == "vlm"),
            generate_descriptions=(self.mode == "text_convert")
        )

        # 4. text
        if self.mode == "vlm":
            answer = self.bridge.query_vlm(question, context, system_prompt)
        else:
            answer = self.bridge.query_llm(question, context, system_prompt)

        return {
            "answer": answer,
            "memories_used": len(results),
            "retrieval_results": [
                {
                    "memory_id": r.memory_id,
                    "score": r.score,
                    "created_at": r.memory.meta.created_at.isoformat()
                }
                for r in results
            ]
        }
