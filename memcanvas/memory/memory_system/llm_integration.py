"""
LLM Integration - 记忆与LLM集成模块

将检索到的记忆注入到LLM中使用。

支持三种方式：
1. VLM方式：直接将画布图像传给视觉语言模型（如Qwen2-VL）
2. Embedding注入：将vision tokens投影后注入LLM embedding层
3. 文本转换：用VLM生成记忆的文本描述，传给纯文本LLM
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Literal
from pathlib import Path
import numpy as np
from PIL import Image


@dataclass
class LLMIntegrationConfig:
    """LLM集成配置"""
    # 集成模式: vlm, embedding_inject, text_convert
    mode: str = "vlm"
    # VLM模型名称
    vlm_model: str = "Qwen/Qwen2-VL-7B-Instruct"
    # 文本LLM模型名称
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    # 设备
    device: str = "cuda"
    # 最大记忆数量
    max_memories: int = 5
    # 最大token数
    max_tokens: int = 4096
    # 是否保存原始画布（用于VLM模式）
    save_canvas: bool = True


@dataclass
class MemoryContext:
    """记忆上下文，用于传递给LLM"""
    # 记忆ID列表
    memory_ids: List[str]
    # 画布图像列表（VLM模式）
    images: Optional[List[Image.Image]] = None
    # Vision tokens列表（embedding注入模式）
    vision_tokens: Optional[List[np.ndarray]] = None
    # 文本描述列表（文本转换模式）
    text_descriptions: Optional[List[str]] = None
    # 元数据
    metadata: List[Dict[str, Any]] = field(default_factory=list)


class MemoryToLLM:
    """
    记忆到LLM的桥接器

    将检索到的记忆转换为LLM可用的格式。

    使用示例：
    ```python
    # 方式1: VLM（推荐，如Qwen2-VL）
    bridge = MemoryToLLM(mode="vlm")
    context = bridge.prepare_context(retrieval_results)
    response = bridge.query_vlm("描述这些记忆的内容", context)

    # 方式2: 文本转换（给纯文本LLM用）
    bridge = MemoryToLLM(mode="text_convert")
    context = bridge.prepare_context(retrieval_results)
    response = bridge.query_llm("根据这些记忆回答问题", context)
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
        准备记忆上下文

        Args:
            retrieval_results: 检索结果列表（RetrievalResult）
            include_images: 是否包含图像
            include_tokens: 是否包含vision tokens
            generate_descriptions: 是否生成文本描述

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

            # 收集元数据
            meta_info = {
                "memory_id": result.memory_id,
                "score": result.score,
                "created_at": memory.meta.created_at.isoformat(),
                "modalities": memory.meta.modalities,
                "source": memory.meta.source
            }
            metadata.append(meta_info)

            # 收集vision tokens
            if include_tokens:
                vision_tokens.append(memory.tokens)

            # 如果需要图像，从存储路径加载
            # 注意：需要在存储时保存原始画布
            if include_images:
                canvas_image = self._load_canvas_image(result.memory_id)
                if canvas_image:
                    images.append(canvas_image)

        # 生成文本描述
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
        """加载画布图像（需要在存储时同时保存）"""
        # 这里需要实现从存储加载原始画布的逻辑
        # 暂时返回None，后续可以扩展
        return None

    def _generate_descriptions(
        self,
        images: List[Image.Image],
        metadata: List[Dict]
    ) -> List[str]:
        """使用VLM生成图像描述"""
        if not images:
            return [f"记忆 {m['memory_id']}，创建于 {m['created_at']}" for m in metadata]

        descriptions = []
        for img, meta in zip(images, metadata):
            desc = self._describe_image(img)
            descriptions.append(f"[记忆 {meta['memory_id']}]\n{desc}")

        return descriptions

    def _describe_image(self, image: Image.Image) -> str:
        """使用VLM描述单张图像"""
        self._init_vlm()

        if self._vlm_model is None:
            return "（无法生成描述）"

        # 使用Qwen2-VL生成描述
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "请详细描述这张图片的内容，包括文字、图表、布局等所有信息。"}
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
            return f"（描述生成失败: {e}）"

    # ==================== VLM 模式 ====================

    def query_vlm(
        self,
        question: str,
        context: MemoryContext,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        使用VLM（如Qwen2-VL）处理记忆和问题

        Args:
            question: 用户问题
            context: 记忆上下文
            system_prompt: 系统提示

        Returns:
            模型回答
        """
        self._init_vlm()

        if self._vlm_model is None:
            return "VLM模型未加载"

        # 构建消息
        content = []

        # 添加图像
        if context.images:
            for i, img in enumerate(context.images):
                content.append({"type": "image", "image": img})
                content.append({
                    "type": "text",
                    "text": f"[记忆 {i+1}，相关度: {context.metadata[i]['score']:.2f}]"
                })

        # 添加问题
        content.append({"type": "text", "text": f"\n根据以上记忆，回答问题：{question}"})

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
            return f"生成失败: {e}"

    # ==================== 文本LLM模式 ====================

    def query_llm(
        self,
        question: str,
        context: MemoryContext,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        使用纯文本LLM处理记忆和问题

        需要先将记忆转换为文本描述。

        Args:
            question: 用户问题
            context: 记忆上下文（需要text_descriptions）
            system_prompt: 系统提示

        Returns:
            模型回答
        """
        self._init_llm()

        if self._llm_model is None:
            return "LLM模型未加载"

        # 构建提示
        memory_text = ""
        if context.text_descriptions:
            for i, desc in enumerate(context.text_descriptions):
                memory_text += f"\n--- 记忆 {i+1} ---\n{desc}\n"
        else:
            # 使用元数据
            for i, meta in enumerate(context.metadata):
                memory_text += f"\n--- 记忆 {i+1} ---\n"
                memory_text += f"ID: {meta['memory_id']}\n"
                memory_text += f"创建时间: {meta['created_at']}\n"
                memory_text += f"相关度: {meta['score']:.2f}\n"

        prompt = f"""以下是从记忆库中检索到的相关记忆：
{memory_text}

请根据以上记忆，回答用户的问题：{question}"""

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
            return f"生成失败: {e}"

    # ==================== 初始化模型 ====================

    def _init_vlm(self):
        """初始化VLM"""
        if self._vlm_model is not None:
            return

        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch

            print(f"加载VLM: {self.config.vlm_model}")
            self._vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.vlm_model,
                torch_dtype=torch.bfloat16,
                device_map=self.config.device
            )
            self._vlm_processor = AutoProcessor.from_pretrained(self.config.vlm_model)

        except Exception as e:
            print(f"VLM加载失败: {e}")
            self._vlm_model = None

    def _init_llm(self):
        """初始化文本LLM"""
        if self._llm_model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"加载LLM: {self.config.llm_model}")
            self._llm_model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model,
                torch_dtype=torch.bfloat16,
                device_map=self.config.device
            )
            self._llm_tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)

        except Exception as e:
            print(f"LLM加载失败: {e}")
            self._llm_model = None


# ==================== 便捷函数 ====================

def create_memory_bridge(
    mode: Literal["vlm", "text_convert"] = "vlm",
    model_name: Optional[str] = None,
    device: str = "cuda"
) -> MemoryToLLM:
    """
    快速创建记忆桥接器

    Args:
        mode: 模式
            - "vlm": 使用视觉语言模型（推荐Qwen2-VL）
            - "text_convert": 转换为文本后使用纯文本LLM
        model_name: 模型名称
        device: 设备

    Returns:
        MemoryToLLM实例
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


# ==================== 完整示例类 ====================

class MemoryAugmentedQA:
    """
    记忆增强的问答系统

    完整的端到端流程：
    1. 接收用户问题
    2. 将问题编码为查询向量
    3. 检索相关记忆
    4. 将记忆和问题传给LLM
    5. 返回答案

    使用示例：
    ```python
    qa = MemoryAugmentedQA(manager, mode="vlm")
    answer = qa.ask("上次会议讨论了什么？")
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

        # 文本编码器
        from .text_query import TextQueryEncoder, TextQueryConfig
        self.text_encoder = TextQueryEncoder(TextQueryConfig(
            encode_mode="clip_text",
            device=device
        ))

        # 记忆桥接器
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
        提问并获取答案

        Args:
            question: 用户问题
            top_k: 检索的记忆数量
            system_prompt: 系统提示

        Returns:
            包含答案和检索信息的字典
        """
        # 1. 编码问题
        query_vector = self.text_encoder.encode(question)

        # 2. 检索相关记忆
        results = self.manager.retrieve(
            query_vector=query_vector,
            top_k=top_k
        )

        if not results:
            return {
                "answer": "没有找到相关记忆。",
                "memories_used": 0,
                "retrieval_results": []
            }

        # 3. 准备上下文
        context = self.bridge.prepare_context(
            results,
            include_images=(self.mode == "vlm"),
            generate_descriptions=(self.mode == "text_convert")
        )

        # 4. 生成答案
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
