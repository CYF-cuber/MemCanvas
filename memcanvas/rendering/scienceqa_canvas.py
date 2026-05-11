#!/usr/bin/env python3
"""
ScienceQA Memory Canvas Generator

从ScienceQA数据集抽取数据，生成记忆画布实例。
将问题、选项、背景知识、解答等多模态信息渲染到统一画布上。
"""

import sys
sys.path.insert(0, '/home/cyf/memory')

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from memory_canvas.canvas import MemoryCanvas, CanvasConfig, CanvasMetadata
from memory_canvas.layers.content_layer import ContentLayer
from memory_canvas.layers.structure_layer import StructureLayer
from memory_canvas.layers.anchor_layer import AnchorLayer
from memory_canvas.renderers.text_renderer import TextRenderer, TextStyle
from memory_canvas.renderers.image_renderer import ImageRenderer, ImageStyle
from PIL import Image, ImageDraw, ImageFont


# 路径配置
SCIENCEQA_DIR = "/home/cyf/memory/ScienceQA"
PROBLEMS_PATH = f"{SCIENCEQA_DIR}/data/scienceqa/problems.json"
CAPTIONS_PATH = f"{SCIENCEQA_DIR}/data/captions.json"
IMAGES_DIR = f"{SCIENCEQA_DIR}/data/scienceqa/images"
OUTPUT_DIR = "/home/cyf/memory/demo_output/scienceqa_canvas"


class ScienceQACanvasGenerator:
    """ScienceQA画布生成器"""

    def __init__(self):
        self.problems = self._load_problems()
        self.captions = self._load_captions()
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _load_problems(self) -> Dict:
        """加载问题数据"""
        with open(PROBLEMS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_captions(self) -> Dict:
        """加载图片描述"""
        with open(CAPTIONS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('captions', {})

    def get_problem(self, pid: str) -> Optional[Dict]:
        """获取指定问题"""
        return self.problems.get(str(pid))

    def get_caption(self, pid: str) -> Optional[str]:
        """获取图片描述"""
        return self.captions.get(str(pid))

    def get_image_path(self, pid: str) -> Optional[str]:
        """获取图片路径"""
        problem = self.get_problem(pid)
        if not problem or not problem.get('image'):
            return None
        split = problem.get('split', 'train')
        img_path = f"{IMAGES_DIR}/{split}/{pid}/image.png"
        if os.path.exists(img_path):
            return img_path
        return None

    def generate_canvas(self, pid: str) -> Optional[MemoryCanvas]:
        """
        为单个问题生成画布

        Args:
            pid: 问题ID

        Returns:
            生成的MemoryCanvas对象
        """
        problem = self.get_problem(pid)
        if not problem:
            print(f"问题 {pid} 不存在")
            return None

        # 创建画布
        canvas = MemoryCanvas(
            config=CanvasConfig(width=1680, height=2240, patch_size=14),
            metadata=CanvasMetadata(
                memory_id=f"scienceqa_{pid}",
                timestamp_start=datetime.now(),
                source=[f"ScienceQA-{pid}"],
                modalities=["text", "image"] if problem.get('image') else ["text"]
            )
        )

        content = ContentLayer(canvas)

        # 1. 标题区域
        self._render_header(content, problem, pid)

        # 2. 图片/图片描述区域
        self._render_image_section(content, problem, pid)

        # 3. 问题和选项
        self._render_question(content, problem)

        # 4. 背景知识 (Lecture)
        self._render_lecture(content, problem)

        # 5. 解答 (Solution)
        self._render_solution(content, problem)

        # 添加结构层
        structure = StructureLayer(canvas)
        structure.draw_region_borders(color=(66, 133, 244, 120))
        structure.draw_modality_indicators()

        # 添加锚点层
        anchor = AnchorLayer(canvas)
        anchor.add_memory_id()
        anchor.add_page_number(1, 1)
        anchor.add_region_labels()

        # 添加元数据
        anchor.add_metadata_block({
            "dataset": "ScienceQA",
            "pid": pid,
            "subject": problem.get('subject', 'N/A'),
            "topic": problem.get('topic', 'N/A'),
            "grade": problem.get('grade', 'N/A'),
            "answer": problem.get('answer', -1)
        })

        return canvas

    def _render_header(self, content: ContentLayer, problem: Dict, pid: str):
        """渲染标题区域"""
        subject = problem.get('subject', 'Unknown')
        topic = problem.get('topic', 'Unknown')
        grade = problem.get('grade', 'Unknown')
        skill = problem.get('skill', '')

        header_text = (
            f"ScienceQA #{pid}\n"
            f"Subject: {subject} | Topic: {topic} | Grade: {grade}\n"
            f"Skill: {skill}"
        )
        content.add_text(header_text, font_size=24)

    def _render_image_section(self, content: ContentLayer, problem: Dict, pid: str):
        """渲染图片区域"""
        img_path = self.get_image_path(pid)
        caption = self.get_caption(pid)
        hint = problem.get('hint', '')

        # 先渲染hint
        if hint:
            content.add_text(f"[Context] {hint}", font_size=20)

        # 渲染实际图片
        if img_path:
            content.add_image(img_path, height=400, fit_mode="contain", alignment="left")
            if caption:
                content.add_text(f"[Caption] {caption}", font_size=18)
        elif caption:
            content.add_text(f"[Image Caption] {caption}", font_size=20)

    def _render_question(self, content: ContentLayer, problem: Dict):
        """渲染问题和选项"""
        question = problem.get('question', '')
        choices = problem.get('choices', [])
        answer_idx = problem.get('answer', -1)

        # 格式化选项
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        formatted_choices = []
        for i, choice in enumerate(choices):
            label = choice_labels[i] if i < len(choice_labels) else str(i)
            marker = "✓" if i == answer_idx else " "
            formatted_choices.append(f"  [{marker}] {label}. {choice}")

        question_text = (
            f"Question:\n{question}\n\n"
            f"Choices:\n" + '\n'.join(formatted_choices)
        )
        content.add_text(question_text, font_size=22)

    def _render_lecture(self, content: ContentLayer, problem: Dict):
        """渲染背景知识"""
        lecture = problem.get('lecture', '')
        if lecture:
            lecture_text = f"Background Knowledge:\n{lecture}"
            content.add_text(lecture_text, font_size=18)

    def _render_solution(self, content: ContentLayer, problem: Dict):
        """渲染解答"""
        solution = problem.get('solution', '')
        if solution:
            solution_text = f"Solution:\n{solution}"
            content.add_text(solution_text, font_size=18)

    def save_canvas(self, canvas: MemoryCanvas, pid: str):
        """保存画布"""
        output_path = f"{OUTPUT_DIR}/scienceqa_{pid}.png"
        canvas.save(output_path)
        print(f"已保存: {output_path}")
        return output_path


def main():
    """主函数"""
    print("=" * 60)
    print("ScienceQA Memory Canvas Generator")
    print("=" * 60)

    generator = ScienceQACanvasGenerator()
    print(f"已加载 {len(generator.problems)} 个问题")
    print(f"已加载 {len(generator.captions)} 个图片描述")

    # 选择一些有代表性的样本
    sample_pids = ['1', '2', '3', '5', '7']

    print(f"\n生成 {len(sample_pids)} 个画布实例...")

    for pid in sample_pids:
        print(f"\n处理问题 #{pid}...")
        canvas = generator.generate_canvas(pid)
        if canvas:
            generator.save_canvas(canvas, pid)

    print("\n" + "=" * 60)
    print(f"完成! 输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
