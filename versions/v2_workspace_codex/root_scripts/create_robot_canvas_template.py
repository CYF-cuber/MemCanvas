"""
MemCanvas Generator for Robot Task Planning
用法: python create_robot_canvas_template.py --image1 <path> --image2 <path>
如果不提供图片，将使用占位符
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import sys
import os

def create_canvas_with_image(image_path, title, task_desc, context_text, 
                             task_plan_text, memory_notes, output_path):
    """
    创建包含真实机器人图片的MemCanvas
    """
    fig = plt.figure(figsize=(14, 12))
    
    # 创建两个子图：左侧图片，右侧文本
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.2], width_ratios=[1, 1], 
                          hspace=0.3, wspace=0.3)
    
    # 上方：标题和任务描述
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    ax_title.text(0.05, 0.9, title, fontsize=18, weight='bold', transform=ax_title.transAxes)
    ax_title.text(0.05, 0.5, task_desc, fontsize=12, style='italic', transform=ax_title.transAxes)
    ax_title.plot([0.05, 0.95], [0.4, 0.4], 'k-', linewidth=1, transform=ax_title.transAxes)
    
    # 左侧：机器人图片
    ax_img = fig.add_subplot(gs[1, 0])
    try:
        img = Image.open(image_path)
        ax_img.imshow(img)
        ax_img.set_title("Robot Setup", fontsize=12, weight='bold')
    except Exception as e:
        ax_img.text(0.5, 0.5, f'Image not found\n{image_path}', 
                   ha='center', va='center', fontsize=10, transform=ax_img.transAxes)
        ax_img.set_title("Robot Setup (Placeholder)", fontsize=11)
    ax_img.axis('off')
    
    # 右侧：文本信息
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis('off')
    
    y_pos = 0.95
    
    # 场景上下文
    ax_text.text(0.05, y_pos, "[Scene Context]", fontsize=11, weight='bold', 
                transform=ax_text.transAxes, va='top')
    y_pos -= 0.08
    ax_text.text(0.05, y_pos, context_text, fontsize=9, transform=ax_text.transAxes, 
                va='top', wrap=True, family='monospace')
    y_pos -= 0.28
    
    # 任务规划
    ax_text.text(0.05, y_pos, "[Task Plan]", fontsize=11, weight='bold',
                transform=ax_text.transAxes, va='top')
    y_pos -= 0.08
    ax_text.text(0.05, y_pos, task_plan_text, fontsize=8.5, transform=ax_text.transAxes,
                va='top', family='monospace')
    y_pos -= 0.35
    
    # 记忆笔记
    ax_text.text(0.05, y_pos, "[Memory Notes]", fontsize=10, weight='bold',
                transform=ax_text.transAxes, va='top')
    y_pos -= 0.08
    ax_text.text(0.05, y_pos, memory_notes, fontsize=8.5, transform=ax_text.transAxes,
                va='top', family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

# 示例配置
configs = [
    {
        'name': 'robot_pick_cup_real',
        'title': 'Robot Task Planning - Example 1: Pick & Place',
        'task': 'Task: Pick up cup from counter and place on dining table',
        'context': '''A Franka robotic arm operates in a kitchen 
environment. A ceramic cup is on the counter.
The dining table is 1.5m away. Task: Pick cup 
safely and place it on the table.''',
        'plan': '''Step 1: Detect cup location (vision)
Step 2: Plan grasp point (center of cup)
Step 3: Navigate gripper above cup
Step 4: Lower and grasp (2-5N force)
Step 5: Lift to safe height
Step 6: Navigate to table destination
Step 7: Place cup on table
Step 8: Release and retract''',
        'memory': '''• Cup properties: ceramic, ~200g, fragile
• Safe grip: vertical approach, pinch grip
• Place location: center of dining table
• Success: cup upright, no spillage''',
        'image_hint': 'Franka robot with gripper picking up a cup'
    },
    {
        'name': 'robot_organize_blocks_real',
        'title': 'Robot Task Planning - Example 2: Object Sorting',
        'task': 'Task: Organize colored blocks on shelf - stack by color',
        'context': '''Robot operates on a workspace shelf with 
6 colored blocks: 3 red, 2 blue, 1 green.
Task: Sort blocks into color-specific stacks
on designated shelf positions.''',
        'plan': '''Phase 1: Inventory (Vision-based detection)
  Detect: Red blocks at pos1, pos2, pos3
  Detect: Blue blocks at pos4, pos5  
  Detect: Green block at pos6

Phase 2: Execution (Grasp & Place)
  Pick red1→red_stack, red2→red_stack, red3→red_stack
  Pick blue1→blue_stack, blue2→blue_stack
  Pick green1→green_stack
  Verify all stacks stable''',
        'memory': '''• Block weight: ~100g each, stable stacks
• Grasp strategy: top-center for stability
• Stack height: max 3 blocks per stack
• Success: 3 color-sorted stacks, all upright''',
        'image_hint': 'Robot manipulator performing object sorting/stacking task'
    }
]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create MemCanvas with real robot images')
    parser.add_argument('--image1', type=str, help='Path to first robot image')
    parser.add_argument('--image2', type=str, help='Path to second robot image')
    
    args = parser.parse_args()
    
    images = [args.image1, args.image2]
    output_dir = '/home/cyf/codex/paper_canvas_examples/final/'
    
    for i, (config, image_path) in enumerate(zip(configs, images)):
        if image_path is None:
            print(f"\n⚠️  Example {i+1}: No image provided")
            print(f"   Image hint: {config['image_hint']}")
            print(f"   To add image, run:")
            print(f"   python create_robot_canvas_template.py --image1 <path1> --image2 <path2>")
            continue
            
        output_path = os.path.join(output_dir, f"9_robot_{config['name']}.png")
        
        print(f"\n📸 Creating canvas {i+1} with image: {image_path}")
        create_canvas_with_image(
            image_path=image_path,
            title=config['title'],
            task_desc=config['task'],
            context_text=config['context'],
            task_plan_text=config['plan'],
            memory_notes=config['memory'],
            output_path=output_path
        )

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("\n1. 找到真实机器人图片 (推荐来源):")
    print("   - RT-1: https://robotics-transformer1.github.io/")
    print("   - DROID: https://droid-dataset.github.io/")
    print("   - MIT CSAIL: 搜索 'Dense Object Nets robot picking'")
    print("\n2. 下载图片到本地")
    print("\n3. 运行命令生成画布:")
    print("   python create_robot_canvas_template.py --image1 <path1> --image2 <path2>")
    print("\n4. 输出文件:")
    print(f"   {output_dir}9_robot_*.png")

