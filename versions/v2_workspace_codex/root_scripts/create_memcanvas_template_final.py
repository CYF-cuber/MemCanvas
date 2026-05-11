import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

def create_memcanvas_template(example_num, output_path):
    """
    创建MemCanvas模板，文案完整，图片位置预留
    """
    
    fig = plt.figure(figsize=(16, 11))
    
    # 配置数据
    if example_num == 1:
        title = "Robot Task Planning - Example 1"
        task = "Task: Pick cup from counter and place on dining table"
        context = """A Franka robotic arm operates in a kitchen environment.
A ceramic cup sits on the kitchen counter near the sink.
The dining table is 1.5 meters away from the counter.
Objective: Pick up the cup safely and place it on the table."""
        
        plan = """Step 1:  Perceive cup location using gripper camera
Step 2:  Plan grasp point (cup center, safe approach)
Step 3:  Navigate gripper above cup (z=0.5m)
Step 4:  Lower gripper slowly to cup (z=0.15m)
Step 5:  Close gripper with appropriate force (2-5N)
Step 6:  Lift cup to safe height (z=0.8m)
Step 7:  Navigate arm to table position (x=5.75m)
Step 8:  Lower cup to table surface (z=0.85m)
Step 9:  Open gripper gradually (release cup)
Step 10: Retract arm to home position"""
        
        memory = """Memory State:
  • Cup properties: Ceramic, ~200g, fragile ★
  • Gripper capability: Max force 170N, min grip 2N
  • Approach strategy: Vertical descent (avoid tipping)
  • Safety constraint: Keep cup upright always
  • Success criterion: Cup placed upright, no spillage
  • Task completed: ✓ Yes"""
    
    else:  # example_num == 2
        title = "Robot Task Planning - Example 2"
        task = "Task: Organize colored blocks on shelf - stack by color"
        context = """A robotic arm operates in a workspace with a shelf.
Initial state: 6 colored blocks scattered on shelf
  - 3 red blocks (locations: L1, L3, L5)
  - 2 blue blocks (locations: L2, L4)
  - 1 green block (location: L6)
Objective: Sort blocks into color-specific stacks."""
        
        plan = """Phase 1: Inventory & Planning (Vision-based)
  Detect red blocks at: (1.2m, 0.1m), (2.4m, 0.2m), (4.2m, 0.15m)
  Detect blue blocks at: (1.8m, 0.1m), (3.6m, 0.2m)
  Detect green block at: (3.0m, 0.1m)
  Plan stacking order: Red→Red→Red, Blue→Blue, Green

Phase 2: Execute Stacking (Grasp & Place)
  Red stack execution:
    Pick R1(1.2) → Place at (6.5, 0.5) [Height: 1 block]
    Pick R2(2.4) → Place at (6.5, 0.5) [Height: 2 blocks]
    Pick R3(4.2) → Place at (6.5, 0.5) [Height: 3 blocks] ✓
  
  Blue stack execution:
    Pick B1(1.8) → Place at (7.5, 0.5) [Height: 1 block]
    Pick B2(3.6) → Place at (7.5, 0.5) [Height: 2 blocks] ✓
  
  Green stack execution:
    Pick G1(3.0) → Place at (8.5, 0.5) [Height: 1 block] ✓"""
        
        memory = """Memory State:
  • Block properties: Plastic, ~100g each, stackable
  • Stack stability: Max 3 blocks per stack (tested)
  • Gripper: Suitable for top-center grasping
  • Task phases: Detection → Planning → Execution
  • Success metrics: 3 color stacks, all upright
  • Task completed: ✓ Yes (100% success rate)"""
    
    # ===== 布局 =====
    ax_main = fig.add_subplot(111)
    ax_main.set_xlim(0, 16)
    ax_main.set_ylim(0, 11)
    ax_main.axis('off')
    
    # ----- 标题区域 -----
    y = 10.5
    ax_main.text(0.5, y, title, fontsize=22, weight='bold', va='top')
    ax_main.text(0.5, y-0.6, task, fontsize=13, style='italic', va='top')
    ax_main.plot([0.5, 15.5], [y-1.0, y-1.0], 'k-', linewidth=2)
    
    # ----- 左侧：图片位置 -----
    img_y_start = 9.2
    img_height = 4.5
    img_width = 6.5
    
    # 绘制图片框
    img_box = Rectangle((0.5, img_y_start - img_height), img_width, img_height,
                        linewidth=2.5, edgecolor='red', facecolor='#FFE6E6', 
                        linestyle='--')
    ax_main.add_patch(img_box)
    
    ax_main.text(0.5 + img_width/2, img_y_start - img_height/2, 
                "【 INSERT ROBOT IMAGE HERE 】\n\n" +
                f"Example {example_num}\n" +
                "请在此位置插入真实机器人操作图片\n" +
                "(高分辨率，640×480px 以上)",
                fontsize=11, ha='center', va='center', weight='bold',
                color='red', family='monospace')
    
    ax_main.text(0.5, img_y_start + 0.3, "[Robot Image Placeholder]", 
                fontsize=10, weight='bold', va='top')
    
    # ----- 右侧上方：场景描述 -----
    x_right = 7.5
    y = 9.2
    
    ax_main.text(x_right, y, "[Scene Context]", fontsize=11, weight='bold', va='top')
    ax_main.text(x_right + 0.2, y - 0.4, context, fontsize=9.5, va='top', 
                family='monospace', 
                bbox=dict(boxstyle='round,pad=0.7', facecolor='#F0F8FF', alpha=0.9))
    
    # ----- 右侧中方：任务规划 -----
    y = 5.8
    ax_main.text(x_right, y, "[Task Plan with Steps]", fontsize=11, weight='bold', va='top')
    ax_main.text(x_right + 0.2, y - 0.4, plan, fontsize=8.5, va='top',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.7', facecolor='#FFFACD', alpha=0.9))
    
    # ----- 底部：记忆笔记 -----
    y = 2.0
    ax_main.text(0.5, y, "[Memory & Task Status]", fontsize=11, weight='bold', va='top')
    ax_main.text(0.7, y - 0.4, memory, fontsize=9.5, va='top',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.7', facecolor='#E6F3FF', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Template {example_num} saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    output_dir = '/home/cyf/codex/paper_canvas_examples/final/'
    
    print("生成MemCanvas模板中...\n")
    create_memcanvas_template(1, f"{output_dir}TEMPLATE_1_pick_cup.png")
    create_memcanvas_template(2, f"{output_dir}TEMPLATE_2_organize_blocks.png")
    
    print("\n" + "="*70)
    print("✅ 完成！两个模板已生成")
    print("="*70)
    print(f"\n📄 模板位置:")
    print(f"   {output_dir}TEMPLATE_1_pick_cup.png")
    print(f"   {output_dir}TEMPLATE_2_organize_blocks.png")
    print("\n📝 文案说明:")
    print("   ✓ 场景描述 (Scene Context)")
    print("   ✓ 任务规划步骤 (Task Plan with Steps)")
    print("   ✓ 记忆状态笔记 (Memory & Task Status)")
    print("\n🖼️  图片位置:")
    print("   红色虚线框标记 → 用真实机器人图片替换")
    print("\n💡 使用方法:")
    print("   1. 下载或准备真实机器人操作图片")
    print("   2. 用图像编辑工具 (Photoshop/Figma/GIMP) 打开模板")
    print("   3. 在红色框位置插入你的机器人图片")
    print("   4. 导出为 PNG/PDF")
    print("="*70)

