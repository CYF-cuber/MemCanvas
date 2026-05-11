import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_robot_canvas_1():
    """机器人任务1: 拿杯子放到桌子上"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 标题
    ax.text(0.5, 11.5, "Robot Task Planning - Example 1", fontsize=18, weight='bold', va='top')
    ax.text(0.5, 11, "Task: Pick cup from counter and place on dining table", fontsize=12, style='italic', va='top')
    ax.plot([0.5, 9.5], [10.8, 10.8], 'k-', linewidth=1)
    
    # 左侧：场景描述
    y_pos = 10.3
    ax.text(0.5, y_pos, "[Scene Context]", fontsize=11, weight='bold', va='top')
    y_pos -= 0.4
    
    context = """A Franka robotic arm is in a kitchen environment. 
There is a ceramic cup on the kitchen counter near 
the sink. The dining table is 1.5 meters away.
The robot needs to pick up the cup safely and 
place it on the dining table."""
    
    ax.text(0.5, y_pos, context, fontsize=10, va='top', wrap=True, family='monospace')
    y_pos -= 2.2
    
    # 绘制简单的场景示意
    # 柜台
    counter = FancyBboxPatch((0.5, y_pos-1.5), 2, 0.8, boxstyle="round,pad=0.05", 
                             edgecolor='black', facecolor='#D2B48C', linewidth=2)
    ax.add_patch(counter)
    ax.text(1.5, y_pos-1.1, "Counter", fontsize=10, ha='center', weight='bold')
    
    # 杯子
    circle = plt.Circle((1.2, y_pos-1.8), 0.15, color='#87CEEB', ec='black', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(1.2, y_pos-2.3, "Cup", fontsize=9, ha='center')
    
    # 机器人手臂（简化）
    ax.plot([0.8, 0.8], [y_pos-0.5, y_pos+0.3], 'k-', linewidth=3, label='Robot Arm')
    circle_base = plt.Circle((0.8, y_pos-0.5), 0.2, color='#FF6B6B', ec='black', linewidth=1.5)
    ax.add_patch(circle_base)
    
    # 餐桌
    table = FancyBboxPatch((4.5, y_pos-1.5), 2.5, 0.8, boxstyle="round,pad=0.05",
                           edgecolor='black', facecolor='#8B4513', linewidth=2)
    ax.add_patch(table)
    ax.text(5.75, y_pos-1.1, "Dining Table", fontsize=10, ha='center', weight='bold')
    
    # 箭头表示动作
    ax.annotate('', xy=(4.5, y_pos-1.1), xytext=(2.7, y_pos-1.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(3.5, y_pos-0.5, "Pick & Place", fontsize=9, ha='center', 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    y_pos -= 3.5
    
    # 任务规划步骤
    ax.text(0.5, y_pos, "[Task Plan]", fontsize=11, weight='bold', va='top')
    y_pos -= 0.4
    
    steps = """Step 1: Move arm to counter position (x=1.2, y=0, z=0.5)
Step 2: Open gripper
Step 3: Move down to cup (z decreases to 0.15)
Step 4: Close gripper (grasp cup)
Step 5: Move up and back (z=0.8)
Step 6: Navigate to table (x=5.75, y=0, z=0.8)
Step 7: Move down to table surface (z=0.85)
Step 8: Open gripper (release cup)"""
    
    ax.text(0.5, y_pos, steps, fontsize=9, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 记忆注释
    y_pos -= 3.2
    ax.text(0.5, y_pos, "[Memory Notes]", fontsize=10, weight='bold', va='top')
    y_pos -= 0.3
    memory = "• Cup weight: ~200g (fragile ceramic)\n• Required grip force: 2-5N\n• Safe approach angle: vertical descent\n• Success criterion: cup placed upright on table"
    ax.text(0.5, y_pos, memory, fontsize=9, va='top', family='monospace')
    
    plt.tight_layout()
    plt.savefig('/home/cyf/codex/paper_canvas_examples/final/8_robot_task_pick_cup.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: robot_task_pick_cup.png")
    plt.close()

def create_robot_canvas_2():
    """机器人任务2: 整理物品"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 标题
    ax.text(0.5, 11.5, "Robot Task Planning - Example 2", fontsize=18, weight='bold', va='top')
    ax.text(0.5, 11, "Task: Organize items on shelf - stack blocks by color", fontsize=12, style='italic', va='top')
    ax.plot([0.5, 9.5], [10.8, 10.8], 'k-', linewidth=1)
    
    # 左侧：场景描述
    y_pos = 10.3
    ax.text(0.5, y_pos, "[Scene Context]", fontsize=11, weight='bold', va='top')
    y_pos -= 0.4
    
    context = """The robot operates in a workspace with a shelf.
There are 6 colored blocks scattered on the 
shelf: 3 red blocks, 2 blue blocks, 1 green block.
Task: Organize blocks into color-sorted stacks
on designated shelf positions."""
    
    ax.text(0.5, y_pos, context, fontsize=10, va='top', family='monospace')
    y_pos -= 1.8
    
    # 绘制货架和积木
    # 货架
    shelf = FancyBboxPatch((0.5, y_pos-2), 8, 0.3, 
                          edgecolor='black', facecolor='#808080', linewidth=2)
    ax.add_patch(shelf)
    ax.text(4.5, y_pos-2.5, "Shelf", fontsize=9, ha='center')
    
    # 绘制散乱的积木
    blocks = [
        (1.2, y_pos-1.5, 'red', 'R1'),
        (1.8, y_pos-1.5, 'blue', 'B1'),
        (2.4, y_pos-1.2, 'red', 'R2'),
        (3.0, y_pos-1.5, 'green', 'G1'),
        (3.6, y_pos-1.3, 'blue', 'B2'),
        (4.2, y_pos-1.5, 'red', 'R3'),
    ]
    
    for x, y, color, label in blocks:
        rect = FancyBboxPatch((x-0.25, y-0.25), 0.5, 0.5, 
                             boxstyle="round,pad=0.02", 
                             edgecolor='black', facecolor=color, linewidth=1.5, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=8, ha='center', va='center', weight='bold')
    
    # 目标位置
    ax.text(5.5, y_pos-0.5, "Target positions →", fontsize=10, weight='bold')
    targets = [
        (6.5, y_pos-1.5, 'red', 'RED\nSTACK'),
        (7.5, y_pos-1.5, 'blue', 'BLUE\nSTACK'),
        (8.5, y_pos-1.5, 'green', 'GREEN\nSTACK'),
    ]
    
    for x, y, color, label in targets:
        rect = FancyBboxPatch((x-0.3, y-0.6), 0.6, 1.2,
                             boxstyle="round,pad=0.03",
                             edgecolor='black', facecolor=color, linewidth=2, 
                             alpha=0.3, linestyle='--')
        ax.add_patch(rect)
        ax.text(x, y-0.9, label, fontsize=8, ha='center', weight='bold')
    
    y_pos -= 3.0
    
    # 任务规划步骤
    ax.text(0.5, y_pos, "[Task Plan with Memory]", fontsize=11, weight='bold', va='top')
    y_pos -= 0.4
    
    steps = """Phase 1 - Inventory (Remember state):
  → Detect 3 red blocks at (1.2, 2.4, 4.2)
  → Detect 2 blue blocks at (1.8, 3.6)
  → Detect 1 green block at (3.0)

Phase 2 - Execute stacking (Update memory):
  Step 1: Pick red block R1 (1.2) → place at (6.5)  [Red stack: 1/3]
  Step 2: Pick red block R2 (2.4) → place at (6.5)  [Red stack: 2/3]
  Step 3: Pick red block R3 (4.2) → place at (6.5)  [Red stack: 3/3] ✓
  Step 4: Pick blue block B1 (1.8) → place at (7.5) [Blue stack: 1/2]
  Step 5: Pick blue block B2 (3.6) → place at (7.5) [Blue stack: 2/2] ✓
  Step 6: Pick green block G1 (3.0) → place at (8.5) [Green stack: 1/1] ✓"""
    
    ax.text(0.5, y_pos, steps, fontsize=8.5, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    y_pos -= 3.8
    
    # 执行结果
    ax.text(0.5, y_pos, "[Result & Memory Update]", fontsize=10, weight='bold', va='top')
    y_pos -= 0.3
    result = "✓ All blocks organized by color\n✓ Memory: Task completed - 6 blocks sorted into 3 color stacks\n✓ Success rate: 100% (all blocks placed correctly)"
    ax.text(0.5, y_pos, result, fontsize=9, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('/home/cyf/codex/paper_canvas_examples/final/8_robot_task_organize_blocks.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: robot_task_organize_blocks.png")
    plt.close()

if __name__ == "__main__":
    create_robot_canvas_1()
    create_robot_canvas_2()
    print("\n✅ Done! Two robot task planning canvas examples created.")
