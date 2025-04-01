import matplotlib.pyplot as plt

def draw(t1, t2, t3):
    list1, list4 = t1
    list2, list5 = t2
    list3, list6 = t3
    # 确保六个列表长度相等
    assert len(list1) == len(list2) == len(list3) == len(list4) == len(list5) == len(list6), "输入列表长度必须相等"

    # 生成横坐标
    x = range(len(list1))

    # 创建包含两个子图的大图，横向布局
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制第一个子图
    axes[0].plot(x, list1, marker='o', linestyle='-', label='List1')
    axes[0].plot(x, list2, marker='s', linestyle='--', label='List2')
    axes[0].plot(x, list3, marker='^', linestyle='-.', label='List3')
    axes[0].set_xlabel('Index (0-len(list))')
    axes[0].set_ylabel('Value')
    axes[0].set_title('First Three Lists Trend')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 绘制第二个子图
    axes[1].plot(x, list4, marker='o', linestyle='-', label='List4')
    axes[1].plot(x, list5, marker='s', linestyle='--', label='List5')
    axes[1].plot(x, list6, marker='^', linestyle='-.', label='List6')
    axes[1].set_xlabel('Index (0-len(list))')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Last Three Lists Trend')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 调整子图布局
    plt.tight_layout()

    # 显示图形
    plt.show()
    