import matplotlib.pyplot as plt
from networkx.generators import line

def draw(t1, t2, t3, t4):
    th1, s1 = t1
    th2, s2 = t2
    th3, s3 = t3
    th4, s4 = t4


    # 生成横坐标
    x = range(len(th1))

    # 创建包含两个子图的大图，横向布局
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    line_width = 6
    # 绘制第一个子图
    axes[0].plot(x, th1, linestyle='-', label='no-balance',linewidth=line_width)
    axes[0].plot(x, th2, linestyle='-', label='revive',linewidth=line_width)
    axes[0].plot(x, th3, linestyle='-', label='GCB',linewidth=line_width)
    axes[0].plot(x, th4, linestyle='-', label='TPCR',linewidth=line_width)
    axes[0].set_xlabel('transaction num')
    axes[0].set_ylabel('successful rate')
    axes[0].set_title('Successful Rate Trend')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 绘制第二个子图

    axes[1].plot(x, s1, linestyle='-', label='no-balance',linewidth=line_width)
    axes[1].plot(x, s2, linestyle='-', label='revive',linewidth=line_width)
    axes[1].plot(x, s3, linestyle='-', label='GCB',linewidth=line_width)
    axes[1].plot(x, s4, linestyle='-', label='TPCR',linewidth=line_width)
    axes[1].set_xlabel('transaction num')
    axes[1].set_ylabel('throughput')
    axes[1].set_title('Throughput Trend')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 调整子图布局
    plt.tight_layout()

    # 显示图形
    plt.show()
    