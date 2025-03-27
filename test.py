def find_almost_equal_sequence(word1, word2):
    m = len(word2)
    n = len(word1)
    INF = float('inf')
    
    # 初始化动态规划表和前驱表
    dp = [[INF] * 2 for _ in range(m + 1)]
    dp[0][0] = -1  # 初始状态，没有选择任何字符
    
    # 前驱表记录（prev_j, prev_k, i）
    prev = [[(None, None, None)] * 2 for _ in range(m + 1)]
    
    for i in range(n):
        # 使用临时数组保存当前状态，避免同一i的处理中状态覆盖
        temp_dp = [row[:] for row in dp]
        for j in range(m + 1):
            for k in range(2):
                if temp_dp[j][k] != INF and temp_dp[j][k] < i:
                    if j < m:
                        # 当前字符匹配，无需修改
                        if word1[i] == word2[j]:
                            if i < dp[j + 1][k]:
                                dp[j + 1][k] = i
                                prev[j + 1][k] = (j, k, i)
                        # 当前字符不匹配，尝试修改一次
                        elif k == 0:
                            if i < dp[j + 1][1]:
                                dp[j + 1][1] = i
                                prev[j + 1][1] = (j, 0, i)
    
    # 检查是否存在有效解
    if dp[m][0] == INF and dp[m][1] == INF:
        return []
    
    # 选择字典序最小的路径
    if dp[m][0] <= dp[m][1]:
        current_j, current_k = m, 0
    else:
        current_j, current_k = m, 1
    
    # 回溯构造下标序列
    sequence = []
    while current_j > 0:
        prev_state = prev[current_j][current_k]
        if prev_state == (None, None, None):
            return []
        prev_j, prev_k, i = prev_state
        sequence.append(i)
        current_j, current_k = prev_j, prev_k
    
    # 反转得到正确顺序
    sequence.reverse()
    
    # 验证序列长度是否正确
    return sequence if len(sequence) == m else []

# 示例用法
word1 = "abcdefg"
word2 = "abd"
result = find_almost_equal_sequence(word1, word2)
print(result)