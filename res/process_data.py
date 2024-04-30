# By ViudiraTech
# GPL-2.0 LICENSE ¯\_(ツ)_/¯

# 导入json模块，用于处理JSON数据格式
import json

# 尝试打开并读取data.txt文件，如果文件不存在则捕获异常
try:
    with open('data.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()  # 读取文件的所有行
except FileNotFoundError:
    # 如果文件不存在，打印错误信息提示用户
    print("无法找到语料文件'data.txt'，请检查文件是否存在。")
    # 等待用户按下回车键后退出程序
    input('按回车键退出...')
    exit()  # 退出程序

# 初始化一个列表用于存储处理后的数据
train_datas = []
# 初始化一个字符串用于临时存储每行的数据
temp_data = ''

# 遍历读取的行，处理数据并存储到train_datas列表中
for line in lines:
    # 如果当前行不是空行（即不是仅包含换行符的行）
    if line != '\n':
        line = line.strip()  # 去除行尾的换行符和多余空白字符
        temp_data += (line + '\t')  # 将处理后的行添加到临时数据字符串中，并用制表符分隔
    else:
        # 如果当前行是空行，将临时数据字符串添加到数据列表中，并重置临时数据字符串
        train_datas.append(temp_data)
        temp_data = ''

# 将处理后的数据写入到dataset.txt文件中
with open('dataset.txt', 'w', encoding='utf-8') as f:
    for train_data in train_datas:
        f.write(train_data + '\n')  # 写入每条数据后添加换行符

# 定义一个函数，用于根据数据生成词汇表
def get_dict(datas):
    # 初始化一个字典用于统计词频
    word_count = {}
    for data in datas:
        data = data.strip().replace('\t', '')  # 去除每条数据的空白字符和制表符
        for word in data:
            word_count.setdefault(word, 0)  # 如果词不在字典中，则添加并设置计数为0
            word_count[word] += 1  # 增加词的计数

    # 初始化一个包含特殊标记的字典
    word2id = {"<pad>": 0, "<unk>": 1, "<sep>": 2}

    # 根据词频统计结果，为每个词分配一个唯一的ID
    temp = {word: i + len(word2id) for i, word in enumerate(word_count.keys())}
    word2id.update(temp)
    # 将ID映射回词，创建id2word字典
    id2word = list(word2id.keys())
    return word2id, id2word

# 程序的主执行部分
if __name__ == '__main__':
    # 读取处理后的数据集文件
    with open('dataset.txt', 'r', encoding='utf-8') as f:
        datas = f.readlines()
    # 调用get_dict函数生成词汇表
    word2id, id2word = get_dict(datas)

    # 创建一个包含词汇表数据的字典
    dict_datas = {"word2id": word2id, "id2word": id2word}

    # 将词汇表字典保存为JSON格式的文件
    json.dump(dict_datas, open('dict_datas.json', 'w', encoding='utf-8'))

    # 打印完成信息
    print("整理任务按计划完成，数据集文件'dataset.txt'和词汇表文件'dict_datas.json'已经输出至当前工作目录。")
    # 等待用户按下回车键后退出程序
    input('按回车键退出...')
    exit()  # 退出程序
    