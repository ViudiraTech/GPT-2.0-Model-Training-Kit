# By ViudiraTech
# MIT LICENSE ¯\_(ツ)_/¯

# 导入PyTorch库，让torch支持DML
import torch_directml

# 导入PyTorch库，用于构建和训练神经网络
import torch

# 打印程序标题
print('GPT-2.0语言模型评估程序\n')

# 从自定义模块gpt_model中导入GPT类和其他相关函数
from gpt_model import *

# 程序的主执行部分
if __name__ == '__main__':

    # 根据环境变量use_gpu的值决定使用GPU还是CPU进行计算
    if use_gpu == 'GPU':
        device = torch.device("cuda")  # 如果可用，使用GPU进行加速
        model = GPT().to(device)  # 将模型移动到GPU上
    elif use_gpu == 'DML':
        device = torch_directml.device(0)  # 如果可用，使用AMD显卡进行加速
        model = GPT().to(device)  # 将模型移动到AMD显卡上
    else:
        device = torch.device("cpu")  # 如果不使用GPU，则使用CPU
        model = GPT().cpu()  # 将模型限制在CPU上

    # 尝试加载预训练的GPT-2.0模型
    try:
        model.load_state_dict(torch.load('GPT2.pt'))
    except FileNotFoundError:
        # 如果模型文件不存在，打印错误信息并退出程序
        print("无法找到GPT-2.0模型文件'GPT2.pt'，请检查文件是否存在。")
        input('按回车键退出...')
        exit()

    # 将模型设置为评估模式，这会影响模型中的某些层的行为（如Dropout）
    model.eval()

    # 初始化一个空字符串用于存储对话历史
    sentence = ''
    # 开始一个无限循环，用于持续接收用户输入
    while True:
        temp_sentence = input("输入:")
        # 将用户的输入添加到对话历史中，并用制表符分隔
        sentence += (temp_sentence + '\t')
        # 如果对话历史的长度超过模型的最大输入长度（此处为200个字符），则进行裁剪
        if len(sentence) > 200:
            t_index = sentence.find('\t')
            sentence = sentence[t_index + 1:]
        # 使用模型生成回答并打印输出
        print("输出:", model.answer(sentence), "\n")
        