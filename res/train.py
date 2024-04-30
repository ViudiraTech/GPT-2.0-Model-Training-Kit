# By ViudiraTech
# GPL-2.0 LICENSE ¯\_(ツ)_/¯

import json # 导入json模块，用于处理JSON数据格式
import torch # 导入PyTorch库，这是一个流行的深度学习框架
import torch.utils.data as Data # 导入PyTorch的数据加载工具，用于加载和管理数据集
from torch import nn, optim # 从PyTorch库中导入神经网络模块，包含构建模型所需的层和函数
import numpy as np # 导入NumPy库，一个用于科学计算的库，常用于数据处理和数学运算
import time # 导入time模块，用于获取时间信息，常用于性能测试
from tqdm import tqdm # 从tqdm库导入进度条工具，用于在控制台显示进度信息
import os # 导入os模块，用于与操作系统交互，如文件路径操作等
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图

# 在脚本顶部定义一个列表来存储损失值
loss_values = []

# 打印程序标题
print('GPT-2.0语言模型训练程序\n')

# 从自定义模块gpt_model中导入GPT类和其他相关函数
from gpt_model import *

# 定义一个函数，名为make_data，它接受一个数据列表作为参数
def make_data(datas):
    # 初始化一个空列表，用于存储处理后的训练数据
    train_datas = []
    # 遍历输入数据列表中的每一个数据项
    for data in datas:
        # 使用strip()方法去除数据项两端的空白字符（如空格、换行符等）
        data = data.strip()
        # 将数据项中的制表符（'\t'）替换为特殊的分隔符"<sep>"，用于标记数据字段的边界
        # 并在数据项的末尾添加一个额外的分隔符
        train_data = [i if i != '\t' else "<sep>" for i in data] + ['<sep>']
        # 将处理后的数据项添加到train_datas列表中
        train_datas.append(train_data)
    # 函数返回处理后的训练数据列表
    return train_datas

# 导入必要的库
import torch
from torch.utils.data import Dataset

# 定义一个自定义的数据集类，继承自torch.utils.data.Dataset
class MyDataSet(Dataset):
    # 初始化方法
    def __init__(self, datas):
        # 存储传入的数据列表
        self.datas = datas

    # 获取数据集中的单个样本的方法
    def __getitem__(self, item):
        # 从数据列表中获取指定索引的样本
        data = self.datas[item]
        
        # 定义解码器的输入和输出
        decoder_input = data[:-1]  # 样本中除了最后一个元素的所有元素作为解码器输入
        decoder_output = data[1:]   # 样本中除了第一个元素的所有元素作为解码器输出
        
        # 计算解码器输入和输出的长度
        decoder_input_len = len(decoder_input)
        decoder_output_len = len(decoder_output)
        
        # 返回一个包含解码器输入、输出及其长度的字典
        return {
            "decoder_input": decoder_input,
            "decoder_input_len": decoder_input_len,
            "decoder_output": decoder_output,
            "decoder_output_len": decoder_output_len
        }

    # 获取数据集的大小的方法
    def __len__(self):
        # 返回数据列表的长度
        return len(self.datas)

    # 对一个批次的数据进行填充的方法
    def padding_batch(self, batch):
        # 提取批次中每个样本的解码器输入和输出的长度
        decoder_input_lens = [d["decoder_input_len"] for d in batch]
        decoder_output_lens = [d["decoder_output_len"] for d in batch]
        
        # 找出批次中最长的解码器输入和输出的长度
        decoder_input_maxlen = max(decoder_input_lens)
        decoder_output_maxlen = max(decoder_output_lens)
        
        # 对批次中的每个样本的解码器输入和输出进行填充
        for d in batch:
            # 填充解码器输入直到最大长度
            d["decoder_input"].extend([word2id["<pad>"]] * (decoder_input_maxlen - d["decoder_input_len"]))
            # 填充解码器输出直到最大长度
            d["decoder_output"].extend([word2id["<pad>"]] * (decoder_output_maxlen - d["decoder_output_len"]))
        
        # 将填充后的解码器输入和输出转换为张量
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch], dtype=torch.long)
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch], dtype=torch.long)
        
        # 返回填充后的解码器输入和输出的张量
        return decoder_inputs, decoder_outputs

# 定义一个函数epoch_time，计算两个时间戳（以秒为单位）之间的差值，并将其转换为分钟和秒的形式返回
def epoch_time(start_time, end_time):
    # 计算经过的时间（单位：秒）
    elapsed_time = end_time - start_time
    
    # 将经过的总秒数转换为整数分钟数
    elapsed_mins = int(elapsed_time / 60)
    
    # 计算剩余的秒数（经过的总秒数减去已计算出的分钟数乘以60）
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    
    # 返回以分钟和秒表示的经过时间
    return elapsed_mins, elapsed_secs

# 定义训练步骤函数
def train_step(model, data_loader, optimizer, criterion, clip=1, print_every=None):
    # 将模型设置为训练模式
    model.train()
    
    # 如果print_every未指定，则默认为1，表示每批次打印一次损失
    if print_every is None or print_every == 0:
        print_every = 1
    
    # 初始化打印损失的累积变量
    print_loss_total = 0  # 用于记录每个打印周期内的损失总和
    print_loss_avg = 0    # 初始化平均损失为0
    epoch_loss = 0
    
    # 使用tqdm库美化进度条输出
    for i, (dec_inputs, dec_outputs) in enumerate(tqdm(data_loader)):
        # dec_inputs和dec_outputs分别是解码器的输入和输出，形状分别为[batch_size, tgt_len]和[batch_size, tgt_len]
        
        # 优化器的梯度归零
        optimizer.zero_grad()
        
        # 根据是否使用GPU来决定数据和模型的运行设备
        if use_gpu == 'GPU':
            # 如果使用GPU，则将输入和输出张量转移到GPU上
            dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
        else:
            # 如果不使用GPU，则将它们转移到CPU上
            dec_inputs, dec_outputs = dec_inputs.cpu(), dec_outputs.cpu()
        
        # 调用模型进行预测，outputs是模型的输出，dec_self_attns是自注意力机制的输出
        outputs, dec_self_attns = model(dec_inputs)
        
        # 计算预测结果和真实标签之间的损失
        loss = criterion(outputs, dec_outputs.view(-1))  # outputs的形状需要调整为[batch_size * tgt_len, tgt_vocab_size]
        
        # 将损失加到累积损失中
        print_loss_total += loss.item()
        epoch_loss += loss.item()
        
        # 反向传播计算梯度
        loss.backward()
        
        # 对梯度进行裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # 根据计算出的梯度更新模型的参数
        optimizer.step()
        
        # 计算当前批次的平均损失
        print_loss_total += loss.item()
        epoch_loss += loss.item()
        batch_loss_avg = print_loss_total / (i + 1)
        
        # 如果满足打印条件，则打印平均损失
        if print_every and (i + 1) % print_every == 0:
            print_loss_avg = batch_loss_avg  # 更新平均损失值
            print(f'\tCurrent Loss: {print_loss_avg:.4f}')  # 打印当前平均损失
        
    # 返回该epoch的平均损失和打印周期内的平均损失
    return epoch_loss / len(data_loader), print_loss_avg if print_loss_avg != 0 else epoch_loss / len(data_loader)

# 定义训练函数train，输入参数包括模型model和数据加载器data_loader
def train(model, data_loader):

    # 根据use_gpu变量判断是否使用GPU进行计算
    if use_gpu == 'GPU':
        # 如果使用GPU，将损失函数设置在GPU设备上并忽略索引为0的标签
        criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    else:
        # 如果不使用GPU，则在CPU上定义损失函数
        criterion = nn.CrossEntropyLoss(ignore_index=0).cpu()

    # 使用Adam优化器初始化模型参数，学习率为1e-4
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 对于每个训练轮次（epoch）
    for epoch in range(epochs):
        # 记录当前epoch开始时间
        start_time = time.time()

        # 执行单个训练步骤并获取训练损失
        train_loss = train_step(model, data_loader, optimizer, criterion, CLIP, print_every=10)

        train_loss, avg_loss = train_step(model, data_loader, optimizer, criterion, CLIP, print_every=10)
        loss_values.append(avg_loss)  # 收集损失值

        # 记录当前epoch结束时间
        end_time = time.time()

        # 在每轮训练结束后保存模型状态字典到文件'GPT2.pt'
        torch.save(model.state_dict(), 'GPT2.pt')

        # 转换并打印该epoch所用的时间（分钟和秒）
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')

        # 输出训练损失信息
        print(f'\tTrain Loss: {train_loss:.3f}')

# 在训练完成后绘制损失曲线
def plot_loss_curve():
    plt.figure(figsize=(10, 5))  # 创建一个新的图形
    plt.plot(loss_values, label='Training loss')  # 绘制损失曲线
    plt.title('Training Loss Over Epochs')  # 添加标题
    plt.xlabel('Epoch')  # 添加x轴标签
    plt.ylabel('Loss')  # 添加y轴标签
    plt.legend()  # 显示图例
    plt.show()  # 显示图形

# 定义一个函数，用于打印模型的总参数数量和可训练参数数量
def print_num_parameters(model):
    # 计算并打印模型中所有参数的数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    # 计算并打印模型中所有需要梯度的参数数量（即可训练参数的数量）
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

# 检查当前Python脚本是否作为主程序运行
if __name__ == '__main__':
    # 尝试打开名为'dataset.txt'的数据集文件
    try:
        with open('dataset.txt', 'r', encoding='utf-8') as f:
            datas = f.readlines()
    except FileNotFoundError:
        # 如果文件不存在，打印错误信息并退出程序
        print("无法找到数据集文件'dataset.txt'，请检查文件是否存在。")
        input('按回车键退出...')
        exit()

    # 调用make_data函数处理数据
    train_data = make_data(datas)

    # 将文本数据转换为数字ID形式，假设word2id是一个预定义的字典，用于词汇转换
    train_num_data = [[word2id[word] for word in line] for line in train_data]

    # 从用户那里获取每次迭代中用于训练的样本数量
    batch_size_input = input("每次迭代中用于训练的样本数量(batch_size):")
    # 将输入的字符串转换为整数
    batch_size = int(batch_size_input)
    
    # 从用户那里获取整个训练数据集被完整遍历和用于训练的次数
    epochs_size_input = input("整个训练数据集被完整遍历和用于训练的次数(epochs):")
    # 将输入的字符串转换为整数
    epochs = int(epochs_size_input)
    print()  # 打印一个空行，为了更好的输出格式

    # 创建一个自定义的数据集对象MyDataSet，并使用Data.DataLoader来创建一个数据加载器
    dataset = MyDataSet(train_num_data)
    data_loader = Data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.padding_batch)

    # 根据环境变量use_gpu的值决定使用GPU还是CPU进行模型训练
    if use_gpu == 'GPU':
        model = GPT().to(device)  # 使用GPU运算
    else:
        model = GPT().cpu()  # 使用CPU运算

    # 这里注释掉的代码是用来加载预训练模型的，但实际代码中没有提供这部分的实现
    # model.load_state_dict(torch.load('GPT2.pt'))

    # 调用train函数进行模型训练，这里的train函数和GPT类需要在其他地方定义
    train(model, data_loader)
    
    # 打印训练完成的信息
    print("\n训练任务按计划完成，GPT-2.0模型文件'GPT2.pt'已经输出至当前工作目录。")
    
    plot_loss_curve()  # 训练完成后绘制损失曲线
    
    # 等待用户按下回车键后退出程序
    input('按回车键退出...')
    exit()
    