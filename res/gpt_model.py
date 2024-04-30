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

# 无限循环，用于让用户选择计算设备
while True:
    # 获取用户输入的运算方式
    use_gpu = input('请选择运算方式("GPU"或"CPU"):')
    # 检查输入是否为"GPU"或"CPU"，如果不是，则提示错误
    if use_gpu not in ["GPU", "CPU"]:
        print("输入错误！\n")
    else:
        # 如果用户选择GPU并且CUDA可用，则继续
        if use_gpu == 'GPU' and not torch.cuda.is_available():
            print("\nCUDA 在您的 GPU 上不可用,请选择CPU运算。")
            # 等待用户按下回车键后退出程序
            input('按回车键退出...')
            # 退出程序
            exit()
        # 如果输入正确或CUDA不可用，退出循环
        print()
        break

# 尝试加载词汇表文件
try:
    # 使用json模块加载词汇表文件
    dict_datas = json.load(open('dict_datas.json', 'r'))
except FileNotFoundError:
    # 如果文件不存在，打印错误信息并退出程序
    print("无法找到词汇表文件'dict_datas.json'，请检查文件是否存在。")
    # 等待用户按下回车键后退出程序
    input('按回车键退出...')
    # 退出程序
    exit()
  
# 根据用户的选择确定使用GPU还是CPU
if use_gpu == 'GPU':
    # 如果使用GPU，创建一个指向CUDA设备的torch设备对象
    device = torch.device("cuda")
else:
    # 如果使用CPU，创建一个指向CPU的torch设备对象
    device = torch.device("cpu")
  
# 从加载的词汇表中获取单词到ID的映射和ID到单词的映射
word2id, id2word = dict_datas['word2id'], dict_datas['id2word']
# 获取词汇表的大小
vocab_size = len(word2id)
# 设置最大位置编码值
max_pos = 1800
# 设置嵌入层的大小
d_model = 768  # Embedding Size
# 设置前馈网络的维度
d_ff = 2048  # FeedForward dimension
# 设置K(=Q)和V的维度
d_k = d_v = 64
# 设置编码器和解码器层的数量
n_layers = 6
# 设置多头注意力中头的数量
n_heads = 8
# 设置梯度裁剪的阈值
CLIP = 1

# 定义make_data函数，用于预处理数据
def make_data(datas):
    # 初始化一个空列表，用于存储处理后的训练数据
    train_datas = []
    # 遍历输入数据列表中的每一个数据项
    for data in datas:
        # 去除数据项两端的空白字符，并替换制表符为特殊分隔符"<sep>"
        train_data = [i if i != '\t' else "<sep>" for i in data] + ['<sep>']
        # 将处理后的数据项添加到train_datas列表中
        train_datas.append(train_data)
    # 返回处理后的训练数据列表
    return train_datas

# 定义MyDataSet类，继承自PyTorch的Data.Dataset基类
class MyDataSet(Data.Dataset):
    # 构造函数，初始化数据集实例
    def __init__(self, datas):
        # 将传入的数据列表赋值给实例变量self.datas
        self.datas = datas

    # 重写getitem方法，它定义了如何获取数据集中的单个样本
    def __getitem__(self, item):
        # 根据索引item从数据列表中获取单个数据样本
        data = self.datas[item]
        # 创建解码器的输入序列和目标输出序列
        decoder_input = data[:-1]
        decoder_output = data[1:]
        # 计算解码器输入和输出序列的长度
        decoder_input_len = len(decoder_input)
        decoder_output_len = len(decoder_output)
        # 返回一个字典，包含解码器的输入序列、输入长度、输出序列和输出长度
        return {
            "decoder_input": decoder_input,
            "decoder_input_len": decoder_input_len,
            "decoder_output": decoder_output,
            "decoder_output_len": decoder_output_len
        }

    # 重写__len__方法，返回数据集中样本的总数
    def __len__(self):
        # 返回self.datas列表的长度
        return len(self.datas)

    # 定义padding_batch方法，用于对一个批次的数据进行填充
    def padding_batch(self, batch):
        # 提取批次中每个样本的解码器输入和输出的长度
        decoder_input_lens = [d["decoder_input_len"] for d in batch]
        decoder_output_lens = [d["decoder_output_len"] for d in batch]
        # 找出批次中最长的解码器输入和输出的长度
        decoder_input_maxlen = max(decoder_input_lens)
        decoder_output_maxlen = max(decoder_output_lens)
        # 对批次中的每个样本进行填充，使用特殊的"<pad>"标记
        for d in batch:
            # 如果解码器输入序列长度小于最大长度，则在序列末尾添加填充
            d["decoder_input"].extend([word2id["<pad>"]] * (decoder_input_maxlen - d["decoder_input_len"]))
            # 如果解码器输出序列长度小于最大长度，则在序列末尾添加填充
            d["decoder_output"].extend([word2id["<pad>"]] * (decoder_output_maxlen - d["decoder_output_len"]))
        # 将填充后的解码器输入序列和输出序列转换为PyTorch张量
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch], dtype=torch.long)
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch], dtype=torch.long)
        # 返回填充后的输入和输出张量
        return decoder_inputs, decoder_outputs

# 定义一个函数，用于获取注意力机制中的填充掩码（pad mask）
def get_attn_pad_mask(seq_q, seq_k):
    # 获取输入序列的批次大小和长度
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # 创建一个与seq_k同形状的张量，其中填充位置为True
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k]
    # 扩展张量以匹配Q和K的形状，并返回
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

# 定义生成子序列掩码的函数
def get_attn_subsequence_mask(seq):
    '''
    生成一个用于注意力机制中的子序列掩码，用于防止未来的信息流入。

    参数seq: [batch_size, tgt_len]，表示批次大小和目标序列的长度。
    '''

    # 获取seq的维度信息，用于构建注意力掩码的形状
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]

    # 创建一个上三角矩阵，其中的元素为1，用于表示合法的注意力关系
    # k=1表示从第2行开始为0，即不允许当前位置关注到未来的位置
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)

    # 将numpy数组转换为PyTorch张量，并转换为byte类型
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()

    # 根据是否使用GPU来决定掩码的运行设备
    if use_gpu == 'GPU':
        # 如果使用GPU，则将掩码转移到GPU上
        subsequence_mask = subsequence_mask.to(device)
    else:
        # 如果不使用GPU，则将掩码转移到CPU上
        subsequence_mask = subsequence_mask.cpu()

    # 返回生成的子序列掩码
    return subsequence_mask  # 返回的形状为[batch_size, tgt_len, tgt_len]

# 定义缩放点积注意力类
class ScaledDotProductAttention(nn.Module):
    # 类的初始化方法
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()  # 调用父类的初始化方法

    # 类的前向传播方法
    def forward(self, Q, K, V, attn_mask):
        '''
        实现了缩放点积注意力机制。

        参数Q: [batch_size, n_heads, len_q, d_k]，表示查询张量
        参数K: [batch_size, n_heads, len_k, d_k]，表示键张量
        参数V: [batch_size, n_heads, len_v(=len_k), d_v]，表示值张量
        参数attn_mask: [batch_size, n_heads, seq_len, seq_len]，表示注意力掩码
        '''

        # 计算查询和键之间的点积，然后缩放，得到注意力分数
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        # 将注意力掩码应用到分数上，掩码位置的分数被设置为一个非常小的负数
        # 这样在经过softmax后，这些位置的权重接近于0
        scores.masked_fill_(attn_mask, -1e9)

        # 应用softmax函数，将分数转换为概率分布
        attn = nn.Softmax(dim=-1)(scores)

        # 计算上下文向量，即注意力权重和值的乘积
        context = torch.matmul(attn, V)

        # 返回上下文向量和注意力权重
        return context, attn

# 定义多头注意力模块
class MultiHeadAttention(nn.Module):
    # 类的初始化方法
    def __init__(self):
        super(MultiHeadAttention, self).__init__()  # 调用父类的初始化方法

        # 定义线性变换层，用于将输入投影到不同的子空间
        # d_model 是输入特征的维度，n_heads 是注意力头的数量，d_k 是每个头的维度
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)

        # 定义输出线性层，将多头注意力的输出合并回原始维度
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        # 定义层归一化模块，用于稳定训练过程
        self.layernorm = nn.LayerNorm(d_model)

    # 类的前向传播方法
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]，查询张量
        input_K: [batch_size, len_k, d_model]，键张量
        input_V: [batch_size, len_v(=len_k), d_model]，值张量
        attn_mask: [batch_size, seq_len, seq_len]，注意力掩码
        '''

        # 保存输入的残差，用于后续的残差连接
        residual = input_Q
        # 获取批次大小
        batch_size = input_Q.size(0)

        # 对输入进行线性变换并分割成多头
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # 调整注意力掩码的形状以匹配多头注意力的维度
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # 调用缩放点积注意力函数，计算上下文向量和注意力权重
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)

        # 调整上下文向量的形状，准备进行合并
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]

        # 将多头注意力的输出通过输出线性层，并与残差相加
        output = self.fc(context)  # [batch_size, len_q, d_model]

        # 应用层归一化，并返回最终的输出和注意力权重
        return self.layernorm(output + residual), attn

# 定义位置感知的前馈网络类
class PoswiseFeedForwardNet(nn.Module):
    # 类的初始化方法
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()  # 调用父类的初始化方法

        # 定义前馈网络的两个线性层，中间通过ReLU激活函数
        # d_model 是输入和输出特征的维度，d_ff 是前馈网络内部特征的维度
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

        # 定义层归一化模块，用于稳定训练过程
        self.layernorm = nn.LayerNorm(d_model)

    # 类的前向传播方法
    def forward(self, inputs):
        '''
        输入 inputs: [batch_size, seq_len, d_model]，表示输入序列的特征
        '''
        # 保存输入的残差，用于后续的残差连接
        residual = inputs
        # 通过前馈网络
        output = self.fc(inputs)
        # 应用层归一化，并返回最终的输出
        return self.layernorm(output + residual)  # 返回的形状为 [batch_size, seq_len, d_model]

# 定义解码器层类
class DecoderLayer(nn.Module):
    # 类的初始化方法
    def __init__(self):
        super(DecoderLayer, self).__init__()  # 调用父类的初始化方法

        # 定义解码器层中的自注意力机制和编码器-解码器注意力机制
        self.dec_self_attn = MultiHeadAttention()  # 自注意力层
        self.dec_enc_attn = MultiHeadAttention()  # 编码器-解码器注意力层
        self.pos_ffn = PoswiseFeedForwardNet()  # 位置感知的前馈网络

    # 类的前向传播方法
    def forward(self, dec_inputs, dec_self_attn_mask):
        '''
        输入 dec_inputs: [batch_size, tgt_len, d_model]，表示解码器的输入序列特征
        输入 dec_self_attn_mask: [batch_size, tgt_len, tgt_len]，表示自注意力掩码
        '''
        # 通过自注意力层，得到解码器的输出和自注意力权重
        # dec_outputs: [batch_size, tgt_len, d_model]，dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        # 将自注意力层的输出传递给前馈网络
        dec_outputs = self.pos_ffn(dec_outputs)  # dec_outputs: [batch_size, tgt_len, d_model]

        # 返回解码器层的输出和自注意力权重
        return dec_outputs, dec_self_attn

# 定义解码器类
class Decoder(nn.Module):
    # 类的初始化方法
    def __init__(self):
        super(Decoder, self).__init__()  # 调用父类的初始化方法

        # 定义目标词汇表的嵌入层，将词汇表索引映射到d_model维的特征向量
        self.tgt_emb = nn.Embedding(vocab_size, d_model)

        # 定义位置编码的嵌入层，将位置索引映射到d_model维的特征向量
        # max_pos是位置编码的最大索引
        self.pos_emb = nn.Embedding(max_pos, d_model)

        # 定义一个模块列表，包含n_layers个DecoderLayer，每个DecoderLayer代表一个解码器层
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    # 类的前向传播方法
    def forward(self, dec_inputs):
        '''
        输入 dec_inputs: [batch_size, tgt_len]，表示解码器的输入序列，包括批次大小和目标长度
        '''
        seq_len = dec_inputs.size(1)  # 获取输入序列的长度

        # 根据是否使用GPU来生成位置编码
        if use_gpu == 'GPU':
            pos = torch.arange(seq_len, dtype=torch.long, device=device)  # 使用GPU运算
        else:
            pos = torch.arange(seq_len, dtype=torch.long).cpu()  # 使用CPU运算

        # 将一维的位置编码扩展为与输入序列相同的形状
        pos = pos.unsqueeze(0).expand_as(dec_inputs)  # [seq_len] -> [batch_size, seq_len]

        # 计算目标词汇表嵌入和位置编码的和，得到解码器的初始输出
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos)  # [batch_size, tgt_len, d_model]

        # 获取解码器自注意力的掩码，包括填充掩码和子序列掩码
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)  # [batch_size, tgt_len, tgt_len]
        # 合并两种掩码，得到最终的自注意力掩码
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)  # [batch_size, tgt_len, tgt_len]

        # 初始化一个列表，用于存储每一层的自注意力权重
        dec_self_attns = []

        # 遍历所有解码器层
        for layer in self.layers:
            # 将解码器的输出和自注意力掩码传递给当前层
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            # 将当前层的自注意力权重添加到列表中
            dec_self_attns.append(dec_self_attn)

        # 返回解码器的最终输出和每一层的自注意力权重
        return dec_outputs, dec_self_attns

# 定义GPT模型类
class GPT(nn.Module):
    # 类的初始化方法
    def __init__(self):
        super(GPT, self).__init__()  # 调用父类的初始化方法

        # 定义解码器组件
        self.decoder = Decoder()  # 使用前面定义的Decoder类
        # 定义输出层，将解码器的输出映射到词汇表的大小
        self.projection = nn.Linear(d_model, vocab_size)  # d_model是嵌入维度，vocab_size是词汇表大小

    # 类的前向传播方法
    def forward(self, dec_inputs):
        """
        输入 dec_inputs: [batch_size, tgt_len]，表示解码器的输入序列，包括批次大小和目标长度
        """
        # 通过解码器生成输出和自注意力权重
        dec_outputs, dec_self_attns = self.decoder(dec_inputs)  # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len]
        # 通过投影层生成logits，即每个词汇表索引的概率
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        # 返回logits的展平版本和自注意力权重
        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns

    # 定义一个贪婪解码方法
    def greedy_decoder(self, dec_input):
        # 初始化结束标志
        terminal = False
        # 记录解码开始的长度
        start_dec_len = len(dec_input[0])
        # 循环生成下一个单词，直到遇到"<sep>"分隔符或达到最大长度限制
        while not terminal:
            # 如果序列长度超过限制，则强制结束并添加"<sep>"分隔符
            if len(dec_input[0]) - start_dec_len > 100:
                next_symbol = word2id['<sep>']
                dec_input = torch.cat(
                    [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break
            # 通过解码器生成输出
            dec_outputs, _ = self.decoder(dec_input)
            # 通过投影层生成logits
            projected = self.projection(dec_outputs)
            # 获取最可能的下一个单词的索引
            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
            next_word = prob.data[-1]
            next_symbol = next_word
            # 如果生成的单词是"<sep>"，则结束循环
            if next_symbol == word2id["<sep>"]:
                terminal = True
            # 将生成的单词添加到输入序列中
            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
        # 返回生成的完整序列
        return dec_input

    # 定义一个生成回答的方法
    def answer(self, sentence):
        # 将句子中的"\t"替换为"<sep>"分隔符，并准备输入张量
        dec_input = [word2id.get(word, 1) if word != '\t' else word2id['<sep>'] for word in sentence]
        dec_input = torch.tensor(dec_input, dtype=torch.long, device=device).unsqueeze(0)

        # 使用贪婪解码方法生成输出序列
        output = self.greedy_decoder(dec_input).squeeze(0)
        # 将输出序列的索引转换为单词
        out = [id2word[int(id)] for id in output]
        # 找到"<sep>"分隔符的索引
        sep_indexs = []
        for i in range(len(out)):
            if out[i] == "<sep>":
                sep_indexs.append(i)
        # 提取两个"<sep>"之间的内容作为回答
        answer = out[sep_indexs[-2] + 1:-1]
        # 将回答的单词列表转换为字符串
        answer = "".join(answer)
        # 返回生成的回答
        return answer
        