# Some part of the code was referenced from below.
# https://github.com/pytorch/examples/tree/master/word_language_model 
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus


tmp=torch.ones(5)
print(tmp)  #tensor([1., 1., 1., 1., 1.])  结果返回这个.

tmp=tmp.exp()
print(tmp)

prob = torch.ones(20)
# 下面一行进行多项式分布抽样.打到随机的目的,抽取一个.
print(torch.multinomial(prob, num_samples=1))
input = torch.multinomial(prob, num_samples=1).unsqueeze(1)  #unsqueeze 给指定位置加上维数为一的维度


print(input)

'''

2里面的网络都是最重要的,需要多跑几次,多熟悉这些网络的写法!!!!!!!!!!!!








语言模型这里面用的是单向lstm.



'''

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 5
num_samples = 1000     # number of words to be sampled
batch_size = 20
seq_length = 30
learning_rate = 0.002

# Load "Penn Treebank" dataset
corpus = Corpus()
ids = corpus.get_data('data/train.txt', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length


# RNN based language model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()#第一行还是经典废话
        self.embed = nn.Embedding(vocab_size, embed_size)

        '''
        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding = nn.Embedding(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])  #词表是0到9之间的整数 
        >>> embedding(input)   结果就是从 2,4  变到 2,4,3   达到了嵌入的目的.底层怎么实现的?
        tensor([[[-0.0251, -1.6902,  0.7172],
                 [-0.6431,  0.0748,  0.6969],
                 [ 1.4970,  1.3448, -0.9685],
                 [-0.3677, -2.7265, -0.1685]],

                [[ 1.4970,  1.3448, -0.9685],
                 [ 0.4362, -0.4004,  0.9400],
                 [-0.6431,  0.0748,  0.6969],
        '''
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)
        
        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)
        
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        
        # Decode hidden states of all time steps
        #这一次使用了所有的信心都扔给linear
        out = self.linear(out)
        return out, (h, c)     #这样out输出的是一个序列.所以结果也就是seq 2 seq

model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states] 

# Train the model
for epoch in range(num_epochs):
    # Set initial hidden and cell states
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))
    
    for i in range(0, ids.size(1) - seq_length, seq_length):
        # Get mini-batch inputs and targets
        inputs = ids[:, i:i+seq_length].to(device)   #输入一堆句子
        targets = ids[:, (i+1):(i+1)+seq_length].to(device)   #目标是句子的下一个单词偏移一下
        #这样就让模型学习给一个句子预测下一个单词的能力!!!!!!!!
        
        # Forward pass
        #运行过detach的变量就不会再求导数了.
        states = detach(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))
        
        # Backward and optimize
        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        step = (i+1) // seq_length
        if step % 100 == 0:
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                   .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

# Test the model
'''
已经训练好了,下面用这个会写作文的,写一些话
'''


with torch.no_grad():
    with open('sample.txt', 'w') as f:
        # Set intial hidden ane cell states
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                 torch.zeros(num_layers, 1, hidden_size).to(device))

        # Select one word id randomly
        '''
        首先随机给他一下开始的话,让他预测后面的话,然后开始写
        '''
        prob = torch.ones(vocab_size)
        #下面一行进行多项式分布抽样.打到随机的目的,抽取一个.
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device) #所以squense的长度就是1.
        #所以虽然是seq2seq的训练但是只是看一个长度.


        for i in range(num_samples):
            # Forward propagate RNN 
            output, state = model(input, state)
            #output是预测的seq
            # Sample a word id
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            # Fill input with sampled word id for the next time step
            input.fill_(word_id)  #用word_id 赋值给input,那么这个input就是下一个预测单词的输入.全部填充成word_id

            # File write
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i+1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i+1, num_samples, 'sample.txt'))

# Save the model checkpoints
torch.save(model.state_dict(), 'model.ckpt')