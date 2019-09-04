import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device,"这次可以开心跑代码了.直接调用服务器gpu,告别vim编程,并且点进去看源码看的都是服务器端安装的"
             "源码,配合远程挂载功能可以实现通过跳板机同步不同网段的代码,节省自己电脑的性能,感觉就跟用一个带图形界面的ubuntu一样了")
# Hyper-parameters
sequence_length = 28 #时间长短,也就是图片的行数
input_size = 28      #表示一个时间点,需要的特征,表示的就是图片28+28的列数
hidden_size = 128   #隐藏层神经元的数量
num_layers = 2    #隐藏层的层数.
num_classes = 10 #分类的数量
batch_size = 100  #batch

num_epochs = 2
learning_rate = 0.003

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)



'''
跟以前一样
'''




# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()#初始化函数的第一行还是一样
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        #pytorch的全连接的写法是Linear.
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        #forward 的第一步是上来写初始化
        # Set initial states  表示lstm计算的初始值.给定0或者很小的数即可.
        '''

        通过下面的例子,看lstm书写中的各个参数的设置.
      rnn = nn.LSTM(10, 20, 2)       input_size, hidden_size, num_layers
        input = torch.randn(5, 3, 10)
        h0 = torch.randn(2, 3, 20)
       c0 = torch.randn(2, 3, 20)
       output, (hn, cn) = rnn(input, (h0, c0))

        看了网上的教程,挺多错误,这里面统一跑代码实践一下.
        '''
        print(x.size(),"x.size是!!!!!!!!!")  #结果是100,28,28   x.size(0)是batch_size
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        print(h0.size(),"h0.size")       #torch.Size([4, 100, 128]) h0.size
        print(c0.size(),"c0.size")       #torch.Size([4, 100, 128]) c0.size

        #因为是blstm所以输入
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        print(out.size(),"out.size")       #torch.Size([100, 28, 256]) out.size  : batch ,seq ,hiddent*num_direction
        print(out[:, -1, :].size())  #torch.Size([100, 256])

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])      #使用最后一个时间,也就是seq最后一个即可,seq表示的是时间.
        #对应到这个问题就是手写图片的最后一行的信息,因为用的是lstm,之前27行的信息都会用lstm长短时记忆来
        #对最后一行赵成影响.只使用最后一行信息就够了.

        #fc的输入是100,256 输出 100,10
        return out

model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss() #多分类就用这个交叉熵.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)  #images:100,28,28
        loss = criterion(outputs, labels)
        #神经网络一定要对每一层的shape理解彻底!
        #outputs:100, 10
        #labels: 100, 10
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        #代码完全一样,只是把train 前面加上with torch.no_grad():
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint ,最下面的代码一定要每次跑完都写上.
torch.save(model.state_dict(), 'model.ckpt')