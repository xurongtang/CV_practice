import torch,random
import copy
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset_load import get_CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm

random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)

class Custom_dataset(torch.utils.data.Dataset):
    def __init__(self,dataset):
        self.input_dataset,self.label_dataset = dataset
        self.one_hot_code_flag = True
        self.num_classes = 10

    def __getitem__(self,index):
        input_img = self.input_dataset[index]
        if self.one_hot_code_flag:
            label_img = self.one_hot_code(self.label_dataset[index],self.num_classes)
        else:
            label_img = self.label_dataset[index]
        return input_img,label_img
    
    def __len__(self):
        length = self.input_dataset.shape[0]
        assert length == self.label_dataset.shape[0]
        return length

    @staticmethod
    def one_hot_code(input_data: torch.Tensor,num_classes) -> torch.Tensor:
        one_hot = np.zeros((num_classes),dtype=np.int8)
        for i in range(num_classes):
            if input_data == i:
                one_hot[i] = 1
                break
        one_hot = torch.Tensor(one_hot)
        return one_hot

# 定义模型
class CNN_to_digital_number_recognize(nn.Module):
    
    def __init__(self,input_dim):
        super().__init__()
        self.input_dim = input_dim
        ksize1 = 5
        ksize2 = 5
        padding1 = 0    # ksize1//2
        padding2 = 0    # ksize2//2
        final_channels = 20
        pooling_size = 2
        # 如果想保持图像大小，padding = kernel_size//2
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=10,kernel_size=ksize1,padding=padding1)
        self.pool1 = nn.MaxPool2d(kernel_size=pooling_size)
        self.conv2 = nn.Conv2d(in_channels=10,out_channels=final_channels,kernel_size=ksize2,padding=padding2)
        self.pool2 = nn.MaxPool2d(kernel_size=pooling_size)
        # 计算最后的元素总数
        final_elements = final_channels * ((((input_dim[1]-(ksize1//2)*2) // pooling_size) - (ksize2//2)*2) // pooling_size)**2
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(final_elements,512)
        self.linear2 = nn.Linear(512,10)
        # 构建
        self.ConvBlock1 = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.pool1)
        self.ConvBlock2 = nn.Sequential(
            self.conv2,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.pool2)
        self.LinearBlock = nn.Sequential(
            self.flatten,
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.Softmax(dim=1))

    def forward(self,x):
        # 卷积输入的数据维度为 (Batch,channel,height,weight)
        x = x.view(-1,self.input_dim[0],self.input_dim[1],self.input_dim[2]).to(torch.float)
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.LinearBlock(x)
        # print(x.shape)
        return x

def training_process(train_set,model,params_dict: dict):
    ## 参数设置 ##
    BATCHSIZE = params_dict['batchsize']
    LEARNING_RATE = params_dict['learning_rate']
    EPOCHS = params_dict['epochs']
    device = torch.device(params_dict['device'])
    val_rate = 0.15
    ## 模型准备 ## 
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()  # 对于分类问题
    ## 数据准备 ##
    length = train_set[0].shape[0]
    train_dataset = Custom_dataset((train_set[0][:int(length*(1-val_rate))],train_set[1][:int(length*(1-val_rate))]))
    train_loader = DataLoader(train_dataset,batch_size=BATCHSIZE,shuffle=True)
    val_dataset = Custom_dataset((train_set[0][:int(length*val_rate)],train_set[1][:int(length*val_rate)]))
    val_loader = DataLoader(val_dataset,batch_size=BATCHSIZE,shuffle=True)
    ## 定义模型验证/测试函数 ##
    def val(model,val_loader):
        val_loss = []
        model.eval()
        for _,(batch_x,batch_y) in enumerate(val_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_out = model(batch_x)
            # 计算误差
            loss = loss_func(batch_out,batch_y)
            val_loss.append(loss.item())
        model.train()
        return np.mean(val_loss)
    ## 记录训练过程 ##
    train_loss_ls = []
    val_loss_ls = []
    early_stopping_count = 0
    ## 开始训练 ## 
    print("Start training...")
    for epo in tqdm(range(EPOCHS),colour='yellow'):
        batch_loss = []
        for _,(batch_x,batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_out = model(batch_x)
            # 计算误差
            loss = loss_func(batch_out,batch_y)
            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            if params_dict['device'] == 'cuda':
                torch.cuda.empty_cache()
        train_loss_ls.append(np.mean(batch_loss))
        val_loss_ls.append(val(model,val_loader))
        if epo == 0:
            best_val_loss = val_loss_ls[-1]
        # print(f"Iters {epo+1:>3}/{EPOCHS} is end, Train_loss:{train_loss_ls[-1]:>8.4f}, Valid_loss:{val_loss_ls[-1]:>8.4f}.")
        if val_loss_ls[-1] <= best_val_loss:
            best_val_loss = val_loss_ls[-1]
            best_model = copy.deepcopy(model)
            early_stopping_count = 0
            # print("Best model has been saved.")
        else:
            early_stopping_count += 1
        if early_stopping_count >= 5:
            print("Early stopping by 5 epochs...")
            break
    print("Training finished.")
    return best_model.eval()

def main():
    train_set,test_set = get_CIFAR10()
    print("train dataset size: ",train_set[0].shape)
    print("test dataset size: ",test_set[0].shape)
    # print(np.unique(np.array(train_set[1]),return_counts=True))
    Model = CNN_to_digital_number_recognize(input_dim=(3,32,32))
    trained_model = training_process(train_set,
                                     Model,
                                     params_dict={
                                            'batchsize':32,
                                            'learning_rate':0.001,
                                            'epochs':100,
                                            'device':'cuda'
                                            })
    # 简单测试，计算预测准确率
    count = 0
    trained_model.cpu()
    test_input_data,test_label_data = test_set
    print("Testing...")
    for i,input_test in enumerate(tqdm(test_input_data,colour='yellow')):
        res = trained_model(input_test.unsqueeze(0))
        pred_label = torch.argmax(res).item()
        if pred_label == test_label_data[i]:
            count += 1
    print(f"Accuracy:{(count/len(test_input_data))*100:>6.2f}%")
    
if __name__ == '__main__':
    main()
