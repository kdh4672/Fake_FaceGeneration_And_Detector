import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision.models as models
import copy
import time
# from torch.utils.tensorboard import SummaryWriter
from adamp import AdamP
import numpy as np
import shutil
from torch.optim import lr_scheduler
from Fake_Dataset import FakeDataset
# Tensorboard : set path

# path = '../fake_detect/resnet50_phase_no_norm'
# try :
#     shutil.rmtree(path)
# except:
#     pass
# writer = SummaryWriter(path)

# 데이터를 한번에 batch_size만큼만 가져오는 dataloader를 만든다.
num_classes = 2
in_channel = 3
batch_size = 64
learning_rate = 0.001
num_epochs = 1000
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')




dir = '/home/capstone_ai1/kong/fft_processed' ## dataset directory (face alignment-phase extracted ) 
total_dataset = FakeDataset(root_dir=dir, normalize = False)

total_size = len(total_dataset)
train_size = int(np.ceil(total_size*0.99))
test_size = int(np.floor(total_size*0.01))
print(test_size)

train_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = models.resnet50()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
model.fc = nn.Linear(2048  , 2)
model.to(device)

criterion = nn.CrossEntropyLoss().to(device)

optimizer = AdamP([
    {'params': model.parameters(), 'weight_decay': 5e-6}
], lr=learning_rate)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma= 0.1)
if __name__ == '__main__':
    total_step = len(train_loader)
    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(train_loader):
            start = time.time()
            # Assign Tensors to Configured Device
            images = images.to(device)
            labels = labels.to(device)

            # Forward Propagation
            outputs = model(images)

            # Get Loss, Compute Gradient, Update Parameters
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# tesnorboard : loss 그래프 생성

#             if i % 10 == 9:
#                 writer.add_scalar('training loss',
#                             loss.item(),
#                             epoch * len(train_loader) + i)

                # writer.add_scalar('learning_rate',
                #             scheduler.get_last_lr()[0],
                #             epoch * len(train_loader) + i)

# ---------------------- #
            # Print Loss for Tracking Training
            if (i+1) % 1 == 0:
                # print("learning_rate:{:.8f}".format(scheduler.get_last_lr()[0]))
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                _, predicted = torch.max(outputs ,1)
                print('Testing data: [Predicted: {} / Real: {}]'.format(predicted, labels))
                print('learning rate: {}'.format(scheduler.get_last_lr()[0]))
            print("걸리는 시간: ", time.time()-start)


            if i % 100 == 99:
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    val_loss = criterion(outputs, labels)
#                     writer.add_scalar('val_loss',
#                                       val_loss.item(),
#                                       epoch * len(train_loader) + i)
#                     writer.add_scalar('learning_rate',
#                                       scheduler.get_last_lr()[0],
#                                       epoch * len(train_loader) + i)
#                     writer.add_scalar('accuracy',
#                     100 * correct / total,
#                     epoch * len(train_loader) + i)
                    print("===========================================================")
                    print('Accuracy of the network on the {} test images:\
                    {} %'.format(len(test_loader)*batch_size, 100 * correct / total))
                    print("===========================================================")
                model.train()
            if i % 1000 == 999:
                torch.save(model.state_dict(), path + '_{}_{}.pth'.format(epoch,i))
            if (epoch * len(train_loader) + i)%4500 ==4499:
                scheduler.step()
        torch.save(model.state_dict(), path+'.pth')
