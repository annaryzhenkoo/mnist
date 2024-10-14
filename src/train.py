import torch
import torchvision.datasets
import torchvision.transforms as transforms
from models.model import FCNet
import matplotlib.pyplot as plt
import wandb

wandb.login(key='72e7d1c8328ef5e267224b9b5a27412b1791355e')
wandb.init(project="frst project")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(28 * 28)),
])

MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True, transform=transform)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False, transform=transform)

plt.imshow(MNIST_train.data[0])
plt.show()
print('Target: ', MNIST_train.targets[0])

val_split = int(len(MNIST_train) * 0.15)
train_split = int(len(MNIST_train) - val_split)

val_split, train_split = torch.utils.data.random_split(MNIST_train, [val_split, train_split])

train_loader = torch.utils.data.DataLoader(dataset=train_split, batch_size=100, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_split, batch_size=len(val_split), shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=MNIST_test, batch_size=len(MNIST_test), shuffle=False)

class train_loop:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, logger, loss):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.loss = loss

    def train(self):
        self.model.train()
        for batch_index, (image, label) in enumerate(self.train_loader):
            prediction = self.model.forward(image)
            loss_value = self.loss(prediction, label)
            loss_value.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.scheduler.step()
def accruracy(prediction, label):
    prediction = prediction.argmax(dim=1)
    return (prediction == label).sum() / len(label)

model = FCNet(35)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
loss = torch.nn.CrossEntropyLoss()

train_looop = train_loop(model, train_loader, val_loader,optimizer,scheduler,4, loss)
epoches =  10


for i in range(0, epoches):

    train_looop.train()
    all_pred = []
    all_label = []
    for batch_index, (image, label) in enumerate(test_loader):
        pred = model.forward(image)
        all_pred.append(pred)
        all_label.append(label)

    accuracy_train = accruracy(torch.cat(all_pred), torch.cat(all_label))
    wandb.log({'accuracy train': accuracy_train,  'learning_rate': optimizer.param_groups[0]['lr']})

    all_pred_val = []
    all_label_val = []
    for batch_index_val, (image_val, label_val) in enumerate(val_loader):
        pred_val = model.forward(image_val)
        all_pred_val.append(pred_val)
        all_label_val.append(label_val)
    accuracy_val = accruracy(torch.cat(all_pred_val), torch.cat(all_label_val))
    wandb.log({'accuracy val': accuracy_val})

    print('Epoch: ', i+1)
    print('Train accuracy: ', accuracy_train)
    print('Val accuracy: ', accuracy_val)

torch.save(model, 'models/smth.pth')

