import numpy as np
from mindspore.dataset import vision
from mindspore.dataset import MnistDataset, GeneratorDataset
import matplotlib.pyplot as plt
from mindspore.dataset import transforms, vision, text
from mindspore import ops, Tensor, nn
import mindspore.context as context
import mindspore
import os
from loguru import logger
from tqdm import tqdm

depth, on_value, off_value = 10, Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32) # For label
train_dataset = MnistDataset("MNIST_Data/train", shuffle=True)
test_dataset = MnistDataset("MNIST_Data/test", shuffle=False)

print(type(train_dataset))
print(type(test_dataset))

class Config:

    epoch = 1000
    batchsize= 256
    num_workers = 8
    device = "cuda:3" 
    
config = Config()
# torch.manual_seed(42)

train_transforms = transforms.Compose(
    [
        # vision.Pad(3),
        # vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        # vision.RandomAffine(degrees=15),
        vision.HWC2CHW(),
    ]
)

test_transforms = transforms.Compose(
    [
        vision.HWC2CHW(),
    ]
)

#train_dataset = train_dataset.map(train_transforms, input_columns='image')
train_dataset = train_dataset.batch(batch_size=config.batchsize)
#test_dataset = test_dataset.map(test_transforms, 'image')
test_dataset = test_dataset.batch(batch_size=config.batchsize)

class epsNet:
    def __init__(self, xsize, ysize, asize, alpha, beta, Jsmall, eps, dist, step):
        self.xsize = xsize
        self.ysize = ysize
        self.asize = asize
        minval = Tensor(-0.1, mindspore.float32)
        maxval = Tensor(0.1, mindspore.float32)
        self.w1 = ops.uniform((ysize, xsize), minval, maxval)
        self.w2 = ops.uniform((asize, ysize), minval, maxval)
        self.b = ops.zeros(ysize, mindspore.float32)
        self.alpha = alpha
        self.beta = beta
        self.Jsmall = Jsmall
        self.eps = eps
        self.dist = dist
        self.setp = step
        self.loss = nn.CrossEntropyLoss()
        
    def neuronDTE(self, gamma, p):
        return (1-self.eps)*p + self.eps*ops.pow(ops.sin(p + gamma), 2)
    
    def neuronADE(self, gamma, p, q):
        p = (1 - self.eps)*p + self.eps*ops.pow(ops.sin(p + gamma), 2)
        q = (1 - self.eps)*q + self.eps*ops.sin(2*(p + gamma)*(1 + q))
        return [p, q]
    
    def cost(self, pred, label):
        J = self.loss(pred, label)
        return J
    
    def train(self, batch):
        batch_size = batch["image"].shape[0]
        y = ops.zeros((batch_size, self.ysize), mindspore.float32)
        p = ops.zeros((batch_size, self.ysize), mindspore.float32)
        q = ops.zeros((batch_size, self.ysize), mindspore.float32)
        num = 0
        while True:
            cnt = 0.0
            x = batch["image"].view(batch_size, -1)
            d = batch["label"].astype(mindspore.int32)
            
            # forward 
            gamma = ops.matmul(x, self.w1.T) + self.b
            y = self.neuronDTE(gamma, y)
            z = ops.matmul(y, self.w2.T)
            a = ops.softmax(z, axis=-1)
            
            cnt += (a.argmax(axis=1).astype(mindspore.int32) == d).sum()
            J = self.cost(a, d)
            
            # label = ops.one_hot(d, depth, on_value, off_value, axis=-1)
            deltaz = a - ops.one_hot(d, depth, on_value, off_value, axis=-1)
            deltaz /= batch_size
            deltav = self.eps*ops.sin(2 *(p+gamma)) * (ops.matmul(deltaz, self.w2))
            
            # Update
            self.w2 -= self.beta * ops.matmul(deltaz.T, y)
            self.w1 -= self.alpha * ops. matmul((deltav * (1+q)).T, x)
            self.b -= self.alpha * (deltav * (1+q)).sum(axis=0)
            
            p,q = self.neuronADE(gamma, p, q)
            batch_acc = cnt / batch_size
            num+=1
            # if num % 1 == 0:
                # print("Num: {}, J: {}".format(num, J))
                
            if J < self.Jsmall:
                break
                
        return batch_acc
    
    def test(self, batch):
        
        batch_size = batch["image"].shape[0]
        y = ops.zeros((batch_size, self.ysize), mindspore.float32)
        
        cnt = 0
        
        while True:
            x = batch["image"].view(batch_size, -1)
            
            gamma = ops.matmul(x, self.w1.T) + self.b
            y_ = self.neuronDTE(gamma, y)
            z = ops.matmul(y_, self.w2.T)
            a = ops.softmax(z, axis=1)
            
            cnt += 1
            if cnt >= 2:
                break
                
        return a.asnumpy()
    
    def save(self, path):
        return # TODO
    
    def load(self, path):
        return # TODO
    
    
net = epsNet(
    xsize=784, ysize=4096, asize=10, alpha=0.1, beta=0.1, Jsmall=2.3, eps=0.1, dist=0.1, step=1024
)

lr_milestone = [100, 200, 400]
lr_decay = 0.5

# ops.manual_seed(24)
suffix = "2023020101_epsNet"
ckpt_path = None

ckpt_folder = os.path.join("ckpts", suffix)
log_path = os.path.join("logs", "{}.log".format(suffix))

logger.add(log_path, level="INFO")

if ckpt_path is not None and os.path.exists(ckpt_path):
    net.load(ckpt_path)
    logger.info("Load checkpoint {}".format(ckpt_path))

if not os.path.exists(ckpt_folder):
    os.makedirs(ckpt_folder)

best_epoch, best_acc = -1, -1

for epoch_id in range(config.epoch):
    if (epoch_id+1) in lr_milestone:
        net.alpha = net.alpha*lr_decay
        net.beta = net.beta*lr_decay
        logger.info("Learning rate decay to: {} and {}".format(net.alpha, net.beta))
        
    ds_iter = train_dataset.create_dict_iterator(output_numpy=False, num_epochs=1) 
    batch_id = 1
    with tqdm(train_dataset.get_dataset_size()) as pbar:
        for batch in ds_iter:
            batch_acc = net.train(batch)
            pbar.update(1)
            # import pdb;pdb.set_trace()
            pbar.set_description("Epoch: {}, Batch: {}/{}, Train Acc: {:.5f}".format(epoch_id, batch_id, train_dataset.get_dataset_size(), batch_acc.asnumpy().item()))
            batch_id += 1
    
    ckpt_path = os.path.join(ckpt_folder, "epoch_{}.pth".format(str(epoch_id)))
    net.save(ckpt_path)

    total, correct = 0., 0.
    
    num_ = 0
    for batch in test_dataset.create_dict_iterator(output_numpy=False, num_epochs=1):
        a_pred = np.argmax(net.test(batch),axis=1)
        a_true = batch["label"].asnumpy()
        correct+=np.sum(a_pred==a_true)
        total+=a_true.shape[0]

    acc = correct/total
    if acc >= best_acc:
        best_acc = acc
        best_epoch = epoch_id
        logger.info("Epoch: %d, Test Acc improved to: %.5f"%(epoch_id, acc))
    else:
        logger.info("Epoch: %d, Test Acc is: %.5f, Best Test Acc is: %.5f in epoch: %d"%(epoch_id, acc, best_acc, best_epoch))
