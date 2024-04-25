# import torch
# import torchvision

# import os
# import shutil
# import sys
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# import torch_xla
# import torch_xla.debug.metrics as met
# import torch_xla.distributed.parallel_loader as pl
# import torch_xla.utils.utils as xu
# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.xla_multiprocessing as xmp
# import torch_xla.test.test_utils as test_utils

# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch_xla.distributed.xla_backend

# n_epochs = 3
# batch_size_train = 8 # 64
# batch_size_test = 10 # 1000
# learning_rate = 0.01
# momentum = 0.5
# log_interval = 10

# random_seed = 1
# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)

# ### load data
# test_loader = xu.SampleGenerator(
#     data=(torch.zeros(8, 1, 28,28), torch.zeros(8, dtype=torch.int64)),
#     sample_count=1000 // 8 // xm.xrt_world_size())

# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)

# print("shape: ", example_data.shape)

# ### build model
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x)


# class MNIST(nn.Module):

#   def __init__(self):
#     super(MNIST, self).__init__()
#     self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#     self.bn1 = nn.BatchNorm2d(10)
#     self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#     self.bn2 = nn.BatchNorm2d(20)
#     self.fc1 = nn.Linear(320, 50)
#     self.fc2 = nn.Linear(50, 10)

#   def forward(self, x):
#     x = F.relu(F.max_pool2d(self.conv1(x), 2))
#     x = self.bn1(x)
#     x = F.relu(F.max_pool2d(self.conv2(x), 2))
#     x = self.bn2(x)
#     x = torch.flatten(x, 1)
#     x = F.relu(self.fc1(x))
#     x = self.fc2(x)
#     return F.log_softmax(x, dim=1)

# device = xm.xla_device()
# network = MNIST().to(device)
# # network = Net().to(device)
# optimizer = optim.SGD(network.parameters(), lr=learning_rate,
#                       momentum=momentum)
# # loss_fn = nn.NLLLoss()

# train_losses = []
# train_counter = []
# test_losses = []
# test_counter = [i*20 for i in range(n_epochs + 1)]

# def test():
#   network.eval()
#   test_loss = 0
#   correct = 0
#   with torch.no_grad():
#     for data, target in test_loader:
#       output = network(data)
#       test_loss += F.nll_loss(output, target, size_average=False).item()
#       pred = output.data.max(1, keepdim=True)[1]
#       correct += pred.eq(target.data.view_as(pred)).sum()
#   test_loss /= 20
#   test_losses.append(test_loss)
#   print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#     test_loss, correct, 20,
#     100. * correct / 20))
#   return test_loss

# # run test model
# def test_mnist():
#   torch.manual_seed(1)

#   test()
#   # target fori_loop
#   for epoch in range(1, n_epochs + 1):
#     test()


# # torch.set_default_dtype(torch.float32)
# accuracy = test_mnist()



import args_parse
from torch_xla import runtime as xr

# MODEL_OPTS = {
#     '--ddp': {
#         'action': 'store_true',
#     },
#     '--pjrt_distributed': {
#         'action': 'store_true',
#     },
# }

FLAGS = args_parse.parse_common_options(
    datadir='/tmp/mnist-data',
    batch_size=128,
    momentum=0.5,
    lr=0.01,
    target_accuracy=98.0,
    num_epochs=18,
    # opts=MODEL_OPTS.items(),
)

import os
import shutil
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla.distributed.xla_backend

import torch_xla.experimental.fori_loop
from torch_xla.experimental.fori_loop import fori_loop


class MNIST(nn.Module):

  def __init__(self):
    super(MNIST, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.bn1 = nn.BatchNorm2d(10)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.bn2 = nn.BatchNorm2d(20)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = self.bn1(x)
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = self.bn2(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


def train_mnist(flags, **kwargs):
  torch.manual_seed(1)

  test_loader = xu.SampleGenerator(
      data=(torch.zeros(flags.batch_size, 1, 28, 28), torch.zeros(flags.batch_size, dtype=torch.int64)),
      sample_count=10000 // flags.batch_size // xm.xrt_world_size())

  # Scale learning rate to num cores
  lr = flags.lr * xm.xrt_world_size()
  device = xm.xla_device()
  model = MNIST().to(device)

  # Initialization is nondeterministic with multiple threads in PjRt.
  # Synchronize model parameters across replicas manually.
  if xr.using_pjrt():
    xm.broadcast_master_param(model)

  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(flags.logdir)
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=flags.momentum)
  loss_fn = nn.NLLLoss()

  def test_loop_fn(loader):
    total_samples = 0
    correct = 0
    model.eval()
    # print("loader: ", loader)
    # print("type loader: ", type(loader))
    for data, target in loader:
      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum()
      total_samples += data.size()[0]

    accuracy = 100.0 * correct.item() / total_samples
    accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
    return accuracy

  # train_device_loader = pl.MpDeviceLoader(train_loader, device)
  test_device_loader = pl.MpDeviceLoader(test_loader, device)
  accuracy, max_accuracy = 0.0, 0.0

  for epoch in range(1, flags.num_epochs + 1):
    # xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
    # train_loop_fn(train_device_loader, epoch)
    # xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
    accuracy = test_loop_fn(test_device_loader)
    xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(epoch, test_utils.now(), accuracy))
    max_accuracy = max(accuracy, max_accuracy)
    # test_utils.write_to_summary(writer, epoch, dict_to_write={'Accuracy/test': accuracy}, write_xla_metrics=True)
    # if flags.metrics_debug: xm.master_print(met.metrics_report())

  ### fori_loop
  # torch.set_grad_enabled(False)
  # new_test_device_loader = pl.MpDeviceLoader(test_loader, device)
  upper = torch.tensor([flags.num_epochs + 1], dtype=torch.int32, device=device) # flags.num_epochs + 1
  lower = torch.tensor([1], dtype=torch.int32, device=device) # 1
  init_val = torch.tensor([1], dtype=torch.int32, device=device)
  # l_in_0 = torch.randn(10, device=xm.xla_device()) # test_device_loader
  # linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())
  def body_fun(test_device_loader):
    res1 = torch.tensor([2], dtype=torch.int32, device=device)
    res2 = torch.tensor([2], dtype=torch.int32, device=device)
    res3 = res1 + res2
    return res3
#   def body_fun(test_device_loader):
#     accuracy = test_loop_fn(test_device_loader)
#     max_accuracy = max(accuracy, max_accuracy)
#     return max_accuracy

  upper_, lower_, one_value_, add_res_x_, l_in_i_plus_1_, weight_, bias_, l_out_ = fori_loop(
      upper, lower, body_fun, init_val, test_device_loader)

  test_utils.close_summary_writer(writer)
  xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
  return max_accuracy


# def _mp_fn(index, flags):
def main_fun(flags):
  torch.set_default_dtype(torch.float32)
  accuracy = train_mnist(flags)
  if flags.tidy and os.path.isdir(flags.datadir):
    shutil.rmtree(flags.datadir)
  if accuracy < flags.target_accuracy:
    print('Accuracy {} is below target {}'.format(accuracy,
                                                  flags.target_accuracy))
    sys.exit(21)


if __name__ == '__main__':
#   xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
#   _mp_fn()
  main_fun(FLAGS)
