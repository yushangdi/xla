import os
import unittest
from typing import Callable, Dict, List

import torch
import torch_xla
# # from torch._higher_order_ops.while_loop import while_loop, while_loop_dense
# from torch_xla.experimental.fori_loop import while_loop
import torch_xla.experimental.fori_loop
from torch._higher_order_ops.while_loop import while_loop
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb

# _TORCH_WHILE_LOOP_OPS = [
#     torch._higher_order_ops.while_loop,
# ]

def _fake_while_loop(cond_fn, body_fn, operands):
  while cond_fn(*operands):
    operands = body_fn(*operands)
  return operands

class WhileLoopTest(unittest.TestCase):

  def test_while_loop_tpu(self):
    def cond_fn(x):
      return x.sum() <= 10

    def body_fn(x):
      return (x + 1,)

    device = xm.xla_device()
    x = torch.ones(1, dtype=torch.int, device=device)
    res = while_loop(cond_fn, body_fn, (x, ))
    print("while_loop result: ", res)
    expected = torch.tensor(11, dtype=torch.int, device=device)
    print("expected result: ", expected)
    self.assertEqual(expected, res[0])


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
