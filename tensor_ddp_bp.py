import torch
import torch.nn as nn
from accelerate import Accelerator

def main():
    accelerator = Accelerator()
    device = accelerator.device

    example = torch.randn(2, 3, 4, device=device, requires_grad=True)
    gt_all_devices = torch.ones(8, 3, 4, device=device)
    loss_fn = nn.MSELoss()

    print(f"Before gather: {example.grad, example.device}")

    # NOTE: Uncomment the following code blocks to see the difference in behavior

    # -------------------- #
    # Try whether gradient can pass through gathered matrix and perform backward
    # NOTE: This won't work
    # gather_example = accelerator.gather(example) # [8, 3, 4]
    # -------------------- #

    # -------------------- #
    # If we detach before gather, and substitute part the part of the gathered tensor with the tensor on our GPU
    # NOTE: This works
    gather_example = accelerator.gather(example.detach())  # [8, 3, 4]
    curr_id = accelerator.process_index
    start_id = curr_id * 2
    end_id = start_id + 2
    gather_example[start_id:end_id] -= example.detach()
    gather_example[start_id:end_id] += example
    # -------------------- #

    # Compute the loss and try to perform backward
    loss = loss_fn(gather_example, gt_all_devices)
    accelerator.backward(loss)
    print(f"After gather: {example.grad, example.device}")

if __name__ == "__main__":
    main()
