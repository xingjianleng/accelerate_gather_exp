import torch
from accelerate import Accelerator

def main():
    accelerator = Accelerator()
    device = accelerator.device

    example = [
        [torch.randn(2, 3, 4, device=device), torch.randn(2, 3, 4, device=device)],
        [torch.randn(2, 3, 4, device=device), torch.randn(2, 3, 4, device=device)],
        [torch.randn(2, 3, 4, device=device), torch.randn(2, 3, 4, device=device)],
    ]
    gather_example = accelerator.gather(example)
    print(f"Before gather {len(example), len(example[0]), example[0][0].shape}")
    print(f"After gather {len(gather_example), len(gather_example[0]), gather_example[0][0].shape}")

if __name__ == "__main__":
    main()
