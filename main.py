from accelerate import Accelerator
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.fc1(x)
        feat = x
        x = self.relu(x)
        act_feat_detached = x.detach()
        x = self.fc2(x)
        return x, feat, act_feat_detached


def main():
    accelerator = Accelerator()
    device = accelerator.device

    torch.manual_seed(42)
    input_size = 100
    output_size = 10
    num_samples = 10000
    n_epochs = 99999

    X = torch.randn(num_samples, input_size)
    y = torch.randn(num_samples, output_size)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # <-- Each GPU has 64 samples

    model = SimpleModel(input_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, feats, act_feats_detach = model(inputs)
            loss = criterion(outputs, labels.argmax(dim=1))

            # NOTE: Uncomment sections below to understand the behavior of the Accelerator

            # --------- loss -------- #
            # print(loss)
            # gathered_loss = accelerator.gather(loss)
            # print(gathered_loss)
            # --------- loss -------- #

            # --------- Feats w/ grad -------- #
            # print(feats.shape, feats.device, feats.grad_fn)
            # gather_feats = accelerator.gather(feats)
            # print(gather_feats.shape, gather_feats.device, gather_feats.grad_fn)
            # --------- Feats w/ grad -------- #

            # -------- Act Feats Detach --------- #
            # print(act_feats_detach.shape, act_feats_detach.device, act_feats_detach.grad_fn)
            # gather_act_feats_detach = accelerator.gather(act_feats_detach)
            # print(gather_act_feats_detach.shape, gather_act_feats_detach.device, gather_act_feats_detach.grad_fn)
            # -------- Act Feats Detach --------- #

            # -------- process id --------- #
            print(accelerator.process_index)
            gather_process_id = accelerator.gather([torch.tensor(accelerator.process_index, device=device)])
            print(gather_process_id)
            # -------- process id --------- #

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            time.sleep(0.5)  # <--- Make it slower to see the terminal output

        # print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(dataloader):.4f}")

    # After training, gather the model to a single device if needed
    model = accelerator.unwrap_model(model)

    # Example: Save the model (only on process 0 to avoid multiple writes)
    if accelerator.is_main_process:
        torch.save(model.state_dict(), f"simple_model_{time.strftime('%Y%m%d-%H%M%S')}.pth")


if __name__ == '__main__':
    main()
