import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from wherept import WherePT, WherePTConfig

EVAL_ITERS = 10
VOCAB_LEN = 61
SAVE_ALL_CHECHPOINTS = False
OUT_DIR = "checkpoints"

# Define sweep config
sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "batch_size": {"values": [8, 12, 16, 24]},
        "epochs": {"values": [1, 2]},
        "lr": {"max": 1e-3, "min": 1e-6},
        "n_embed": {"values": [16, 32, 64, 128, 256]},
        "n_head": {"values": [2, 4, 6, 8]},
        "n_layer": {"values": [1, 2, 4, 6]},
        "block_size": {"values": [4, 8, 16, 32, 48]},
        "dropout": {"max": 0.75, "min": 0.0},
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="woher")


def get_batch(split):
    if split == "train":
        df = train_df
    elif split == "val":
        df = val_df
    x = []
    y = []
    sample_idx = torch.randint(0, len(df), (wandb.config.batch_size,))
    for sidx in sample_idx:
        target_len = df.iloc[int(sidx)]["target_len"]
        idx = torch.randint(0, target_len - 1, (1,)).int()
        
        x_tensor = torch.tensor(df.iloc[int(sidx)]["tokenized"][idx:idx+wandb.config.block_size])
        y_tensor = torch.tensor(df.iloc[int(sidx)]["tokenized"][idx+1:idx+wandb.config.block_size+1])
        x.append(x_tensor)
        y.append(y_tensor)
        
    x = torch.stack(x)
    y = torch.stack(y)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    run = wandb.init()

    modelConfig = WherePTConfig(
        VOCAB_LEN,
        wandb.config.n_embed,
        wandb.config.n_head,
        wandb.config.n_layer,
        wandb.config.block_size,
        wandb.config.dropout)
    
    model = WherePT(modelConfig)
    
    run.log({"m_params": sum(p.numel() for p in model.parameters())/1e6})

    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)

    LOG_INTERVAL = 10
    wandb.watch(model, log_freq=LOG_INTERVAL)


    for epoch in range(wandb.config.epochs):
        for batch_idx in range(len(train_df) // wandb.config.batch_size):
            optimizer.zero_grad()
            xb, yb = get_batch("train", wandb.config.batch_size)
            logits, loss = model(xb, yb)
            loss.backward()
            optimizer.step()
            if batch_idx % LOG_INTERVAL == 0:
                losses = estimate_loss(model)
                run.log({"loss": losses})
                if losses['val'] < best_val_loss or SAVE_ALL_CHECHPOINTS:
                    best_val_loss = losses['val']
                    if batch_idx > 0:
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'config': wandb.config,
                            'epoch': epoch,
                            'batch_idx': batch_idx,
                            'best_val_loss': best_val_loss,
                        }
                        print(f"saving checkpoint to {OUT_DIR}")
                        torch.save(checkpoint, os.path.join(OUT_DIR, 'ckpt.pt'))


if __name__ == "__main__":
    dataset = wandb.Api().artifact("woher/wherept-data:latest")
    train_df = dataset.get("train_df").get_dataframe()
    val_df = dataset.get("val_df").get_dataframe()
    wandb.agent(sweep_id, function=main, count=4)