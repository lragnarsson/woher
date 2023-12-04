import os
import pandas as pd
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from wherept import WherePT, WherePTConfig
from dataset import START_TOKEN, decode

EVAL_ITERS = 10
VOCAB_LEN = 61
SAVE_ALL_CHECHPOINTS = False
OUT_DIR = "checkpoints"
train_df = None
val_df = None

# Define sweep config
sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "batch_size": {"values": [8, 12, 16]},
        "epochs": {"values": [4]},
        "lr": {"max": 5e-3, "min": 1e-6},
        "n_embed": {"values": [8, 16, 32, 64, 128, 256]},
        "n_head": {"values": [1, 2, 4, 8]},
        "n_layer": {"values": [1, 2, 4, 6]},
        "block_size": {"values": [4, 8, 16, 32, 48]},
        "dropout": {"max": 0.1, "min": 0.0},
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="woher")
#sweep_id = "ocuds3m5"

def get_batch(split):
    if split == "train":
        df = train_df
    elif split == "val":
        df = val_df
    x = []
    y = []
    batch = torch.randint(0, len(df), (wandb.config.batch_size,))
    for sample in batch:
        target_len = df.iloc[int(sample)]["target_len"]
        max_idx = VOCAB_LEN - wandb.config.block_size - 3
        idx = torch.randint(0, min(target_len - 1, max_idx), (1,)).int()
        
        x_tensor = torch.tensor(df.iloc[int(sample)]["tokenized"][idx:idx+wandb.config.block_size])
        y_tensor = torch.tensor(df.iloc[int(sample)]["tokenized"][idx+1:idx+wandb.config.block_size+1])
        if x_tensor.shape[0] < wandb.config.block_size or y_tensor.shape[0] < wandb.config.block_size:
            print(df.iloc[int(sample)], x_tensor.shape, y_tensor.shape)
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

def log_examples(model, run):
    examples = [decode(model.generate(torch.tensor([START_TOKEN]).unsqueeze(0), 32)[0].tolist()) for _ in range(20)]
    examples = pd.DataFrame(examples, columns=["generated"])
    run.log({"example_outputs": wandb.Table(dataframe=examples)})

def main():
    run = wandb.init()

    modelConfig = WherePTConfig(
        vocab_len=VOCAB_LEN,
        n_embed=wandb.config.n_embed,
        n_head=wandb.config.n_head,
        n_layer=wandb.config.n_layer,
        block_size=wandb.config.block_size,
        dropout=wandb.config.dropout
    )
    
    model = WherePT(modelConfig)
    
    run.log({"params": sum(p.numel() for p in model.parameters())})

    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)

    LOG_INTERVAL = 10
    wandb.watch(model, log_freq=LOG_INTERVAL)

    best_val_loss = float("inf")
    for epoch in range(wandb.config.epochs):
        best_val_batch_idx = 0
        for batch_idx in range(len(train_df) // wandb.config.batch_size):
            optimizer.zero_grad()
            xb, yb = get_batch("train")
            logits, loss = model(xb, yb)
            loss.backward()
            optimizer.step()
            if batch_idx % LOG_INTERVAL == 0:
                losses = estimate_loss(model)
                run.log({"loss": losses, "val_loss": losses['val']})
                if losses['val'] < best_val_loss or SAVE_ALL_CHECHPOINTS:
                    best_val_loss = losses['val']
                    best_val_batch_idx = batch_idx
                    if batch_idx > 0:
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'config': dict(wandb.config),
                            'epoch': epoch,
                            'batch_idx': batch_idx,
                            'best_val_loss': best_val_loss,
                        }
                        print(f"saving checkpoint to {OUT_DIR}")
                        torch.save(checkpoint, os.path.join(OUT_DIR, 'ckpt.pt'))
            # Early stopping:
            if losses['val'] > 1.2 * best_val_loss or batch_idx - best_val_batch_idx > 300:
                print("Early stopping")
                log_examples(model, run)
                return
    print("Finished training")
    log_examples(model, run)


if __name__ == "__main__":
    dataset = wandb.Api().artifact("woher/wherept-data:latest")
    train_df = dataset.get("train_df").get_dataframe()
    val_df = dataset.get("val_df").get_dataframe()

    COUNTRY_CODE = "DE"
    train_df = train_df[train_df["country_code"] == "DE"]
    val_df = val_df[val_df["country_code"] == "DE"]
    wandb.agent(sweep_id, function=main, count=8)