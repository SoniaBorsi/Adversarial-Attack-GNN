# ───────────────────────────── main.py ─────────────────────────────
import time
import torch
import torch.optim as optim
import wandb

from model  import GCN
from data   import load_data
from config import config
from utils  import train, test


def main() -> None:
    # ------------------------- wandb init -------------------------------
    wandb.init(
        project="gnn",
        config=config,
        name=f"{config['active_dataset']}_run_{int(time.time())}",
    )
    cfg = wandb.config

    # ------------------------- load data --------------------------------
    root = cfg["datasets"][cfg["active_dataset"]]
    adj, feats, labels, idx_tr, idx_te = load_data(
        root,
        dataset=cfg["active_dataset"],
        use_perturbed=True,
    )

    # ------------------------- build model ------------------------------
    model = GCN(
        nfeat   = feats.shape[1],
        nhid    = cfg["hidden"],
        nclass  = labels.max().item() + 1,
        dropout = cfg["dropout"],
    )

    optimiser = optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    use_cuda = cfg["use_cuda"] and torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        feats, adj, labels = feats.cuda(), adj.cuda(), labels.cuda()
        idx_tr, idx_te    = idx_tr.cuda(), idx_te.cuda()

    # ------------------------- training loop ----------------------------
    best, impatience = 0.0, 0
    for epoch in range(cfg["epochs"]):
        train_loss, _ = train(model, optimiser,
                              feats, adj, labels, idx_tr, epoch, use_cuda)
        val_loss, acc = test(model, feats, adj, labels, idx_te, epoch)

        if acc > best:
            best, impatience = acc, 0
        else:
            impatience += 1

        if impatience >= cfg["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch} – best acc {best:.4f}")
            break

    print("Training completed.")


if __name__ == "__main__":
    main()




