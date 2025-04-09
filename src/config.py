config = {
    "datasets": {
        "cora": "data/cora/",
        "citeseer": "data/citeseer/",
        "pubmed": "data/pubmed/"
    },
    "active_dataset": "cora",
    "epochs": 300,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "hidden": 16,
    "dropout": 0.5,
    "early_stopping_patience": 200,  # number of epochs with no improvement before stopping
    "use_cuda": True,
    "apply_attack": False,
    "n_perturbations": 3
}