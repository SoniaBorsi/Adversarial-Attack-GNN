config = {
    "datasets": {
        "cora":   "data/cora/",
        "citeseer":"data/citeseer/",
        "pubmed": "data/pubmed/",
        "polblogs":"data/polblogs/",
        "texas":"data/texas/"
    },
    "active_dataset":      "cora",
    "use_perturbed":       True,    
    "epochs":              200,
    "lr":                  0.001,
    "weight_decay":        5e-4,
    "hidden":              16,
    "dropout":             0.5,
    "early_stopping_patience": 50,
    "use_cuda":            True
}

