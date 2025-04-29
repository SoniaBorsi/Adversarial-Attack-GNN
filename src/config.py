config = {
    "datasets": {
        "cora":   "data/cora/",
        "citeseer":"data/citeseer/",
        "pubmed": "data/pubmed/",
        "polblogs":"data/polblogs/",
        "texs":"data/texas/"
    },
    "active_dataset":      "citeseer",
    "use_perturbed":       False,    
    "perturbed_budget": 400, 
    "epochs":              300,
    "lr":                  0.01,
    "weight_decay":        5e-4,
    "hidden":              16,
    "dropout":             0.5,
    "early_stopping_patience": 200,
    "use_cuda":            True
}
