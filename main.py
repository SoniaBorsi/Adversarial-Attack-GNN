import random
import torch
import os
import constants

from datasets      import load_dataset, patch_data, perturbed_dataset
from model         import GCN, train_model
from nettack       import apply_nettack
from evaluation    import before_attack, after_attack, compare_results, avg_std
from visualization import visualize_graph, plot_accuracy_boxplot

def run_experiment(dataset_name, first_run):
    print("*" * 100)

    # 1) Load & patch
    dataset = load_dataset(dataset_name)
    data    = patch_data(dataset, dataset_name)
    num_classes  = len(torch.unique(data.y))
    pert_rate    = 0.08
    total_budget = int(data.edge_index.size(1) * pert_rate)

    # Ensure output folders
    os.makedirs("results",       exist_ok=True)
    os.makedirs("clean_models",  exist_ok=True)
    os.makedirs("perturbed_data",exist_ok=True)
    os.makedirs("poisoned_models",exist_ok=True)
    os.makedirs("acc_boxplots",  exist_ok=True)
    os.makedirs("visuals",       exist_ok=True)

    # Log header on first run
    if first_run:
        with open(constants.RES_PATH, "a") as f:
            f.write("*" * 100 + "\n")
            f.write(f"Running Nettack with {total_budget} total perturbations\n\n")

    # 2) Train & eval clean
    clean_model = GCN(data.num_features, 16, num_classes)
    clean_path  = os.path.join("clean_models", f"{dataset_name}_clean.pt")
    if os.path.exists(clean_path):
        clean_model.load_state_dict(torch.load(clean_path))
    else:
        clean_model = train_model(clean_model, data)
        torch.save(clean_model.state_dict(), clean_path)
    clean_model.eval()
    acc_clean, prec_clean, rec_clean, f1_clean = before_attack(
        clean_model, data, dataset_name
    )

    # Visualize original graph
    if data.num_nodes <= 5000:
        visualize_graph(
            data.edge_index,
            title=f"{dataset_name}_clean",
            save_dir="visuals"
        )

    # Cache clean-model outputs for targeted metrics
    with torch.no_grad():
        logits_clean = clean_model(data.x, data.edge_index)
        probs_clean  = torch.softmax(logits_clean, dim=1)
        pred_clean   = logits_clean.argmax(dim=1)

    # 3) Pick ~8% of test nodes & per-node budget
    all_tests       = data.test_mask.nonzero(as_tuple=False).view(-1).tolist()
    n_targets       = max(1, int(len(all_tests) * pert_rate))
    targets         = random.sample(all_tests, n_targets)
    budget_per_node = max(1, total_budget // n_targets)

    # Containers for targeted metrics
    targeted_success = []
    confidence_drop  = []

    # 4) Sequentially poison each target
    poisoned_data = data.clone()
    for t in targets:
        # a) train surrogate
        surrogate = GCN(poisoned_data.num_features, 16, num_classes)
        surrogate = train_model(surrogate, poisoned_data)
        surrogate.eval()

        # b) apply Nettack with structure+feature flips
        poisoned_data = apply_nettack(
            model=surrogate,
            data=poisoned_data,
            target_node=t,
            n_perturbations=budget_per_node,
            attack_structure=True,
            attack_features=True
        )

        # c) rebuild perturbed adj/features
        poisoned_perturbed_adj = torch.zeros(
            (poisoned_data.num_nodes, poisoned_data.num_nodes),
            dtype=torch.int32
        )
        poisoned_perturbed_adj[
            poisoned_data.edge_index[0],
            poisoned_data.edge_index[1]
        ] = 1
        poisoned_perturbed_features = poisoned_data.x

        # d) dump per‑target perturbed graph
        perturbed_dataset(
            poisoned_perturbed_adj,
            poisoned_perturbed_features,
            data,
            dataset_name,
            total_budget,
            t
        )

        # e) record targeted metrics
        with torch.no_grad():
            logits_p = surrogate(poisoned_data.x, poisoned_data.edge_index)
            probs_p  = torch.softmax(logits_p, dim=1)
            pred_p   = logits_p.argmax(dim=1)

        targeted_success.append((pred_p[t] != pred_clean[t]).item())
        confidence_drop.append(
            (probs_clean[t, data.y[t]] - probs_p[t, data.y[t]]).item()
        )

    # 5) Retrain & evaluate globally on the fully poisoned_data
    poisoned_model = GCN(poisoned_data.num_features, 16, num_classes)
    poisoned_model = train_model(poisoned_model, poisoned_data)

    # Save final poisoned model
    torch.save(
        poisoned_model.state_dict(),
        os.path.join("poisoned_models",
                     f"{dataset_name}_poisoned{total_budget}_final.pt")
    )

    poisoned_model.eval()
    acc_p, prec_p, rec_p, f1_p = after_attack(
        poisoned_model,
        poisoned_data,
        dataset_name,
        poisoned_perturbed_adj,
        poisoned_perturbed_features
    )

    # Visualize final poisoned graph
    edge_index_perturbed = poisoned_perturbed_adj.nonzero().t().long()
    if poisoned_data.num_nodes <= 5000:
        visualize_graph(
            edge_index_perturbed,
            title=f"{dataset_name}_poisoned_final",
            save_dir="visuals"
        )

    # 6) Log global metrics
    compare_results(
        dataset_name,
        acc_clean, prec_clean, rec_clean, f1_clean,
        acc_p,    prec_p,    rec_p,    f1_p
    )
    avg_std([acc_p], [prec_p], [rec_p], [f1_p], dataset_name)

    # 7) Log targeted aggregates
    tsr      = sum(targeted_success) / len(targeted_success)
    avg_drop = sum(confidence_drop)  / len(confidence_drop)
    with open(constants.RES_PATH, "a") as f:
        f.write(f"Targeted Success Rate (Nettack): {tsr:.4f}\n")
        f.write(f"Average Confidence Drop:         {avg_drop:.4f}\n\n")

    # 8) Plot global accuracy
    plot_accuracy_boxplot(
        acc_clean, [acc_p],
        int(pert_rate * 100),
        dataset_name,
        total_budget
    )


    
def main():
    """
    Run the experiment for all datasets.
    Uncomment the datasets you want to run.
    If you want to run all datasets, uncomment the line below.
    Note: Some datasets may take a long time to run or may require more memory.
    """
    # all datasets, uncomment when debugged all
    #datasets = [constants.CORA, constants.CITESEER, constants.POLBLOGS, constants.TEXAS, constants.FLICKR, constants.PUBMED, constants.OGBN_PROTEINS]

    # For now, core datasets 
    #datasets = [constants.CORA, constants.CITESEER, constants.POLBLOGS, constants.TEXAS]

    # TODO: debug these datasets
    # These datasets crash - reason: too big, use all available RAM
    #datasets = [constants.PUBMED, constants.FLICKR, constants.OGBN_PROTEINS]

    # For debugging purposes, uncomment the dataset you want to run
    #datasets = [constants.CORA]
    datasets = [constants.CITESEER]
    #datasets = [constants.POLBLOGS]
    #datasets = [constants.TEXAS]
    #datasets = [constants.FLICKR]
    #datasets = [constants.PUBMED]
    #datasets = [constants.OGBN_PROTEINS]

    
    first_run = True
        
    for dataset_name in datasets:
        run_experiment(dataset_name, first_run)
        first_run = False



if __name__ == "__main__": 
    main()