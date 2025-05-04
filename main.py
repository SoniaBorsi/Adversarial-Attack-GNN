import torch
import os
import constants

# Import the other files
from datasets import load_dataset, split_masks, patch_data, perturbed_dataset
from model import GCN, train_model
from metattack import apply_metattack
from evaluation import evaluate_model, before_attack, after_attack, compare_results, avg_std
from visualization import visualize_graph, plot_accuracy_boxplot

def run_experiment(dataset_name, first_run):
    """
    Run the experiment for a given dataset.
    Args:
        dataset_name (str): Name of the dataset to run the experiment on.
    """

    print("*" * 100) # Separate the dataset run 

    # Load the dataset and patch --------------------------------
    dataset = load_dataset(dataset_name)
    data = patch_data(dataset, dataset_name)
    poisoned_data = data.clone()
    num_classes = len(torch.unique(data.y))
    pert_rate = 0.08
    num_perturbations = int(data.edge_index.size(1) * pert_rate)#int(data.num_nodes * 0.3)

    print(f"Number of perturbations: {num_perturbations}")

    if first_run: # Write num perturbations in file
        os.makedirs("results", exist_ok=True)  
        with open(constants.RES_PATH, "a") as f:  
            f.write("*" * 100 + "\n")
            f.write(f"Running Metattack with {num_perturbations} perturbations\n\n")

    print(f"Dataset: {dataset_name}")
    print(f"    Number of nodes: {data.num_nodes}") 
    print(f"    Number of features: {data.num_features}")
    print(f"    Number of classes: {num_classes}\n")

    # Train and evaluate on clean model, save graph ----------------------------------------
    clean_model = GCN(data.num_features, 16, num_classes)
    os.makedirs("clean_models", exist_ok=True)  # Make sure the folder exists
    clean_model_path = os.path.join("clean_models", f"{dataset_name}_clean{num_perturbations}.pt")

    if os.path.exists(clean_model_path):
        print(f"Model already exists at {clean_model_path}. Loading the model.")
        clean_model.load_state_dict(torch.load(clean_model_path, weights_only=True))
    else:
        print(f"Training and saving clean model to {clean_model_path}")
        clean_model = train_model(clean_model, data)
        torch.save(clean_model.state_dict(), clean_model_path)

    clean_model.eval()
    acc_clean, prec_clean, rec_clean, f1_clean = before_attack(
        clean_model, data, dataset_name
    )
    
    if data.num_nodes <= 5000:
        # Visualize the original graph if it is small enough
        print("-" * 100)
        print("Original Graph:")
        visualize_graph(
            data.edge_index, 
            title=f"{dataset_name}_clean_pert{num_perturbations}", 
            save_dir="visuals"
        )

    # Save results ---------------------------------------------------------
    acc_poisoned_list = []
    prec_poisoned_list = []
    rec_poisoned_list = []
    f1_poisoned_list = []

    # Apply Metattack ------------------------------------------------------
    for run in range(3):
        print(f"---- Run {run + 1} ---\n")

        with open(constants.RES_PATH, "a") as f:
            f.write(f"---- Run {run + 1} ----\n")
    
        attack_model = GCN(poisoned_data.num_features, 16, num_classes)
        poisoned_perturbed_adj, poisoned_perturbed_features = apply_metattack(
            attack_model, 
            poisoned_data, 
            num_perturbations=num_perturbations
        )

        original_adj = torch.zeros((data.num_nodes, data.num_nodes), dtype=torch.int32)
        original_adj[data.edge_index[0], data.edge_index[1]] = 1
        poisoned_adj = poisoned_perturbed_adj.to(torch.int32)

        edge_index_perturbed = poisoned_perturbed_adj.nonzero().t()
    
        # Train and evaluate on poisoned dataset, save graph --------------------------------------------------------------
        poisoned_data.edge_index = edge_index_perturbed
        poisoned_data.x = poisoned_perturbed_features
        poisoned_model = GCN(poisoned_data.num_features, 16, num_classes)
        poisoned_model = train_model(poisoned_model, poisoned_data)

        # Save model and dataset
        poisoned_model_path = os.path.join("poisoned_models", f"{dataset_name}_poisoned{num_perturbations}_run{run+1}.pt")
        os.makedirs("poisoned_models", exist_ok=True)
        torch.save(poisoned_model.state_dict(), poisoned_model_path)
        perturbed_dataset(poisoned_perturbed_adj, poisoned_perturbed_features, data, dataset_name, num_perturbations, run)

        poisoned_model.eval()
        acc_poisoned, prec_poisoned, rec_poisoned, f1_poisoned = after_attack(
            poisoned_model, poisoned_data, dataset_name,
            poisoned_perturbed_adj, poisoned_perturbed_features
        )

        # Store the results for the current run
        acc_poisoned_list.append(acc_poisoned)
        prec_poisoned_list.append(prec_poisoned)
        rec_poisoned_list.append(rec_poisoned)
        f1_poisoned_list.append(f1_poisoned)        

        if data.num_nodes <= 5000:
            print("-" * 100)
            print("Perturbed Graph:")
            visualize_graph(
                edge_index_perturbed,
                title=f"{dataset_name}_poisoned_pert{num_perturbations}_run{run+1}",
                save_dir="visuals"
            )
        
        # Compare metrics --------------------------------------------------------------
        compare_results(
            dataset_name,
            acc_clean, prec_clean, rec_clean, f1_clean,
            acc_poisoned, prec_poisoned, rec_poisoned, f1_poisoned
        )
    
    with open(constants.RES_PATH, "a") as f:
        f.write(f"---- Poisoned Overall ----\n")
    # Save the average results for the dataset
    avg_std(acc_poisoned_list, prec_poisoned_list, rec_poisoned_list, f1_poisoned_list, dataset_name)

    # plot accuracy
    plot_accuracy_boxplot(acc_clean, acc_poisoned_list, int(pert_rate * 100), dataset_name, num_perturbations)

    
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
    #datasets = [constants.CITESEER]
    #datasets = [constants.POLBLOGS]
    #datasets = [constants.TEXAS]
    #datasets = [constants.FLICKR]
    datasets = [constants.PUBMED]
    #datasets = [constants.OGBN_PROTEINS]

    first_run = True
        
    for dataset_name in datasets:
        run_experiment(dataset_name, first_run)
        first_run = False



if __name__ == "__main__": 
    main()