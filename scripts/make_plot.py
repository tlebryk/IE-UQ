# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

plt.rcParams.update(
    {
        "font.size": 24,  # Double the default font size
        "axes.titlesize": 32,  # Titles of individual plots
        "axes.labelsize": 22,  # Axis labels
        "xtick.labelsize": 24,  # X-axis tick labels
        "ytick.labelsize": 24,  # Y-axis tick labels
        "legend.fontsize": 24,  # Legend text
    }
)


def create_plot(file_paths, file_name="entity_scores_plot.png"):
    # Define labels for each file
    labels = ["Random Baseline", "Active Learning", "Synthetic Span Generation"]

    # Combine all data into a single DataFrame
    all_data = []

    for file_path, label in zip(file_paths, labels):
        if os.path.exists(file_path):  # Check if file exists
            data = pd.read_csv(file_path)
            data["Source"] = label  # Add a column to indicate the data source
            all_data.append(data)
        else:
            print(f"File not found: {file_path}")

    # Concatenate all data
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)

        # Define entities for subplots
        entities = ["basemats", "dopants", "link triplets"]

        # Create a figure with one subplot per entity
        fig, axes = plt.subplots(1, len(entities), figsize=(36, 12), sharey=True)

        for i, entity in enumerate(entities):
            # Filter data for the current entity
            entity_data = combined_data[combined_data["entity"] == entity]

            # Pivot data for easier bar plotting
            pivot_data = entity_data.pivot_table(
                index="metric", columns="Source", values="score"
            ).reset_index()

            # Create a bar plot
            pivot_data_melted = pd.melt(
                pivot_data, id_vars="metric", value_name="score", var_name="Source"
            )
            sns.barplot(
                ax=axes[i],
                data=pivot_data_melted,
                x="metric",
                y="score",
                hue="Source",
                palette="Set2",
                hue_order=[
                    "Random Baseline",
                    "Active Learning",
                    "Synthetic Span Generation",
                ],
            )
            axes[i].set_title(f"{entity.capitalize()} Scores")
            axes[i].set_xlabel("Metric")
            axes[i].set_ylabel("Score")

        plt.tight_layout()
        plt.savefig(file_name, dpi=400, bbox_inches="tight")
        plt.show()
    else:
        print("No data loaded. Check file paths and data structure.")


# %%
# LLAMA 3B plots
# CHANGE TO WHATEVER YOUR LOCAL PATH IS
file_paths = [
    r"G:\My Drive\nlp\Final_project\IE-UQ\runs\random_baseline_uq_results_10_ep\scores.csv",
    r"G:\My Drive\nlp\Final_project\IE-UQ\runs\active_learning_uq_results_10_ep\scores.csv",
    r"G:\My Drive\nlp\Final_project\IE-UQ\runs\synth_span_gen-llama3b_10_final_ep\scores.csv",
]
create_plot(file_paths, file_name="entity_scores_plot_llama3b.png")
# %%
# LLAMA 8B plots
file_paths = [
    r"G:\My Drive\nlp\Final_project\IE-UQ\runs\random_baseline_uq_results_llama8b_10_ep\scores.csv",
    r"G:\My Drive\nlp\Final_project\IE-UQ\runs\active_learning_uq_results_llama8b_10_ep\scores.csv",
    r"G:\My Drive\nlp\Final_project\IE-UQ\runs\synth_extraction_ft_llama8b_10_ep\scores.csv",
]
create_plot(file_paths, file_name="entity_scores_plot_llama8b.png")

# %%
