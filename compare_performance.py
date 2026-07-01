import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

def plot_base_losses():
    runs = {
        "full encoder": np.loadtxt("losses_25base-full.txt"),
        "5 layers": np.loadtxt("losses_25base-5layers.txt"),
        "4 layers": np.loadtxt("losses_25base-4layers.txt"),
        "4 layers w/out normalization": np.loadtxt("losses_25base-4layers-no-norm.txt"),
        "3 layers": np.loadtxt("losses_25base-3layers.txt"),
        "2 layers": np.loadtxt("losses_25base-2layers.txt"),
        "1 layer": np.loadtxt("losses_25base-1layer.txt"),
        "conv2": np.loadtxt("losses_25base-conv2.txt"),
        "conv1": np.loadtxt("losses_25base-conv1.txt"),
    }

    plt.figure()
    # colors = ['#f113ac', '#13acf1', '#acf113', '#12a180',  '#222000','#223FFF', '#']  
    for name, loss in runs.items():
        plt.plot(loss, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss Comparison with Embeddings from Different Stages of Whisper-base")
    plt.legend()

    plt.grid(True)

    plt.savefig("loss_comparison_base_mean.png")
    plt.close()

def plot_small_losses():
    runs = {
        "full encoder": np.loadtxt("losses_small-full.txt"),
        "11 layers": np.loadtxt("losses_small_mean-11layers.txt"),
        "10 layers": np.loadtxt("losses_small-10layers.txt"),
        "9 layers": np.loadtxt("losses_small_mean-9layers.txt"),
        "8 layers": np.loadtxt("losses_small-8layers.txt"),
        "7 layers": np.loadtxt("losses_small_mean-7layers.txt"),
        "6 layers": np.loadtxt("losses_small-6layers.txt"),
        "4 layers": np.loadtxt("losses_small-4layers.txt"),
        # "2 layers": np.loadtxt("losses_small-2layers.txt"),
        "conv2": np.loadtxt("losses_small-conv2.txt"),
        "conv1": np.loadtxt("losses_small-conv1.txt"),
    }

    plt.figure()
    # colors = ['#f113ac', '#13acf1', '#acf113', '#12a180',  '#222000','#223FFF', '#']  
    for name, loss in runs.items():
        plt.plot(loss, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss Comparison with Embeddings from Different Stages of Whisper-small")
    plt.legend()

    plt.grid(True)

    plt.savefig("loss_comparison_small_mean.png")
    plt.close()

def plot_medium_losses():
    runs = {
        "full encoder": np.loadtxt("losses_medium_mean-full.txt"),
        "23 transfomer layers": np.loadtxt("losses_medium_mean-23layers.txt"),
        "22 transfomer layers": np.loadtxt("losses_medium_mean-22layers.txt"),
        "21 transfomer layers": np.loadtxt("losses_medium_mean-21layers.txt"),
        "20 transfomer layers": np.loadtxt("losses_medium_mean-20layers.txt"),
        "19 transfomer layers": np.loadtxt("losses_medium_mean-19layers.txt"),
        # "18 transfomer layers": np.loadtxt("losses_medium_mean-18layers.txt"),
        # "17 transfomer layers": np.loadtxt("losses_medium_mean-17layers.txt"),
        "16 transfomer layers": np.loadtxt("losses_medium_mean-16layers.txt"),
        # "15 transfomer layers": np.loadtxt("losses_medium_mean-15layers.txt"),
        "14 transfomer layers": np.loadtxt("losses_medium_mean-14layers.txt"),
        # "13 transfomer layers": np.loadtxt("losses_medium_mean-13layers.txt"),
        # "12 transfomer layers": np.loadtxt("losses_medium_mean-12layers.txt"),
        # "11 transfomer layers": np.loadtxt("losses_medium_mean-11layers.txt"),
        # "10 transfomer layers": np.loadtxt("losses_medium_mean-10layers.txt"),
        # "8 transfomer layers": np.loadtxt("losses_medium_mean-8layers.txt"),
        # "4 transfomer layers": np.loadtxt("losses_medium_mean-4layers.txt"),
        "conv2": np.loadtxt("losses_medium_mean-conv2.txt"),
        # "conv1": np.loadtxt("losses_medium_mean-conv1.txt"),

    
    
    }

    plt.figure()
    # colors = ['#f113ac', '#13acf1', '#acf113', '#12a180',  '#222000','#223FFF', '#']  
    for name, loss in runs.items():
        plt.plot(loss, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss Comparison with Embeddings from Different Stages of Whisper-medium")
    plt.legend()

    plt.grid(True)

    plt.savefig("loss_comparison_medium_mean.png", bbox_inches="tight")
    plt.close()

def plot_large_losses():
    # Read and filter rows
    with open("full_output_log.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row["model"] == "large-v1"]
    # Define which layer indices to keep
    keep_layers = {0,24,8,30,32,23,25, 31, 28, 16}  # example values

    # Filter rows to only those with matching layer_index
    rows = [row for row in rows if int(float(row["layer_index"])) in keep_layers]

    # Sort by layer_index descending
    rows.sort(key=lambda r: int(float(r["layer_index"])), reverse=True)
    keep = []
    # Build the runs dict
    runs = {}
    for row in rows:

        if row["layer_index"] == 32:
            label = "full encoder"
        else:
            label = row["layer_type"]  # or customize the label however you like
        filename = f"losses_{row['run_name']}.txt"
        runs[label] = np.loadtxt(filename)



    plt.figure()
    for name, loss in runs.items():
        plt.plot(loss, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss Comparison with Embeddings from Different Stages of Whisper-large")
    plt.legend()

    plt.grid(True)

    plt.savefig("loss_comparison_large_mean.png", bbox_inches="tight")
    plt.close()


def plot_losses():
    # specify training runs i want to compare based on saved loss files in losses folder
    runs = {
        "base embeddings": np.loadtxt("losses_25base.txt"), 
        "small embeddings": np.loadtxt("losses_25small.txt"), 
        "medium embeddings": np.loadtxt("losses_25medium.txt"),
        "large embeddings": np.loadtxt("losses_25large.txt"),
        "large-v2 embeddings": np.loadtxt("losses_25largev2.txt"),
        "large-v3 embeddings": np.loadtxt("losses_25largev3.txt"),
        "large-v3-turbo embeddings": np.loadtxt("losses_25largev3turbo.txt")
    }

    plt.figure()
    # colors = ['#f113ac', '#13acf1', '#acf113', '#12a180',  '#222000','#223FFF', '#']  
    for name, loss in runs.items():
        plt.plot(loss, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Model Training Loss Comparison based on Different Embeddings")
    plt.legend()

    plt.grid(True)

    plt.savefig("loss_comparison.png")
    plt.close()


def plot_f1_base():
    dl_df = pd.read_csv("detect_lang_official.csv")
    dl_df_base = dl_df.loc[dl_df["run"] == "base"]
    df = pd.read_csv("full_output_log.csv")
    base_mean_df = df.loc[(df["model"]=="base") & (df["aggregation"]=="mean")]
    
    base_mean_df = base_mean_df.iloc[::-1]

    
    base_mean_filtered = base_mean_df[
        ~(
            (base_mean_df["layer_index"] > 0) &
            (~base_mean_df["normalization"])
        )
    ]    # choose only rows where normalization was done

    base_mean_filtered = base_mean_filtered[base_mean_filtered['layer_index'] >= 3]

    small_mean_df = df.loc[(df["model"]=="small") & (df["aggregation"]=="mean")]
    small_mean_filtered = small_mean_df.sort_values(by='layer_index', ascending=False)

    small_mean_filtered = small_mean_filtered[
        ~(
            (small_mean_filtered["layer_index"] > 0) &
            (~small_mean_filtered["normalization"])
        )
    ]  

    small_mean_filtered = small_mean_filtered[small_mean_filtered['layer_index'] >= 6]

    

    plt.figure(figsize=(10, 8))
    # plt.bar(df["run"], df["macro_f1"], color='#25f914', label="MLP with Final Hidden Whisper Embeddings")
    # plt.bar("whisper-base", dl_df["macro_f1"], color='#000000', label="Whisper's Built-in Language Detection")
    plt.bar("whisper-small", dl_df["macro_f1"], color='#000000', label="Whisper's Built-in Language Detection")

    # plt.bar(base_mean_filtered["layer_type"], base_mean_filtered["macro_f1"], color="#ba19e1", label="MLP on layer-wise Whisper-base encoder embeddings (mean pooled)")
    plt.bar(small_mean_filtered["layer_type"], small_mean_filtered["macro_f1"], color="#1913a1", label="MLP on layer-wise Whisper-small encoder embeddings (mean pooled)")

    plt.ylabel("Macro F1 Score")
    plt.xlabel("Experiment")
    plt.title("Macro F1 Score Comparison for Different Usages of Whisper-small")
    plt.legend(loc="best")
    plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()


    plt.savefig("f1_comparison_small_mean_relevant.png", bbox_inches='tight')
    plt.close()

def plot_f1_medium():
    dl_df = pd.read_csv("detect_lang_official.csv")
    dl_df_base = dl_df.loc[dl_df["run"] == "medium"]
    df = pd.read_csv("full_output_log.csv")
    base_mean_df = df.loc[(df["model"]=="medium") & (df["aggregation"]=="mean")]
    
    base_mean_df = base_mean_df.sort_values(by='layer_index', ascending=False)

    
    base_mean_filtered = base_mean_df[
        ~(
            (base_mean_df["layer_index"] > 0) &
            (~base_mean_df["normalization"])
        )
    ]    # choose only rows where normalization was done
    max_points_row = df.loc[df['macro_f1'].idxmax()]
    print(max_points_row)
    base_mean_filtered = base_mean_filtered[base_mean_filtered['layer_index'] >= 12]

    

    plt.figure(figsize=(10, 8))

    plt.bar("whisper-medium", dl_df["macro_f1"], color='#000000', label="Whisper's Built-in Language Detection")

    # plt.bar(base_mean_filtered["layer_type"], base_mean_filtered["macro_f1"], color="#ba19e1", label="MLP on layer-wise Whisper-base encoder embeddings (mean pooled)")
    plt.bar(base_mean_filtered["layer_type"], base_mean_filtered["macro_f1"], color="#13ed92", label="MLP on layer-wise Whisper-medium encoder embeddings (mean pooled)")

    plt.ylabel("Macro F1 Score")
    plt.xlabel("Experiment")
    plt.title("Macro F1 Score Comparison for Different Usages of Whisper-medium")
    plt.legend(loc="lower right")
    plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()


    plt.savefig("f1_comparison_medium_mean_relevant.png", bbox_inches='tight')
    plt.close()


def plot_f1_vs_inference_time_base():
    dl_df = pd.read_csv("detect_lang_official.csv")
    dl_df_base = dl_df.loc[dl_df["run"] == "base"]

    df = pd.read_csv("full_output_log.csv")
    base_mean_df = df.loc[(df["model"]=="base") & (df["aggregation"]=="mean")]
    base_mean_filtered = base_mean_df
    # base_mean_filtered = base_mean_df[base_mean_df['layer_index'].isin([0,2,3,4,5,6])]
    base_mean_filtered = base_mean_filtered[
        ~(
            (base_mean_filtered["layer_index"] > 0) &
            (~base_mean_filtered["normalization"])
        )
    ]    
    print(base_mean_filtered)
    # small_mean_df = df.loc[(df["model"]=="small") & (df["aggregation"]=="mean")]
    # small_mean_filtered = small_mean_df[small_mean_df['layer_index'].isin([4,6,8,10,12])]
    # small_mean_filtered = small_mean_filtered[
    #     ~(
    #         (small_mean_filtered["layer_index"] > 0) &
    #         (~small_mean_filtered["normalization"])
    #     )
    # ]  
    
    plt.figure(figsize=(10, 8))
    # plt.scatter(df["inference_time"], df["macro_f1"], color='#25f914', label="MLP with Final Hidden Whisper Embeddings")
    plt.scatter(dl_df_base["inference_time"], dl_df_base["macro_f1"], color='#000000', label="Whisper's Built-in Language Detection")
    plt.scatter(base_mean_filtered["inference_time"], base_mean_filtered["macro_f1"], color="#ba19e1", label="MLP on layer-wise Whisper-base encoder embeddings (mean pooled)")
    # plt.scatter(small_mean_filtered["inference_time"], small_mean_filtered["macro_f1"], color="#1913a1", label="MLP on layer-wise Whisper-small encoder embeddings (mean pooled)")

    plt.xlabel("Inference Time (seconds)")
    plt.ylabel("Macro F1 Score")
    plt.title("Macro F1 Score vs Inference Time")
    plt.legend()

    for i, row in base_mean_filtered.iterrows():
        plt.annotate(row["layer_type"], (row["inference_time"], row["macro_f1"]))
    # for i, row in small_mean_filtered.iterrows():
    #     plt.annotate(row["layer_type"], (row["inference_time"], row["macro_f1"]))
    for i, row in dl_df_base.iterrows():
        plt.annotate(row["run"], (row["inference_time"], row["macro_f1"]))

    plt.savefig("f1_vs_inference_time_base_mean.png")
    plt.close()

def plot_f1_vs_inference_time_small():
    dl_df = pd.read_csv("detect_lang_official.csv")
    dl_df_base = dl_df.loc[dl_df["run"] == "small"]

    df = pd.read_csv("full_output_log.csv")
   
    small_mean_df = df.loc[(df["model"]=="small") & (df["aggregation"]=="mean")]
    small_mean_filtered = small_mean_df
    # small_mean_filtered = small_mean_df[small_mean_df['layer_index'].isin([4,6,8,10,12])]
    small_mean_filtered = small_mean_filtered[
        ~(
            (small_mean_filtered["layer_index"] > 0) &
            (~small_mean_filtered["normalization"])
        )
    ]  
    
    plt.figure(figsize=(10, 8))
    # plt.scatter(df["inference_time"], df["macro_f1"], color='#25f914', label="MLP with Final Hidden Whisper Embeddings")
    plt.scatter(dl_df_base["inference_time"], dl_df_base["macro_f1"], color='#000000', label="Whisper's Built-in Language Detection")
    # plt.scatter(base_mean_filtered["inference_time"], base_mean_filtered["macro_f1"], color="#ba19e1", label="MLP on layer-wise Whisper-base encoder embeddings (mean pooled)")
    plt.scatter(small_mean_filtered["inference_time"], small_mean_filtered["macro_f1"], color="#1913a1", label="MLP on layer-wise Whisper-small embeddings (mean pooled)")

    plt.xlabel("Inference Time (seconds)")
    plt.ylabel("Macro F1 Score")
    plt.title("Macro F1 Score vs Inference Time for Whisper-Small")
    plt.legend()

    # for i, row in base_mean_filtered.iterrows():
    #     plt.annotate(row["layer_type"], (row["inference_time"], row["macro_f1"]))
    for i, row in small_mean_filtered.iterrows():
        plt.annotate(row["layer_type"], (row["inference_time"], row["macro_f1"]))
    for i, row in dl_df_base.iterrows():
        plt.annotate(row["run"], (row["inference_time"], row["macro_f1"]))

    plt.savefig("f1_vs_inference_time_small_mean.png")
    plt.close()

def plot_f1_vs_inference_time_medium():
    dl_df = pd.read_csv("detect_lang_official.csv")
    dl_df_base = dl_df.loc[dl_df["run"] == "medium"]

    df = pd.read_csv("full_output_log.csv")
   
    small_mean_df = df.loc[(df["model"]=="medium") & (df["aggregation"]=="mean")]
    small_mean_filtered = small_mean_df
    small_mean_filtered = small_mean_df[small_mean_df['layer_index'].isin([0,4,8,10,11,12,16,19,24])]
    small_mean_filtered = small_mean_filtered[
        ~(
            (small_mean_filtered["layer_index"] > 0) &
            (~small_mean_filtered["normalization"])
        )
    ]  
    
    plt.figure(figsize=(10, 8))
    # plt.scatter(df["inference_time"], df["macro_f1"], color='#25f914', label="MLP with Final Hidden Whisper Embeddings")
    plt.scatter(dl_df_base["inference_time"], dl_df_base["macro_f1"], color='#000000', label="Whisper's Built-in Language Detection")
    # plt.scatter(base_mean_filtered["inference_time"], base_mean_filtered["macro_f1"], color="#ba19e1", label="MLP on layer-wise Whisper-base encoder embeddings (mean pooled)")
    plt.scatter(small_mean_filtered["inference_time"], small_mean_filtered["macro_f1"], color="#13ed92", label="MLP on layer-wise Whisper-medium embeddings (mean pooled)")

    plt.xlabel("Inference Time (seconds)")
    plt.ylabel("Macro F1 Score")
    plt.title("Macro F1 Score vs Inference Time for Whisper-Medium")
    plt.legend()

    # for i, row in base_mean_filtered.iterrows():
    #     plt.annotate(row["layer_type"], (row["inference_time"], row["macro_f1"]))
    for i, row in small_mean_filtered.iterrows():
        plt.annotate(row["layer_type"], (row["inference_time"], row["macro_f1"]))
    for i, row in dl_df_base.iterrows():
        plt.annotate(row["run"], (row["inference_time"], row["macro_f1"]))

    plt.savefig("f1_vs_inference_time_medium_mean.png")
    plt.close()


def plot_f1_vs_inference_time_large():
    dl_df = pd.read_csv("detect_lang_official.csv")
    dl_df_base = dl_df.loc[dl_df["run"] == "large"]

    df = pd.read_csv("full_output_log.csv")
   
    small_mean_df = df.loc[(df["model"]=="large-v1") & (df["aggregation"]=="mean")]
    small_mean_filtered = small_mean_df
    small_mean_filtered = small_mean_df[small_mean_df['layer_index'].isin([22,23,24,25,28,30,31,32])]
    small_mean_filtered = small_mean_filtered[
        ~(
            (small_mean_filtered["layer_index"] > 0) &
            (~small_mean_filtered["normalization"])
        )
    ]  
    
    plt.figure(figsize=(10, 8))
    # plt.scatter(df["inference_time"], df["macro_f1"], color='#25f914', label="MLP with Final Hidden Whisper Embeddings")
    plt.scatter(dl_df_base["inference_time"], dl_df_base["macro_f1"], color='#000000', label="Whisper's Built-in Language Detection")
    # plt.scatter(base_mean_filtered["inference_time"], base_mean_filtered["macro_f1"], color="#ba19e1", label="MLP on layer-wise Whisper-base encoder embeddings (mean pooled)")
    plt.scatter(small_mean_filtered["inference_time"], small_mean_filtered["macro_f1"], color="#a1807e", label="MLP on layer-wise Whisper-large-v1 embeddings (mean pooled)")

    plt.xlabel("Inference Time (seconds)")
    plt.ylabel("Macro F1 Score")
    plt.title("Macro F1 Score vs Inference Time for Whisper-Large-v1")
    plt.legend(loc="best")

    # for i, row in base_mean_filtered.iterrows():
    #     plt.annotate(row["layer_type"], (row["inference_time"], row["macro_f1"]))
    for i, row in small_mean_filtered.iterrows():
        plt.annotate(row["layer_type"], (row["inference_time"], row["macro_f1"]))
    for i, row in dl_df_base.iterrows():
        plt.annotate(row["run"], (row["inference_time"], row["macro_f1"]))

    plt.savefig("f1_vs_inference_time_large_relevant.png")
    plt.close()


def plot_f1_vs_inference_time():
    dl_df = pd.read_csv("detect_lang_official.csv")
    df = pd.read_csv("full_output_log.csv")
    base_mean_df = df.loc[(df["model"]=="base") & (df["aggregation"]=="mean")]
    base_mean_filtered = base_mean_df[base_mean_df['layer_index'].isin([3,4,5,6])]
    base_mean_filtered = base_mean_filtered[
        ~(
            (base_mean_filtered["layer_index"] > 0) &
            (~base_mean_filtered["normalization"])
        )
    ]    
    print(base_mean_filtered)
    small_mean_df = df.loc[(df["model"]=="small") & (df["aggregation"]=="mean")]
    small_mean_filtered = small_mean_df[small_mean_df['layer_index'].isin([4,6,8,10,12])]
    small_mean_filtered = small_mean_filtered[
        ~(
            (small_mean_filtered["layer_index"] > 0) &
            (~small_mean_filtered["normalization"])
        )
    ]  

    medium_mean_df = df.loc[(df["model"]=="medium") & (df["aggregation"]=="mean")]
    medium_mean_filtered = medium_mean_df[
        ~(
            (medium_mean_df["layer_index"] > 0) &
            (~medium_mean_df["normalization"])
        )
    ]  
    medium_mean_filtered = medium_mean_filtered[medium_mean_filtered['layer_index']>0]
    
    large_mean_df = df.loc[(df["model"]=="large-v1") & (df["aggregation"]=="mean")]
    large_mean_filtered = large_mean_df[
        ~(
            (large_mean_df["layer_index"] > 0) &
            (~large_mean_df["normalization"])
        )
    ]  
    large_mean_filtered = large_mean_filtered[large_mean_filtered['layer_index']>0]


    plt.figure(figsize=(17, 8))
    # plt.scatter(df["inference_time"], df["macro_f1"], color='#25f914', label="MLP with Final Hidden Whisper Embeddings")
    plt.scatter(dl_df["inference_time"], dl_df["macro_f1"], color='#000000', label="Whisper's Built-in Language Detection")
    plt.scatter(base_mean_filtered["inference_time"], base_mean_filtered["macro_f1"], color="#ba19e1", label="MLP on Whisper-base encoder embeddings")
    plt.scatter(small_mean_filtered["inference_time"], small_mean_filtered["macro_f1"], color="#1913a1", label="MLP on Whisper-small encoder embeddings")
    plt.scatter(medium_mean_filtered["inference_time"], medium_mean_filtered["macro_f1"], color="#13ed92", label="MLP on Whisper-medium encoder embeddings")
    plt.scatter(large_mean_filtered["inference_time"], large_mean_filtered["macro_f1"], color="#a1807e", label="MLP on Whisper-large encoder embeddings")


    plt.xlabel("Inference Time (seconds)")
    plt.ylabel("Macro F1 Score")
    plt.title("Macro F1 Score vs Inference Time on Mean Pooled Embeddings from Different Encoder Layers")
    plt.legend()

    for i, row in base_mean_filtered.iterrows():
        plt.annotate(row["layer_type"], (row["inference_time"], row["macro_f1"]))
    for i, row in small_mean_filtered.iterrows():
        plt.annotate(row["layer_type"], (row["inference_time"], row["macro_f1"]))
    for i, row in medium_mean_filtered.iterrows():
        plt.annotate(row["layer_type"], (row["inference_time"], row["macro_f1"]))
    for i, row in large_mean_filtered.iterrows():
        plt.annotate(row["layer_type"], (row["inference_time"], row["macro_f1"]))
    for i, row in dl_df.iterrows():
        plt.annotate(row["run"], (row["inference_time"], row["macro_f1"]))

    plt.savefig("f1_vs_inference_time_mean_aggr.png")
    plt.close()


def plot_gpu():
    df = pd.read_csv("full_output_log.csv")
    dl_df = pd.read_csv("detect_lang_official.csv")
    dl_df_base = dl_df.loc[dl_df["run"] == "small"]

    base_mean_df = df.loc[(df["model"]=="base") & (df["aggregation"]=="mean")]
    base_mean_filtered = base_mean_df[base_mean_df['layer_index'].isin([0,2,3,4,5,6])]
    base_mean_filtered = base_mean_filtered[
        ~(
            (base_mean_filtered["layer_index"] > 0) &
            (~base_mean_filtered["normalization"])
        )
    ]  
    small_mean_df = df.loc[(df["model"]=="small") & (df["aggregation"]=="mean")]
    small_mean_filtered = small_mean_df[
        ~(
            (small_mean_df["layer_index"] > 0) &
            (~small_mean_df["normalization"])
        )
    ]  
    small_mean_filtered = small_mean_df[small_mean_df['layer_index'].isin([-1,0,4,6,10,12])]

    medium_mean_df = df.loc[(df["model"]=="medium") & (df["aggregation"]=="mean")]
    medium_mean_filtered = medium_mean_df[
        ~(
            (medium_mean_df["layer_index"] > 0) &
            (~medium_mean_df["normalization"])
        )
    ]  
    # medium_mean_filtered = medium_mean_filtered[medium_mean_filtered['layer_index']>0]
    
    large_mean_df = df.loc[(df["model"]=="large-v1") & (df["aggregation"]=="mean")]
    large_mean_filtered = large_mean_df[
        ~(
            (large_mean_df["layer_index"] > 0) &
            (~large_mean_df["normalization"])
        )
    ]  
    # large_mean_filtered = large_mean_filtered[large_mean_filtered['layer_index']>0]

    plt.figure(figsize=(17, 8))
    # plt.scatter(df["peak_memory"], df["macro_f1"], color='#25f914', label="MLP with Final Hidden Whisper Embeddings")
    # plt.scatter(dl_df["peak_memory"], dl_df["macro_f1"], color='#000000', label="Whisper's Built-in Language Detection")
    plt.scatter(dl_df_base["peak_memory"], dl_df_base["macro_f1"], color='#000000', label="Whisper's Built-in Language Detection")

    # plt.scatter(base_mean_filtered["peak_memory"], base_mean_filtered["macro_f1"], color="#ba19e1", label="MLP on Whisper-base encoder embeddings")
    plt.scatter(small_mean_filtered["peak_memory"], small_mean_filtered["macro_f1"], color="#1913a1", label="MLP on Whisper-small embeddings")
    # plt.scatter(medium_mean_filtered["peak_memory"], medium_mean_filtered["macro_f1"], color="#13ed92", label="MLP on Whisper-medium encoder embeddings")
    # plt.scatter(large_mean_filtered["peak_memory"], large_mean_filtered["macro_f1"], color="#a1807e", label="MLP on Whisper-large encoder embeddings")

    plt.xlabel("Peak Memory Usage (MB)")
    plt.ylabel("Macro F1 Score")
    plt.title("Macro F1 Score vs Peak Memory Usage on Mean Pooled Embeddings from Different Encoder Layers")
    plt.legend()

    # for i, row in df.iterrows():
    #     plt.annotate(row["run"], (row["peak_memory"], row["macro_f1"]))
    for i, row in dl_df.iterrows():
        plt.annotate(row["run"], (row["peak_memory"], row["macro_f1"]))
    for i, row in base_mean_filtered.iterrows():
        plt.annotate(row["layer_type"], (row["peak_memory"], row["macro_f1"]))
    for i, row in small_mean_filtered.iterrows():
        plt.annotate(row["layer_type"], (row["peak_memory"], row["macro_f1"]))
    for i, row in medium_mean_filtered.iterrows():
        plt.annotate(row["layer_type"], (row["peak_memory"], row["macro_f1"]))
    for i, row in large_mean_filtered.iterrows():
        plt.annotate(row["layer_type"], (row["peak_memory"], row["macro_f1"]))
    plt.savefig("f1_vs_gpu_mean_small.png")
    plt.close()

def plot_gpu_vs_inference_time():
    df = pd.read_csv("full_output_log.csv")
    dl_df = pd.read_csv("detect_lang_official.csv")
    base_mean_df = df.loc[(df["model"]=="base") & (df["aggregation"]=="mean")]
    base_mean_filtered = base_mean_df[
        ~(
            (base_mean_df["layer_index"] > 0) &
            (~base_mean_df["normalization"])
        )
    ]  
    base_mean_filtered = base_mean_filtered[base_mean_filtered['layer_index'].isin([0,6])]

    small_mean_df = df.loc[(df["model"]=="small") & (df["aggregation"]=="mean")]
    small_mean_filtered = small_mean_df[small_mean_df['layer_index'].isin([12])]
    small_mean_filtered = small_mean_filtered[
        ~(
            (small_mean_filtered["layer_index"] > 0) &
            (~small_mean_filtered["normalization"])
        )
    ]  
    small_mean_filtered = small_mean_df[small_mean_df['layer_index'].isin([0,4,6,8,10,12])]

    medium_mean_df = df.loc[(df["model"]=="medium") & (df["aggregation"]=="mean")]
    medium_mean_filtered = medium_mean_df[
        ~(
            (medium_mean_df["layer_index"] > 0) &
            (~medium_mean_df["normalization"])
        )
    ]  
    # medium_mean_filtered = medium_mean_filtered[medium_mean_filtered['layer_index']>0]
    
    large_mean_df = df.loc[(df["model"]=="large-v1") & (df["aggregation"]=="mean")]
    large_mean_filtered = large_mean_df[
        ~(
            (large_mean_df["layer_index"] > 0) &
            (~large_mean_df["normalization"])
        )
    ]  
    # large_mean_filtered = large_mean_filtered[large_mean_filtered['layer_index']>0]


    plt.figure(figsize=(10, 8))
    # plt.scatter(df["peak_memory"], df["inference_time"], color='#25f914', label="MLP with Final Hidden Whisper Embeddings")
    plt.scatter(dl_df["peak_memory"], dl_df["inference_time"], color='#000000', label="Whisper's Built-in Language Detection")
    plt.scatter(base_mean_filtered["peak_memory"], base_mean_filtered["inference_time"], color="#ba19e1", label="MLP on Whisper-base embeddings")
    plt.scatter(small_mean_filtered["peak_memory"], small_mean_filtered["inference_time"], color="#1913a1", label="MLP on Whisper-small embeddings")
    plt.scatter(medium_mean_filtered["peak_memory"], medium_mean_filtered["inference_time"], color="#13ed92", label="MLP on Whisper-medium embeddings")
    plt.scatter(large_mean_filtered["peak_memory"], large_mean_filtered["inference_time"], color="#a1807e", label="MLP on Whisper-large embeddings")
    plt.xlabel("Peak Memory Usage (MB)")
    plt.ylabel("Inference Time (seconds)")
    plt.title("Inference Time vs Peak Memory Usage on Mean Pooled Embeddings from Different Encoder Layers")
    plt.legend()

    # for i, row in df.iterrows():
    #     plt.annotate(row["run"], (row["peak_memory"], row["inference_time"]))
    for i, row in dl_df.iterrows():
        plt.annotate(row["run"], (row["peak_memory"], row["inference_time"]))
    for i, row in base_mean_filtered.iterrows():
        plt.annotate(row["layer_type"], (row["peak_memory"], row["inference_time"]))
    for i, row in small_mean_filtered.iterrows():
        plt.annotate(row["layer_type"], (row["peak_memory"], row["inference_time"]))
    for i, row in medium_mean_filtered.iterrows():
        plt.annotate(row["layer_type"], (row["peak_memory"], row["inference_time"]))
    for i, row in large_mean_filtered.iterrows():
        plt.annotate(row["layer_type"], (row["peak_memory"], row["inference_time"]))


    plt.savefig("inference_time_vs_gpu_mean.png")
    plt.close()

def filter_best_performance():
    df = pd.read_csv("full_output_log.csv")
    # Get the row with max macro_f1 for each model
    top4 = (
        df.sort_values('macro_f1', ascending=False)
        .groupby('model')
        .head(4)
    )

    # top4.to_csv('top4_macro_f1_per_model.csv', index=False)
    # best_rows = df.loc[df.groupby('model')['macro_f1'].idxmax()]

    top4.to_csv('best_macro_f1_per_model.csv', index=False)



    

# plot_large_losses()
# # # plot_f1()
# # plot_f1_vs_inference_time_small()
# plot_medium_losses()
# plot_small_losses()
# plot_base_losses()
# plot_gpu_vs_inference_time()
# plot_f1_base()
plot_f1_vs_inference_time_large()
# plot_f1_medium()
# filter_best_performance()