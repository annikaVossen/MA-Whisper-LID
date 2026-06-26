import config
import main
import subprocess
import csv
# large: 20-31 minus 24
def iter_train():
    layers = [20,21,22,23,25,26,27,28,29,30,31]
    # run_names = 
    for i in range(len(layers)):
        subprocess.run([
            "python",
            "main.py",
            "--run-name", "largev1_mean-" + str(layers[i]) + "layers",
            "--stop-layer", str(layers[i]),
            "--train"

        ], check=True)

def iter_gpu(csv_path= "outputs/compare/full_output_log.csv"):
    # iteratively checking gpu
    # first conv1, then conv2, then all the layers
    # iterate through csv !!

    MODEL_ORDER = {
    "base": 0,
    "small": 1,
    "medium": 2,
    "large-v1": 3,
    }

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Sort first by layer_index (int), then by custom model order
    rows.sort(
    key=lambda row: (
        int(float(row["layer_index"])),
        MODEL_ORDER.get(row["model"], float("inf")),
        )
    )

    # Run each experiment
    for row in rows:
        print(
            f"Running: layer={row['layer_index']}, "
            f"model={row['model']}, "
            f"run_name={row['run_name']}"
        )

        subprocess.run(
            [
                "python",
                "main.py",
                "--run-name",
                row["run_name"],
                "--stop-layer",
                row["layer_index"],
                "--model",        # add arg
                row["model"],
                # "--normalization", # add arg 
                # row['normalization']
                "--measure-gpu",
            ],
            check=True,
        )


# layer_indices = np.arange(-1, 33)

# print(layer_indices)
# for layer in layer_indices:
#     models = ["large-v1"]
#     if layer <= 24:
#         models.append("medium")
#         if layer <= 12:
#             models.append("small")
#             if layer <= 6:
#                 models.append("base")
#     for model in models.reverse():
#         subprocess.run([
#         "python",
#         "main.py",
#         "--run-name", "largev1_mean-" + str(layers[i]) + "layers",  # run names unfortunately differ a lot
#         "--stop-layer", str(layer),
#         "--model-id", model, # ADD THIS ARGUMENT TO MAIN
#         "--measure-gpu", 

#         ], check=True)'



iter_gpu()