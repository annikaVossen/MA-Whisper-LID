import config
import main
import subprocess
import csv
from itertools import groupby
from operator import itemgetter
# large: 20-31 minus 24
def iter_train():
    # layers = [15,17,18,19,20,21,22,23]
    layers = [16]
    run_names = ["large_mean-16layers"]
    # run_names = 
    for i in range(len(layers)):
        subprocess.run([
            "python",
            "main.py",
            "--run-name", run_names[i],
            "--stop-layer", str(layers[i]),
            "--model", "large-v1",
            "--evaluate-only"

        ], check=True)


def normalize_layer_index(csv_path="outputs/compare/full_output_log.csv"):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        row["layer_index"] = int(float(row["layer_index"]))

    # Write rows back to the same (or different) CSV file
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def iter_train_max_aggr(csv_path="outputs/compare/full_output_log.csv"):

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)


    # csv values come in as strings, so convert the columns 
    for row in rows:
        row['layer_index'] = int(row['layer_index'])
        row['macro_f1'] = float(row['macro_f1'])

    # sort by model, then layer_index (groupby requires sorted input by the group key)
    rows.sort(key=lambda r: (r['model'], r['layer_index']))

    for model, group in groupby(rows, key=itemgetter('model')):
        group = list(group)  # group is an iterator, materialize it so we can scan it
        started = False
        for row in group:
            if not started and row['macro_f1'] > 0.65:
                started = True
            if started:
                # process this row
                run_name = row["model"] + "_" + str(row["layer_index"]) + "_max"
                subprocess.run([
                "python",
                "main.py",
                "--run-name", run_name,
                "--stop-layer", str(int(row["layer_index"])),
                "--model", row["model"],
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
                "--measure-gpu", "--measure-inference-time"
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


normalize_layer_index()
iter_train_max_aggr()