import argparse
import numpy as np
from beispiele import gpu_function
import torch, torchaudio
import os
import random

from sklearn.preprocessing import LabelEncoder
import config

from models.mlp_model import LanguageMLP, train_model
from evaluate_saved_model import evaluate_and_plot_model

# from utils.embedding_utils import WhisperEmbedder
from utils.save_utils import load_label_encoder, save_label_encoder, save_model, load_model

import librosa
import time
import whisper


random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

import pandas as pd

from audio_preprocessing import audio_to_mel, get_embedding_inter, build_embeddings
import gc
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_partial_encoder(model_id, stop_layer, device):
    # load everything to CPU first - no GPU memory used yet
    model = whisper.load_model(model_id, download_root='whisper_models', device='cpu')
    encoder = model.encoder

    if stop_layer == -1:  # conv1 only
        del encoder.conv2
        del encoder.blocks
        del encoder.positional_embedding
        del encoder.ln_post
        print("Keeping only first convolution layer")
    elif stop_layer == 0:  # conv1 + conv2
        del encoder.blocks
        del encoder.positional_embedding
        del encoder.ln_post
        print("Keeping only both convolution layers")
    else:
        full_blocks = len(model.encoder.blocks)
        # conv layers + positional embedding + N transformer blocks + ln_post
        encoder.blocks = encoder.blocks[:stop_layer]
        print(f"Keeping {stop_layer} of {full_blocks} transformer blocks")


    encoder = encoder.to(device)
    
    # explicitly free the rest
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return encoder


def train_and_evaluate(data_train, data_test, device, model):


    X_train, y_train = build_embeddings(data_train, max_per_lang=config.MAX_PER_LANG, tt='train', reuse_embeddings=False, model=model)
    X_test, y_test = build_embeddings(data_test, tt='test', reuse_embeddings=False, model=model)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_enc, dtype=torch.long)

    model = LanguageMLP(
        input_dim=X_train.shape[1],
        hidden_dim=config.HIDDEN_DIM,
        num_classes=len(le.classes_)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    losses = train_model(
        model,
        optimizer,
        X_train_t,
        y_train_t,
        device,
        epochs=config.EPOCHS
    )

    np.savetxt("outputs/compare/losses_" + config.RUN_NAME + ".txt", losses)

    os.makedirs("outputs/checkpoints", exist_ok=True)
    save_model(model, "outputs/checkpoints/mlp_model_" + config.RUN_NAME + ".pt")
    save_label_encoder(le, "outputs/checkpoints/label_encoder_"  + config.RUN_NAME + ".pkl")

    evaluate_and_plot_model(model, le, data_test, device, output_dir="outputs", X_test=X_test, y_test=y_test)

    _, _, _, _, results_df = evaluate_and_plot_model(model, le, data_test, device, output_dir="outputs", X_test=X_test, y_test=y_test)
    return results_df

# @profile
def evaluate_only(data_test, device, checkpoint_path="outputs/checkpoints/mlp_model_" + config.RUN_NAME + ".pt", encoder_path="outputs/checkpoints/label_encoder_"  + config.RUN_NAME + ".pkl", reuse_embeddings=True, timing=False, whisper_model=None):

    le = load_label_encoder(encoder_path)

    # embedder = WhisperEmbedder(MODEL_ID, device)
    # print("before even getting the embedding??", torch.cuda.max_memory_allocated() / 1024**2)

    X_test, y_test = build_embeddings(data_test, tt='test', model=whisper_model, reuse_embeddings=reuse_embeddings)

    model = LanguageMLP(
        input_dim=X_test.shape[1],
        hidden_dim=config.HIDDEN_DIM,
        num_classes=len(le.classes_)
    ).to(device)

    mlp_model = load_model(model, checkpoint_path, device)
    mlp_model.eval()

    _, _, _, _, results_df = evaluate_and_plot_model(mlp_model, le, data_test, device, output_dir="outputs", X_test=X_test, y_test=y_test, timing=timing)
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and/or evaluate the Whisper language MLP model.")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training and evaluate an existing saved model.")
    parser.add_argument("--measure-inference-time", action="store_true", help="Measure the inference time of the model.")
    parser.add_argument("--measure-gpu", action="store_true", help="Measure GPU utilization during inference.")
    parser.add_argument("--train", action="store_true", help="Training model")
    parser.add_argument("--test-one", action="store_true", help="Testing on a single audio file")
    parser.add_argument("--stop-layer", type=float, default=config.STOP_LAYER)
    parser.add_argument("--run-name", type=str, default=config.RUN_NAME)
    parser.add_argument('--model', type=str, default=config.MODEL_ID)
    

    args = parser.parse_args()

    if args.stop_layer is not None:
        config.STOP_LAYER = int(args.stop_layer)

    if args.run_name is not None:
        config.RUN_NAME = args.run_name

    if args.model is not None:
        config.MODEL_ID = args.model
    model = load_partial_encoder(config.MODEL_ID, config.STOP_LAYER, device)   # problem! what if config.MODEL_ID is modified in sweep???
    print(f"Successfully loaded {config.MODEL_ID} model")
    print(next(model.parameters()).device)

    if args.evaluate_only:
        print("Evaluating saved model...")
        results_df = evaluate_only(config.TEST_DIR, device, checkpoint_path="outputs/checkpoints/mlp_model_" + config.RUN_NAME + ".pt", encoder_path="outputs/checkpoints/label_encoder_"  + config.RUN_NAME + ".pkl", reuse_embeddings=False, whisper_model=model)
        csvpath = "outputs/compare/full_output_log.csv"
    
        if os.path.exists(csvpath):
            results_df.to_csv(csvpath, mode='a', header=False, index=False)
        else:
            results_df.to_csv(csvpath, index=False)

    if args.measure_inference_time:
        print("Measuring inference time...")
        if not torch.cuda.is_available():
            print("CUDA nicht verfügbar.")
            exit()
        # get a single item for testing the inference time 
        all_files = []

        for root, _, files in os.walk(config.TEST_DIR):
            for f in files:
                if f.endswith((".wav", ".mp3")):
                    all_files.append(os.path.join(root, f))

        test_file = random.choice(all_files)

        results = evaluate_only(test_file, device, checkpoint_path="outputs/checkpoints/mlp_model_" + config.RUN_NAME + ".pt", encoder_path="outputs/checkpoints/label_encoder_"  + config.RUN_NAME + ".pkl", reuse_embeddings=False, timing=True, whisper_model=model)
        # warmup
        # WICHTIG:
        # Warmup zum Initialisieren/Speicher allokieren...
        # Erster Durchlauf dauert in der Regel länger
        start = time.perf_counter()
        evaluate_only(test_file, device, checkpoint_path="outputs/checkpoints/mlp_model_" + config.RUN_NAME + ".pt", encoder_path="outputs/checkpoints/label_encoder_"  + config.RUN_NAME + ".pkl", reuse_embeddings=False, timing=True, whisper_model=model)
       
        torch.cuda.synchronize()
        end = time.perf_counter()

        print(f"Laufzeit Warmup: {end - start:.3f} s")
        laufzeiten = np.zeros(15)
        # need to add in a part about not saving the embeddings here each round
        for i in range(15):
            test_file = random.choice(all_files)
            # WICHTIG:
            # CUDA arbeitet asynchron.
            # Ohne synchronize() misst man häufig zu wenig Zeit.
            start = time.perf_counter()
            results = evaluate_only(test_file, device, checkpoint_path="outputs/checkpoints/mlp_model_" + config.RUN_NAME + ".pt", encoder_path="outputs/checkpoints/label_encoder_"  + config.RUN_NAME + ".pkl", reuse_embeddings=False, timing=True, whisper_model=model)
            torch.cuda.synchronize()
            end = time.perf_counter()
            laufzeiten[i] = end - start
            print(f"Laufzeit ({i}): {end - start:.3f} s")
        print(f"Laufzeiten: {laufzeiten}")
        durschnitt = np.mean(laufzeiten)
        print(f"Durchschnittliche Laufzeit: {durschnitt:.3f} s")
        ## to save inference time to corresponding row in csv
        csvpath = "outputs/compare/full_output_log.csv"

        df = pd.read_csv(csvpath)
        if "inference_time" not in df.columns:
            df["inference_time"] = None

        df.loc[df["run_name"] == config.RUN_NAME, "inference_time"] = durschnitt

        df.to_csv("outputs/compare/full_output_log.csv", index=False)

    if args.measure_gpu:
        

        print(f"Measuring GPU with {config.RUN_NAME}")
        if not torch.cuda.is_available():
            print("CUDA nicht verfügbar.")
            exit()
        # get a single item for testing the inference time 
        all_files = []

        for root, _, files in os.walk(config.TEST_DIR):
            for f in files:
                if f.endswith((".wav", ".mp3")):
                    all_files.append(os.path.join(root, f))

        test_file = random.choice(all_files)
        
        # warmup
        # for _ in range(3):
        #     evaluate_only(test_file, device, checkpoint_path="outputs/checkpoints/mlp_model_" + config.RUN_NAME + ".pt", encoder_path="outputs/checkpoints/label_encoder_"  + config.RUN_NAME + ".pkl", reuse_embeddings=False, timing=True)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.memory.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print("when we start", torch.cuda.max_memory_allocated() / 1024**2)
        # call gpu_function aka our inference on one file
        results = evaluate_only(test_file, device, checkpoint_path="outputs/checkpoints/mlp_model_" + config.RUN_NAME + ".pt", encoder_path="outputs/checkpoints/label_encoder_"  + config.RUN_NAME + ".pkl", reuse_embeddings=False, timing=True, whisper_model=model)

        torch.cuda.synchronize()
        gpu_stats = torch.cuda.memory.memory_stats() # ->allocated_bytes peak
        peak_mem = gpu_stats["allocated_bytes.all.peak"] / (1024 ** 2) # in MB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2) # in MB
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2) # in MB

        print(f"Peak GPU Memory Usage: {peak_mem:.2f} MB")
        print(f"Max Allocated Memory: {max_allocated:.2f} MB")
        print(f"Max Reserved Memory: {max_reserved:.2f} MB")    

        csvpath = "outputs/compare/full_output_log.csv"

        df = pd.read_csv(csvpath)
        if "peak_memory" not in df.columns:
            df["peak_memory"] = None

        df.loc[df["run_name"] == config.RUN_NAME, "peak_memory"] = peak_mem
        df.loc[df["run_name"] == config.RUN_NAME, "max_allocated"] = max_allocated
        df.loc[df["run_name"] == config.RUN_NAME, "max_reserved"] = max_reserved
        



        df.to_csv("outputs/compare/full_output_log.csv", index=False)
        # held = []
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) and obj.is_cuda:
        #             mb = obj.element_size() * obj.nelement() / 1024**2
        #             if mb > 0.1:  # catch everything >100 KB
        #                 held.append((obj.shape, mb))
        #     except:
        #         pass

        # held.sort(key=lambda x: -x[1])
        # total = sum(mb for _, mb in held)
        # print(f"Total tracked: {total:.1f} MB across {len(held)} tensors")
        # for shape, mb in held[:30]:
        #     print(f"  {mb:.1f} MB  {shape}")
    
    if args.train:
        print(f"Training and evaluating {config.RUN_NAME} model with {config.STOP_LAYER} layers and {config.AGGR} aggregation")
        
        results_df = train_and_evaluate(config.TRAIN_DIR, config.TEST_DIR, device, model=model)
        csvpath = "outputs/compare/full_output_log.csv"

        if os.path.exists(csvpath):
            results_df.to_csv(csvpath, mode='a', header=False, index=False)
        else:
            results_df.to_csv(csvpath, index=False)

    if args.test_one:
        if not torch.cuda.is_available():
            print("CUDA nicht verfügbar.")
            exit()
        # get a single item for testing the inference time 
        all_files = []

        for root, _, files in os.walk(config.TEST_DIR):
            for f in files:
                if f.endswith((".wav", ".mp3")):
                    all_files.append(os.path.join(root, f))

        test_file = random.choice(all_files)
        lang = os.path.basename(os.path.dirname(test_file))
        get_embedding_inter(test_file, aggr=config.AGGR)
     


