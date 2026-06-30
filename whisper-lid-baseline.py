import os
import whisper
from sklearn.metrics import classification_report, f1_score, accuracy_score
import pandas as pd
from utils.plotting_utils import  plot_cm
import time
import numpy as np
import torch
import random
import argparse


TEST_DIR = "fleurs_data/test"
model_name = "small"
model = whisper.load_model(model_name)
# whisper model either base, small, medium, large-v2/v1 (v3 uses 128 channels, but out input is only 80)

def run_whisper_lid(dataset_dir=TEST_DIR, model=model, my_languages=None):
    
    y_true = []
    y_pred = []
    
    # print(my_languages)

    if os.path.isfile(dataset_dir):
        path = dataset_dir
        # single file mode for inference time testing
        language = os.path.basename(os.path.dirname(dataset_dir))

        audio = whisper.load_audio(path)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        _, probs = model.detect_language(mel)
        # to get language with max probability, but only if its in our dataset
        predicted_lang = max(
        (lang for lang in probs if lang in my_languages),
        key=probs.get
        )

        y_true.append(language)
        y_pred.append(predicted_lang)

        result = {
            "run": model_name,
            "model": f"whisper_{model_name}_detect_language",
            "hidden_dim": "----",
            "epochs": "----",
            "lr": "----",
        }
        return result, y_true, y_pred

    
    for language in my_languages:
        lang_dir = os.path.join(dataset_dir, language)

        if not os.path.isdir(lang_dir):
            continue

        print(f"Processing {language}")

        for file in os.listdir(lang_dir):

            if not file.endswith((".wav", ".mp3", ".flac")):
                continue

            path = os.path.join(lang_dir, file)

       
            audio = whisper.load_audio(path)
            audio = whisper.pad_or_trim(audio)

            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            _, probs = model.detect_language(mel)
            # to get language with max probability, but only if its in our dataset
            predicted_lang = max(
            (lang for lang in probs if lang in my_languages),
            key=probs.get
            )

            y_true.append(language)
            y_pred.append(predicted_lang)



    labels=sorted(set(y_true))

    # plot confusion matrix with ALL possible predicted labels from whisper
    all_labels = sorted(set(y_true) | set(y_pred))

    report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=labels,
            zero_division=0,
            digits=4
        )

    report_path = f"outputs/whisper_lid/classification_report_whisper_{model_name}.txt"
    with open(report_path, "w") as f:
        f.write(report)


    plot_cm(y_true, y_pred, all_labels, f"Confusion Matrix for Whisper {model_name.capitalize()} Classification Model", f"outputs/whisper_lid/confusion_matrix_whisper_{model_name}.png")

    # add macro_f1 to experiments.csv for comparison
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    result = {
        "run": model_name,
        "model": f"whisper_{model_name}_detect_language",
        "hidden_dim": "----",
        "epochs": "----",
        "lr": "----",
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": macro_f1,
        "weighted_f1": f1_score(y_true, y_pred, average='weighted'),

    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Whisper built-in detect_language function.")
    parser.add_argument("--full-inference", action="store_true", help="Run the full inference on the test set and save results.")
    parser.add_argument("--measure-inference-time", action="store_true", help="Measure the inference time of the model.")
    parser.add_argument("--measure-gpu", action="store_true", help="Measure GPU utilization during inference.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA nicht verfügbar.")
        exit()

    my_languages = set(os.listdir(TEST_DIR))
    if args.measure_inference_time:
        print("Measuring inference time...")
        # still need to do same thing and pick single audio file
        all_files = []
        for root, _, files in os.walk(TEST_DIR):
            for f in files:
                if f.endswith((".wav", ".mp3")):
                    all_files.append(os.path.join(root, f))

        test_file = random.choice(all_files)
        # WICHTIG:
        # Warmup zum Initialisieren/Speicher allokieren...
        # Erster Durchlauf dauert in der Regel länger
        start = time.perf_counter()
        run_whisper_lid(dataset_dir=test_file, model=model, my_languages=my_languages)
        torch.cuda.synchronize()
        end = time.perf_counter()

        print(f"Laufzeit Warmup: {end - start:.3f} s")
        laufzeiten = np.zeros(15)
        for i in range(15):
            # WICHTIG:
            # CUDA arbeitet asynchron.
            # Ohne synchronize() misst man häufig zu wenig Zeit.
            start = time.perf_counter()
            result, y_true, y_pred = run_whisper_lid(dataset_dir=test_file, model=model, my_languages=my_languages)
            torch.cuda.synchronize()

            end = time.perf_counter()
            results_df = pd.DataFrame([result])
            print(f"TRUE: {y_true}, PRED: {y_pred}")

            laufzeiten[i] = end - start
            print(f"Laufzeit ({i}): {end - start:.3f} s")
        print(f"Laufzeiten: {laufzeiten}")
        durschnitt = np.mean(laufzeiten)
        print(f"Durchschnittliche Laufzeit: {durschnitt:.3f} s")

        csvpath = "outputs/compare/detect_lang_official.csv"

        df = pd.read_csv(csvpath)
        if "inference_time" not in df.columns:
            df["inference_time"] = None

        df.loc[df["run"] == model_name, "inference_time"] = durschnitt

        df.to_csv("outputs/compare/detect_lang_official.csv", index=False)


    if args.measure_gpu:
        print("Measuring GPU memory usage during inference...")
        all_files = []
        for root, _, files in os.walk(TEST_DIR):
            for f in files:
                if f.endswith((".wav", ".mp3")):
                    all_files.append(os.path.join(root, f))

        test_file = random.choice(all_files)

        # warmup
        for _ in range(3):
            result = run_whisper_lid(dataset_dir=test_file, model=model, my_languages=my_languages)

        
        torch.cuda.empty_cache()
        torch.cuda.memory.reset_peak_memory_stats()
        torch.cuda.synchronize()
        # call gpu_function aka our inference on one file
        result = run_whisper_lid(dataset_dir=test_file, model=model, my_languages=my_languages)
        torch.cuda.synchronize()
        gpu_stats = torch.cuda.memory.memory_stats() # ->allocated_bytes peak
        peak_mem = gpu_stats["allocated_bytes.all.peak"] / (1024 ** 2) # in MB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2) # in MB
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2) # in MB

        print(f"Peak GPU Memory Usage: {peak_mem:.2f} MB")
        print(f"Max Allocated Memory: {max_allocated:.2f} MB")
        print(f"Max Reserved Memory: {max_reserved:.2f} MB")    

        csvpath = "outputs/compare/detect_lang_official.csv"

        df = pd.read_csv(csvpath)
        if "peak_memory" not in df.columns:
            df["peak_memory"] = None

        df.loc[df["run"] == model_name, "peak_memory"] = peak_mem

        df.to_csv("outputs/compare/detect_lang_official.csv", index=False)



    if args.full_inference:
        print("Running full inference on test set...")
        result = run_whisper_lid(dataset_dir=TEST_DIR, model=model, my_languages=my_languages)
        results_df = pd.DataFrame([result])
        csvpath = "outputs/compare/detect_lang_official.csv"

        if os.path.exists(csvpath):
            results_df.to_csv(csvpath, mode='a', header=False, index=False)
        else:
            results_df.to_csv(csvpath, index=False)
