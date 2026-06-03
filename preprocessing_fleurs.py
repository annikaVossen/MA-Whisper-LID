import librosa
from datasets import load_dataset
import soundfile as sf
import os
import random


def load_audio(path):
    audio, sr = librosa.load(path, sr=16000, mono=True)
    return audio


def build_dataset(data_dir, embedder, max_per_lang=50):
    X, y = [], []

    for lang in os.listdir(data_dir):
        lang_folder = os.path.join(data_dir, lang)

        if not os.path.isdir(lang_folder):
            continue

        files = [f for f in os.listdir(lang_folder)
                 if f.endswith((".wav", ".mp3"))]

        selected = random.sample(files, min(max_per_lang, len(files)))

        for file in selected:
            path = os.path.join(lang_folder, file)

            try:
                from data import load_audio
                audio = load_audio(path)
                emb = embedder.get_embedding(audio)

                X.append(emb)
                y.append(lang)

            except Exception as e:
                print(f"Skipping {path}: {e}")

    return X, y

def save_fleurs_language(lang_code, save_root, max_samples):
    print(f"Loading {lang_code}...")

    ds_train = load_dataset(
        "google/fleurs",
        lang_code,
        split="train",
        streaming=True
    )

    ds_test = load_dataset(
        "google/fleurs",
        lang_code,
        split="test",
        streaming=True
    )

    save_root_train = os.path.join(save_root, "train")
    lang_dir_train = os.path.join(save_root_train, lang_code)

    save_root_test = os.path.join(save_root, "test")
    lang_dir_test = os.path.join(save_root_test, lang_code)

    
    
    os.makedirs(lang_dir_train, exist_ok=True)

    for i, sample in enumerate(ds_train):
        path = os.path.join(lang_dir_train, f"{i}.wav")

        if os.path.exists(path):
            continue
        
        if i >= max_samples:
            break

        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]

        # trim to 5 seconds (important for speed/storage)
        audio = audio[: 5 * sr]

        path = os.path.join(lang_dir_train, f"{i}.wav")
        sf.write(path, audio, sr)

    os.makedirs(lang_dir_test, exist_ok=True)

    for i, sample in enumerate(ds_test):
        path = os.path.join(lang_dir_test, f"{i}.wav")

        if os.path.exists(path):
            continue
        
        if i >= max_samples:
            break

        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]

        # trim to 5 seconds (important for speed/storage)
        audio = audio[: 5 * sr]

        path = os.path.join(lang_dir_test, f"{i}.wav")
        sf.write(path, audio, sr)

    print(f"Saved {i+1} samples for {lang_code}")



# languages = ["en_us", "de_de", "es_419", "fr_fr", "pl_pl", "ar_eg", "uk_ua", "bn_in", ""da_dk","el_gr", "fa_ir",
# languages = [ "fi_fi", "hi_in", "hu_hu", "id_id", "it_it", "ja_jp", "nl_nl", "pt_br", "ro_ro", "ru_ru", "sv_se", "ta_in", "tr_tr", "cmn_hans_cn"]
for lang in languages:
    save_fleurs_language(lang, save_root="fleurs_data", max_samples=5000)  # most languages seem to max out around 600-1000 sammples
