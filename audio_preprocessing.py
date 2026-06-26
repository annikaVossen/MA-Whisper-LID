import whisper
import torch
import torch.nn.functional as F
import torch.nn as nn

import inspect

import sys

# setting path
from config import MODEL_ID, TEST_DIR
import random
import os
import numpy as np

model = whisper.load_model(MODEL_ID)
device = model.device
def load_audio_whisper_style(path):
    audio = whisper.load_audio(path)          # load + resample to 16 kHz
    # print("whisper load", torch.cuda.max_memory_allocated() / 1024**2)
    audio = whisper.pad_or_trim(audio)        # 30s crop/pad 
    return audio


def audio_to_mel(path):
    audio = load_audio_whisper_style(path)
    # print("loaded audio", torch.cuda.max_memory_allocated() / 1024**2)
    # log-mel spectrogram the way whisper does it
    mel = whisper.log_mel_spectrogram(audio).unsqueeze(0)

    # move to model device + dtype match
    mel = mel.to(device)

    return mel

def whisper_encoder_until_layer(input_features, stop_layer):
        encoder = model.encoder
        # print("encoder init", torch.cuda.max_memory_allocated() / 1024**2)

        # print(torch.is_grad_enabled())
        # frontend 
        x = encoder.conv1(input_features)
        x = F.gelu(x)
        # print("conv1", torch.cuda.max_memory_allocated() / 1024**2)

        # print(f"CONV1 SHAPE: {x.shape}")
        if stop_layer == -1: 
            # print("Stopped after conv1")
            return x

        x = encoder.conv2(x)
        x = F.gelu(x)
        # print(f"CONV2 SHAPE: {x.shape}")
        # print("conv2", torch.cuda.max_memory_allocated() / 1024**2)

        if stop_layer == 0: # stop here
            # print("Stopped after conv2")
            return x

        # (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)
        # print("permute", torch.cuda.max_memory_allocated() / 1024**2)
        # positional embedding
        x = (x + encoder.positional_embedding).to(x.dtype)

        # print("positional embedding", torch.cuda.max_memory_allocated() / 1024**2)
        # transformer layers 
        # transformer blocks
        for i, layer in enumerate(encoder.blocks):
            x = layer(x)
            # print(f"transformer layer {i}", torch.cuda.max_memory_allocated() / 1024**2)
            # return intermediate layer output
            if i + 1 == stop_layer and stop_layer < len(encoder.blocks):
                x = encoder.ln_post(x)
                return x

        # only reached when all blocks were run
        x = encoder.ln_post(x)
        return x

def get_embedding_inter(audio_path, stop_layer, aggr="mean"):
    # print("before loading audio", torch.cuda.max_memory_allocated() / 1024**2)
    features = audio_to_mel(audio_path)
    # print("after loading audio", torch.cuda.max_memory_allocated() / 1024**2)
    
    with torch.inference_mode():
        layer_embedding = whisper_encoder_until_layer(
            features,
            stop_layer=stop_layer   # layer 3 = 4th layer (0-indexed)
        )
    # print("after whisper encoder until layer", torch.cuda.max_memory_allocated() / 1024**2)
# 
    if aggr == "mean":
        embedding = layer_embedding.mean(dim=1) # mean aggregation technique for now
    elif aggr == "max":
        embedding = layer_embedding.max(dim=1).values
    elif aggr == "mean + max":
        mean_emb = layer_embedding.mean(dim=1)
        max_emb = layer_embedding.max(dim=1)
        embedding = torch.cat([mean_emb, max_emb], dim=-1)
    elif aggr == "mean + std":
        mean_emb = layer_embedding.mean(dim=1)
        std_emb = layer_embedding.std(dim=1)
        embedding = torch.cat([mean_emb, std_emb], dim=-1)
  
        

    # elif aggr == "conv1d + global max pooling":
    #     conv = nn.Conv1d(
    #         in_channels=hidden_size,
    #         out_channels=256,
    #         kernel_size=5,
    #         padding=2
    #     )
    #     x = layer_embedding.transpose(1, 2)  # [B,H,S]
    #     x = torch.relu(conv(x))

    #     embedding = x.max(dim=2).values

    # elif aggr == "attention pooling":
    #     self.attn = nn.Linear(hidden_size, 1)

    #     scores = self.attn(layer_embedding)      # [B,S,1]
    #     weights = torch.softmax(scores, dim=1)

    #     embedding = (weights * layer_embedding).sum(dim=1)

    return embedding.squeeze(0).cpu().numpy()

    

def save_model_structure(model, path):
    with open(path, "w") as f:
        for name, module in model.named_modules():
            indent = name.count(".")
            f.write(f"{'  '*indent}{name or 'root'} -> {module.__class__.__name__}\n")


# test
if __name__ == "__main__":
    all_files = []

    for root, _, files in os.walk(TEST_DIR):
        for f in files:
            if f.endswith((".wav", ".mp3")):
                all_files.append(os.path.join(root, f))

    test_file = random.choice(all_files)
    lang = os.path.basename(os.path.dirname(test_file))

    # mel = audio_to_mel(test_file)

    from utils.embedding_utils import WhisperEmbedder
    import librosa 

    embedder = WhisperEmbedder("openai/whisper-base", device)
    audio, _ = librosa.load(test_file, sr=16000, mono=True)


    with torch.no_grad():
        x1 = embedder.get_embedding_inter(audio, stop_layer=4)

        x2 = get_embedding_inter(
            test_file,
            stop_layer=4
        )

    print("old:", x1.shape)
    print("new:", x2.shape)
    diff = np.abs(x1 - x2)

    print("max abs diff:", diff.max())
    print("mean abs diff:", diff.mean())

    # new methods shows negligible difference to old! yay!

    # with torch.no_grad():
    #     x1 = F.gelu(model.encoder.conv1(mel))
    #     x2 = F.gelu(model.encoder.conv1(mel))

    #     x1 = F.gelu(model.encoder.conv2(x1))
    #     x2 = F.gelu(model.encoder.conv2(x2))

    #     x1 = x1.permute(0,2,1)
    #     x2 = x2.permute(0,2,1)

    #     x1 = (x1 + model.encoder.positional_embedding).to(x1.dtype)
    #     x2 = (x2 + model.encoder.positional_embedding).to(x2.dtype)

    #     for i, block in enumerate(model.encoder.blocks):
    #         x1 = block(x1)
    #         x2 = model.encoder.blocks[i](x2)

    #         diff = (x1 - x2).abs().max()
    #         print(i, diff.item())

