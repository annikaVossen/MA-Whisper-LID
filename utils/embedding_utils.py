import os

import torch
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperModel, WhisperFeatureExtractor
import time

os.environ["LINE_PROFILE"] = "1"
from line_profiler import profile
from config import MODEL_ID


import inspect



class WhisperEmbedder:
    def __init__(self, model_id, device):
        self.device = device
        self.fe = WhisperFeatureExtractor.from_pretrained(model_id)
        self.model = WhisperModel.from_pretrained(model_id).to(device)
        self.model.eval()


    # @profile
    def get_embedding(self, audio_array):
        inputs = self.fe(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.inference_mode():
            features = inputs.input_features
            encoder_outputs = self.model.encoder(
                features.to(self.device, non_blocking=True), 
                output_hidden_states=True
            )

        embedding = encoder_outputs.last_hidden_state.mean(dim=1)  # change this for earlier embeddings, and to get different aggregation strategies

        return embedding.squeeze(0).cpu().numpy()
    
    def whisper_encoder_until_layer(self, input_features, stop_layer):
        encoder = self.model.encoder
        # print(torch.is_grad_enabled())
        # frontend 
        x = encoder.conv1(input_features)
        x = F.gelu(x)
        # print(f"CONV1 SHAPE: {x.shape}")
        if stop_layer == -1: 
            # print("Stopped after conv1")
            return x

        x = encoder.conv2(x)
        x = F.gelu(x)
        # print(f"CONV2 SHAPE: {x.shape}")

        if stop_layer == 0: # stop here
            # print("Stopped after conv2")
            return x

        # (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)

        # positional embedding
        x = x + encoder.embed_positions.weight

        # transformer layers 
        count=0
        for i,layer in enumerate(encoder.layers[:stop_layer]):
            count+=1
            out = layer(x, attention_mask=None, layer_head_mask=None, output_attentions=False)  # ignoring masks for now, do I need them?
            x = out[0]
            del out
            # print(i, x.shape)

            # if i >= stop_layer - 1: # 0 indexing
            #     # print(f"Stopped afer layer: {i}")
            #     break
            # print("executed", count)

        # final layer norm is normally applied AFTER all layers
        # optional normalization of embedding
        x = encoder.layer_norm(x)

        return x
    
    def get_embedding_inter(self, audio_array, stop_layer=32):
        inputs = self.fe(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )
        with torch.inference_mode():
            features = inputs.input_features.to(self.device)
            layer_embedding = self.whisper_encoder_until_layer(
                features,
                stop_layer=stop_layer   # layer 3 = 4th layer (0-indexed)
            )

        embedding = layer_embedding.mean(dim=1) # mean aggregation technique for now
        return embedding.squeeze(0).cpu().numpy()
    

    def test_timing(self, audio_array):
        for stop_layer in [1,1,1, 2, 4, 6]:
            inputs = self.fe(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )
            with torch.inference_mode():
                input_features = inputs.input_features.to(self.device)
                torch.cuda.synchronize()
                start = time.perf_counter()

                _ = self.whisper_encoder_until_layer(input_features, stop_layer)

                torch.cuda.synchronize()
                end = time.perf_counter()

                print(stop_layer, end - start)
    

def save_model_structure(model, path):
    with open(path, "w") as f:
        for name, module in model.named_modules():
            indent = name.count(".")
            f.write(f"{'  '*indent}{name or 'root'} -> {module.__class__.__name__}\n")

# test stuff 
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # print(device)
# embedder = WhisperEmbedder("openai/whisper-base", device)
# model = embedder.model
# print(model.encoder)

# for name, module in model.encoder.named_children():
#     print(name, type(module))
# with open("encoder_forward.txt", "w") as f:
#     f.write(inspect.getsource(model.encoder.forward))

# save_model_structure(model.encoder, "whisper_base.txt")
# embedder = WhisperEmbedder("openai/whisper-small", device)
# model = embedder.model
# save_model_structure(model.encoder, "whisper_small.txt")
# embedder = WhisperEmbedder("openai/whisper-medium", device)
# model = embedder.model
# save_model_structure(model.encoder, "whisper_medium.txt")
# embedder = WhisperEmbedder("openai/whisper-large", device)
# model = embedder.model
# save_model_structure(model.encoder, "whisper_large.txt")

# print(inspect.signature(
#     embedder.model.encoder.layers[0].forward
# ))

