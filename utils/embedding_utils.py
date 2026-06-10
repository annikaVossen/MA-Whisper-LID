import os

import torch
from transformers import WhisperProcessor, WhisperModel, WhisperFeatureExtractor

os.environ["LINE_PROFILE"] = "1"
from line_profiler import profile

class WhisperEmbedder:
    def __init__(self, model_id, device):
        self.device = device
        self.fe = WhisperFeatureExtractor.from_pretrained(model_id)
        self.model = WhisperModel.from_pretrained(model_id).to(device)
        self.model.eval()


    @profile
    def get_embedding(self, audio_array):
        inputs = self.fe(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.inference_mode():
            features = inputs.input_features
            encoder_outputs = self.model.encoder(
                features.to(self.device, non_blocking=True)
            )

        embedding = encoder_outputs.last_hidden_state.mean(dim=1)  # change this for earlier embeddings?

        return embedding.squeeze(0).cpu().numpy()