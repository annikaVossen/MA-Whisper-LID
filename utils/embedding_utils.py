import torch
from transformers import WhisperProcessor, WhisperModel


class WhisperEmbedder:
    def __init__(self, model_id, device):
        self.device = device

        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperModel.from_pretrained(model_id).to(device)
        self.model.eval()

    def get_embedding(self, audio_array):
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.no_grad():
            encoder_outputs = self.model.encoder(
                inputs["input_features"].to(self.device)
            )

        embedding = encoder_outputs.last_hidden_state.mean(dim=1)

        return embedding.squeeze(0).cpu().numpy()