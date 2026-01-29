
import torch
import torch.nn as nn
from pocket_tts.models.mimi import MimiModel
from onnx_export.export_utils import unflatten_state, flatten_state
from pocket_tts.modules.stateful_module import increment_steps

class MimiWrapper(nn.Module):
    def __init__(self, mimi: MimiModel, state_structure):
        super().__init__()
        self.mimi = mimi
        self.state_structure = state_structure
        
    def forward(self, input, *flat_state):
        try:
            # Un-normalize latent: scale and shift back
            # mimi_decoding_input = latent * self.emb_std + self.emb_mean
            
            # Transpose: [B, T, D] -> [B, D, T]
            transposed = input # latent.transpose(-1, -2)
            
            model_state, _ = unflatten_state(flat_state, self.state_structure)

            # Project: [B, dim, 1]
            quantized = self.mimi.quantizer.decode(transposed)
            
            # Decode
            audio_frame = self.mimi.decode_from_latent(quantized, model_state)

            if torch.jit.is_tracing():
                from torch.onnx import operators
                seq_len = operators.shape_as_tensor(input)[2]
            else:
                seq_len = input.shape[2]
            
            # Increment by the hop factor (25Hz transformer / 12.5Hz latent = 2)
            increment = seq_len * 2
            increment_steps(self.mimi, model_state, increment=increment)
            
            new_flat_state = flatten_state(model_state)
            
            return (audio_frame, *new_flat_state)
        except Exception as e:
            print(f"Error in MimiWrapper forward: {e}")
            raise e


class MimiEncoderWrapper(nn.Module):
    """Wrapper for Mimi encoder that takes raw audio and returns latent embeddings."""
    def __init__(self, mimi: MimiModel, state_structure):
        super().__init__()
        self.mimi = mimi
        self.state_structure = state_structure

    def forward(self, input, *flat_state):
        model_state, _ = unflatten_state(flat_state, self.state_structure)

        # audio: [B, C, T] -> latent: [B, T', D]
        encoded = self.mimi.encode_to_latent(input, model_state)
        # encoded is [B, D, T'], we need [B, T', D]
        latents = encoded # encoded.transpose(-1, -2)
        
        seq_len = input.shape[1]
        increment = seq_len * 2
        increment_steps(self.mimi, model_state, increment=increment)
    
        new_flat_state = flatten_state(model_state)
        
        return (latents, *new_flat_state)


class TextConditionerWrapper(nn.Module):
    """Wrapper for text conditioner that takes token IDs and returns embeddings."""
    def __init__(self, conditioner):
        super().__init__()
        self.conditioner = conditioner

    def forward(self, token_ids):
        # token_ids: [B, T] -> embeddings: [B, T, D]
        from pocket_tts.conditioners.base import TokenizedText
        return self.conditioner(TokenizedText(token_ids))
