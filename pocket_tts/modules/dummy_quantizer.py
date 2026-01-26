import torch
from torch import nn


class MimiEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""

    def __init__(self, config: MimiConfig, epsilon: float = 1e-5):
        super().__init__()
        embed = torch.zeros(config.codebook_size, config.codebook_dim)

        self.codebook_size = config.codebook_size

        self.register_buffer("initialized", torch.tensor([True], dtype=torch.float32))
        self.register_buffer("cluster_usage", torch.ones(config.codebook_size))
        self.register_buffer("embed_sum", embed)
        self._embed = None
        self.epsilon = epsilon

    @property
    def embed(self) -> torch.Tensor:
        if self._embed is None:
            self._embed = self.embed_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        return self._embed

    def quantize(self, hidden_states):
        # Projects each vector in `hidden_states` over the nearest centroid and return its index.
        # `hidden_states` should be `[N, D]` with `N` the number of input vectors and `D` the dimension.
        dists = torch.cdist(hidden_states[None].float(), self.embed[None].float(), p=2)[0]
        embed_ind = dists.argmin(dim=-1)
        return embed_ind

    # Copied from transformers.models.encodec.modeling_encodec.EncodecEuclideanCodebook.encode
    def encode(self, hidden_states):
        shape = hidden_states.shape
        # pre-process
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        # quantize
        embed_ind = self.quantize(hidden_states)
        # post-process
        embed_ind = embed_ind.view(*shape[:-1])
        return embed_ind

    # Copied from transformers.models.encodec.modeling_encodec.EncodecEuclideanCodebook.decode
    def decode(self, embed_ind):
        quantize = nn.functional.embedding(embed_ind, self.embed)
        return quantize


# Copied from transformers.models.encodec.modeling_encodec.EncodecVectorQuantization with Encodec->Mimi
class MimiVectorQuantization(nn.Module):
    """
    Vector quantization implementation. Currently supports only euclidean distance.
    """

    def __init__(self, config: MimiConfig):
        super().__init__()
        self.codebook = MimiEuclideanCodebook(config)

    def encode(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1)
        embed_in = self.codebook.encode(hidden_states)
        return embed_in

    def decode(self, embed_ind):
        quantize = self.codebook.decode(embed_ind)
        quantize = quantize.permute(0, 2, 1)
        return quantize


class MimiResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer."""

    def __init__(self, config: MimiConfig, num_quantizers: int | None = None):
        super().__init__()
        self.codebook_size = config.codebook_size
        self.frame_rate = config.frame_rate
        self.num_quantizers = num_quantizers if num_quantizers is not None else config.num_quantizers
        self.layers = nn.ModuleList([MimiVectorQuantization(config) for _ in range(self.num_quantizers)])

        self.input_proj = None
        self.output_proj = None
        if config.vector_quantization_hidden_dimension != config.hidden_size:
            self.input_proj = torch.nn.Conv1d(
                config.hidden_size, config.vector_quantization_hidden_dimension, 1, bias=False
            )
            self.output_proj = torch.nn.Conv1d(
                config.vector_quantization_hidden_dimension, config.hidden_size, 1, bias=False
            )

    def encode(self, embeddings: torch.Tensor, num_quantizers: int | None = None) -> torch.Tensor:
        """
        Encode a given input tensor with the specified frame rate at the given number of quantizers / codebooks. The RVQ encode method sets
        the appropriate number of quantizers to use and returns indices for each quantizer.
        """
        if self.input_proj is not None:
            embeddings = self.input_proj(embeddings)

        num_quantizers = num_quantizers if num_quantizers is not None else self.num_quantizers

        residual = embeddings
        all_indices = []
        for layer in self.layers[:num_quantizers]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes of shape [B, K, T] to the quantized representation."""
        quantized_out = torch.tensor(0.0, device=codes.device)
        codes = codes.transpose(0, 1)
        for i, indices in enumerate(codes):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized

        if self.output_proj is not None:
            quantized_out = self.output_proj(quantized_out)
        return quantized_out

class MimiSplitResidualVectorQuantizer(nn.Module):
    """Split Residual Vector Quantizer."""

    def __init__(self, config: MimiConfig):
        super().__init__()
        self.codebook_size = config.codebook_size
        self.frame_rate = config.frame_rate
        self.max_num_quantizers = config.num_quantizers

        self.num_semantic_quantizers = config.num_semantic_quantizers
        self.num_acoustic_quantizers = config.num_quantizers - config.num_semantic_quantizers

        self.semantic_residual_vector_quantizer = MimiResidualVectorQuantizer(config, self.num_semantic_quantizers)
        self.acoustic_residual_vector_quantizer = MimiResidualVectorQuantizer(config, self.num_acoustic_quantizers)

    def encode(self, embeddings: torch.Tensor, num_quantizers: float | None = None) -> torch.Tensor:
        """
        Encode a given input tensor with the specified frame rate at the given number of quantizers / codebooks. The RVQ encode method sets
        the appropriate number of quantizers to use and returns indices for each quantizer.
        """

        num_quantizers = self.max_num_quantizers if num_quantizers is None else num_quantizers

        if num_quantizers > self.max_num_quantizers:
            raise ValueError(
                f"The number of quantizers (i.e codebooks) asked should be lower than the total number of quantizers {self.max_num_quantizers}, but is currently {num_quantizers}."
            )

        if num_quantizers < self.num_semantic_quantizers:
            raise ValueError(
                f"The number of quantizers (i.e codebooks) asked should be higher than the number of semantic quantizers {self.num_semantic_quantizers}, but is currently {num_quantizers}."
            )

        # codes is [K, B, T], with T frames, K nb of codebooks.
        codes = self.semantic_residual_vector_quantizer.encode(embeddings)

        if num_quantizers > self.num_semantic_quantizers:
            acoustic_codes = self.acoustic_residual_vector_quantizer.encode(
                embeddings, num_quantizers=num_quantizers - self.num_semantic_quantizers
            )
            codes = torch.cat([codes, acoustic_codes], dim=0)

        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""

        # The first num_semantic_quantizers codebooks are decoded using the semantic RVQ
        quantized_out = self.semantic_residual_vector_quantizer.decode(codes[:, : self.num_semantic_quantizers])

        # The rest of the codebooks are decoded using the acoustic RVQ
        if codes.shape[1] > self.num_semantic_quantizers:
            quantized_out += self.acoustic_residual_vector_quantizer.decode(codes[:, self.num_semantic_quantizers :])
        return quantized_out
