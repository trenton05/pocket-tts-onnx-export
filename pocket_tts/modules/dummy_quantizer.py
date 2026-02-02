import torch
from torch import nn


class MimiEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""

    def __init__(self, epsilon: float = 1e-5):
        super().__init__()
        embed = torch.zeros(2048, 256)

        self.codebook_size = 2048

        self.register_buffer("initialized", torch.tensor([True], dtype=torch.float32))
        self.register_buffer("cluster_usage", torch.ones(2048))
        self.register_buffer("embed_sum", embed)
        self._embed = None
        self.epsilon = epsilon

    @property
    def embed(self) -> torch.Tensor:
        # if self._embed is None:
        #     self._embed = self.embed_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        # return self._embed
        return self.embed_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]

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

# class MimiEuclideanCodebook(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self._epsilon = torch.tensor([1e-5], dtype=torch.float32)
#         self.register_buffer("initialized", torch.tensor([True], dtype=torch.float32))
#         self.register_buffer("cluster_usage", torch.ones(2048))
#         self.register_buffer("embed_sum", torch.zeros(2048, 256))

#     def update(self, parameters: dict) -> nn.Module:
#         super().update(parameters)
#         cluster_usage = torch.maximum(self.cluster_usage, self._epsilon)[:, None]
#         embedding = self.embed_sum / cluster_usage
#         c2 = embedding.square().sum(axis=-1) / 2
#         return self

#     def encode(self, xs: torch.Tensor) -> torch.Tensor:
#         cluster_usage = torch.maximum(self.cluster_usage, self._epsilon)[:, None]
#         embedding = self.embed_sum / cluster_usage
#         c2 = embedding.square().sum(axis=-1) / 2

#         target_shape = xs.shape[:-1]
#         xs = xs.flatten(end_dim=-2)
#         dot_prod = xs @ embedding.swapaxes(-1, -2)
#         return (c2 - dot_prod).argmin(axis=-1).reshape(target_shape)

#     def decode(self, xs: torch.Tensor) -> torch.Tensor:
#         target_shape = list(xs.shape) + [256]

#         cluster_usage = torch.maximum(self.cluster_usage, self._epsilon)[:, None]
#         embedding = self.embed_sum / cluster_usage
#         return nn.functional.embedding(xs, embedding)
#         # return torch.take(embedding, xs.flatten()).reshape(target_shape)


# Copied from transformers.models.encodec.modeling_encodec.EncodecVectorQuantization with Encodec->Mimi
class MimiVectorQuantization(nn.Module):
    """
    Vector quantization implementation. Currently supports only euclidean distance.
    """

    def __init__(self):
        super().__init__()
        self.codebook = MimiEuclideanCodebook()

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

    def __init__(self, num_quantizers: int | None = None):
        super().__init__()
        self.codebook_size = 2048
        self.frame_rate = 12.5
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList([MimiVectorQuantization() for _ in range(self.num_quantizers)])

        self.input_proj = None
        self.output_proj = None
        self.input_proj = torch.nn.Conv1d(
            512, 256, 1, bias=False
        )
        self.output_proj = torch.nn.Conv1d(
            256, 512, 1, bias=False
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
            if indices >= 0:
                quantized = layer.decode(indices)
                quantized_out = quantized_out + quantized

        if self.output_proj is not None:
            quantized_out = self.output_proj(quantized_out)
        return quantized_out

class MimiSplitResidualVectorQuantizer(nn.Module):
    """Split Residual Vector Quantizer."""

    def __init__(self):
        super().__init__()
        self.codebook_size = 2048
        self.frame_rate = 12.5
        self.max_num_quantizers = 32

        self.num_semantic_quantizers = 1
        self.num_acoustic_quantizers = 31

        self.semantic_residual_vector_quantizer = MimiResidualVectorQuantizer(self.num_semantic_quantizers)
        self.acoustic_residual_vector_quantizer = MimiResidualVectorQuantizer(self.num_acoustic_quantizers)

    def encode(self, embeddings: torch.Tensor, num_quantizers: int | float | None = None) -> torch.Tensor:
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
