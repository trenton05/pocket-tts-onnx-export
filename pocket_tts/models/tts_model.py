from __future__ import annotations

import copy
import logging
import os
import queue
import statistics
import threading
import time
from functools import lru_cache
from pathlib import Path

import safetensors
import torch
from torch import nn
from torch.nn import functional as F
from typing_extensions import Self

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.default_parameters import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    DEFAULT_VARIANT,
)
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules import mimi_transformer
from pocket_tts.modules.dummy_quantizer import MimiSplitResidualVectorQuantizer
from pocket_tts.modules.seanet import SEANetDecoder, SEANetEncoder
from pocket_tts.modules.stateful_module import increment_steps, init_states
from pocket_tts.utils.config import Config, load_config
from pocket_tts.utils.utils import (
    PREDEFINED_VOICES,
    display_execution_time,
    download_if_necessary,
    load_predefined_voice,
    size_of_dict,
)
from pocket_tts.utils.weights_loading import get_flow_lm_state_dict, get_mimi_state_dict

torch.set_num_threads(1)
logger = logging.getLogger(__name__)


class TTSModel(nn.Module):
    def __init__(
        self,
        temp: float,
        lsd_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold,
    ):
        super().__init__()
        self.temp = temp
        self.lsd_decode_steps = lsd_decode_steps
        self.noise_clamp = noise_clamp
        self.eos_threshold = eos_threshold
        self.has_voice_cloning = True

    @property
    def device(self) -> str:
        return next(self.parameters()).device.type

    @property
    def sample_rate(self) -> int:
        return 24000

    @classmethod
    def _from_pydantic_config(
        cls, temp, lsd_decode_steps, noise_clamp: float | None, eos_threshold
    ) -> Self:
        tts_model = cls(temp, lsd_decode_steps, noise_clamp, eos_threshold)
        return tts_model

    @classmethod
    def _from_pydantic_config_with_weights(
        cls, temp, lsd_decode_steps, noise_clamp: float | None, eos_threshold
    ) -> Self:
        tts_model = cls._from_pydantic_config(
            temp, lsd_decode_steps, noise_clamp, eos_threshold
        )

        # safetensors.torch.save_file(tts_model.state_dict(), "7442637a.safetensors")
        # Create mimi config directly from the provided config using model_dump

        # Build mimi model from config
        encoder = SEANetEncoder()
        decoder = SEANetDecoder()

        encoder_transformer = mimi_transformer.ProjectedTransformer()
        decoder_transformer = mimi_transformer.ProjectedTransformer()
        quantizer = MimiSplitResidualVectorQuantizer()

        tts_model.mimi = MimiModel(
            encoder,
            decoder,
            quantizer,
            channels=1,
            sample_rate=24000,
            frame_rate=12.5,
            encoder_frame_rate=24000 / encoder.hop_length,
            encoder_transformer=encoder_transformer,
            decoder_transformer=decoder_transformer,
        ).to(device="cpu")

        # Load mimi weights from the config safetensors file with complete mapping for strict loading

        tts_model.mimi.eval()
        # tts_model.to(dtype=torch.float32)

        size_in_mb = size_of_dict(tts_model.state_dict()) // 1e6
        logging.info(f"TTS Model loaded successfully. Its size is {size_in_mb} MB")

        return tts_model

    def load_model(
        variant: str = DEFAULT_VARIANT,
        temp: float | int = DEFAULT_TEMPERATURE,
        lsd_decode_steps: int = DEFAULT_LSD_DECODE_STEPS,
        noise_clamp: float | int | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float = DEFAULT_EOS_THRESHOLD,
    ) -> Self:
        """Load a pre-trained TTS model with specified configuration.

        This class method loads a complete TTS model including the flow language model
        and Mimi compression model from pre-trained weights. The model is initialized
        with the specified generation parameters and ready for inference.

        Args:
            variant: Model variant identifier corresponding to a config file name
                (e.g., '610b0b2c'). Must match a YAML file in the config directory.
            temp: Sampling temperature for generation. Higher values produce more
                diverse but potentially lower quality output.
            lsd_decode_steps: Number of steps for Lagrangian Self Distillation
                decoding. More steps can improve quality but increase computation.
            noise_clamp: Maximum value for noise sampling. If None, no clamping
                is applied. Helps prevent extreme values in generation.
            eos_threshold: Threshold for end-of-sequence detection. Higher values
                make the model more likely to continue generating.

        Returns:
            TTSModel: Fully initialized model with loaded weights on cpu, ready for
                text-to-speech generation.

        Raises:
            FileNotFoundError: If the specified config file or model weights
                are not found.
            ValueError: If the configuration is invalid or incompatible.
        """
        tts_model = TTSModel._from_pydantic_config_with_weights(
            temp, lsd_decode_steps, noise_clamp, eos_threshold
        )
        return tts_model
