"""Keyless refinement backend: Gemma 3 running locally on the Colab GPU.

google/gemma-3-4b-it quantized to 4-bit (~3 GB VRAM) fits a free Colab T4
comfortably and needs no API key of any kind, so it is the default backend.
The weights download once per runtime from HF hub; the repo is license-gated,
so if your HF account has not accepted the Gemma license yet, put HF_TOKEN=...
in api.txt. Expected throughput on a T4 is roughly 20-35 tok/s, i.e. about
1-2 min per page — fine for an unattended overnight run with resume.
"""
from __future__ import annotations

import logging

log = logging.getLogger("ocr_pipeline.refine.gemma")


class GemmaLocalRefiner:
    name = "gemma_local"

    def __init__(self, model_id: str = "google/gemma-3-4b-it", quantization: str = "4bit",
                 max_new_tokens: int = 2048, temperature: float = 0.1, hf_token: str = ""):
        self.model_id = model_id
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.hf_token = hf_token or ""
        self._model = None
        self._tok = None

    def available(self) -> bool:
        # Always attemptable; load() may still fail at runtime (no GPU, hub down, ...)
        return True

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        kwargs: dict = {"device_map": "auto"}
        if self.quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        token = self.hf_token or None
        log.info("Loading local refiner %s (%s)...", self.model_id, self.quantization)
        self._tok = AutoTokenizer.from_pretrained(self.model_id, token=token)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, token=token, **kwargs)
        self._model.eval()

    def refine(self, prompt: str) -> str:
        self.load()
        import torch

        msgs = [{"role": "user", "content": prompt}]
        inputs = self._tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=0.9,
                eos_token_id=self._tok.eos_token_id,
                pad_token_id=self._tok.pad_token_id,
            )
        gen = out[0][inputs.shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True).strip()
