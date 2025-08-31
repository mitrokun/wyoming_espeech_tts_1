import os
import contextlib
import sys # <-- Импортируем sys для проверки версии Python
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from ruaccent import RUAccent
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.model import DiT

MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
MODEL_REPO = "ESpeech/ESpeech-TTS-1_RL-V2"
MODEL_FILE = "espeech_tts_rlv2.pt"
VOCAB_FILE = "vocab.txt"

class F5_Engine:
    def __init__(self, ref_audio_path: str, ref_text: str, nfe_steps: int, debug: bool = False):
        print("Инициализация движка F5-TTS...")
        self.debug = debug

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA не найдена. Этот сервер требует наличия GPU и корректно установленного PyTorch с поддержкой CUDA."
            )
        self.device = torch.device("cuda")
        
        self.model = self._load_tts_model()
        self.vocoder = load_vocoder()
        self.accentizer = self._load_accentizer()
        
        print(f"Перемещение моделей на устройство: {self.device}")
        self.model.to(self.device)
        self.vocoder.to(self.device)

        print("Предобработка референсного аудио...")
        processed_ref_text = self.accentizer.process_all(ref_text) if '+' not in ref_text else ref_text
        self.ref_audio_proc, self.processed_ref_text_final = preprocess_ref_audio_text(
            ref_audio_path,
            processed_ref_text
        )
        
        self.nfe_steps = nfe_steps
        print(f"Движок F5-TTS готов. Устройство: {self.device}")

    def _load_tts_model(self) -> DiT:
        print(f"Загрузка модели '{MODEL_FILE}' из репозитория '{MODEL_REPO}'...")
        try:
            model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
            vocab_path = hf_hub_download(repo_id=MODEL_REPO, filename=VOCAB_FILE)
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить файлы модели с Hugging Face Hub: {e}")
        if not os.path.exists(model_path) or not os.path.exists(vocab_path):
            raise FileNotFoundError("Файлы модели или словаря не найдены после попытки загрузки.")
        print("Инициализация TTS модели...")
        return load_model(DiT, MODEL_CFG, model_path, vocab_file=vocab_path)

    def _load_accentizer(self) -> RUAccent:
        print("Загрузка модели для расстановки ударений (RUAccent)...")
        accentizer = RUAccent()
        accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False)
        return accentizer

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        processed_gen_text = self.accentizer.process_all(text) if '+' not in text else text

        infer_args = (
            self.ref_audio_proc,
            self.processed_ref_text_final,
            processed_gen_text,
            self.model,
            self.vocoder,
        )
        infer_kwargs = dict(nfe_step=self.nfe_steps)

        if self.debug:
            # В режиме отладки вызываем как есть, со всем выводом
            final_wave, final_sample_rate, _ = infer_process(*infer_args, **infer_kwargs)
        else:
            # В обычном режиме подавляем ВЕСЬ вывод в консоль
            with open(os.devnull, 'w') as f, \
                 contextlib.redirect_stdout(f), \
                 contextlib.redirect_stderr(f):
                final_wave, final_sample_rate, _ = infer_process(*infer_args, **infer_kwargs)
            
        return final_wave, final_sample_rate