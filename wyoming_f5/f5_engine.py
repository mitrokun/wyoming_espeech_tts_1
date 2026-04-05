import os
import contextlib
import sys
import torch
import torchaudio
import numpy as np
import re
from huggingface_hub import hf_hub_download
from silero_stress import load_accentor

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
    def __init__(self, voice_configs: dict, nfe_steps: int, speed: float, debug: bool = False):
        print("Инициализация движка F5-TTS...")
        self.debug = debug
        self.speed = speed

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA не найдена.")
        self.device = torch.device("cuda")
        
        self.model = self._load_tts_model()
        self.vocoder = load_vocoder()
        self.accentizer = self._load_accentizer()
        
        self.model.to(self.device)
        self.vocoder.to(self.device)

        self.voice_references = {}
        for name, config in voice_configs.items():
            ref_audio_path = config["ref_audio"]
            ref_text = config["ref_text"]
            
            processed_ref_text = self._apply_hybrid_accentuation(ref_text)
            ref_audio_proc, processed_ref_text_final = preprocess_ref_audio_text(
                ref_audio_path,
                processed_ref_text
            )
            self.voice_references[name] = (ref_audio_proc, processed_ref_text_final)
        
        self.nfe_steps = nfe_steps
        print(f"Движок готов. Устройство: {self.device}.")

    def _load_tts_model(self) -> DiT:
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        vocab_path = hf_hub_download(repo_id=MODEL_REPO, filename=VOCAB_FILE)
        return load_model(DiT, MODEL_CFG, model_path, vocab_file=vocab_path)

    def _load_accentizer(self):
        try:
            accentizer = load_accentor()
            accentizer.to('cpu')
            return accentizer
        except:
            return None

    def _apply_hybrid_accentuation(self, text: str) -> str:
        if not self.accentizer or '+' not in text:
            return self.accentizer(text) if self.accentizer else text
        tokens = re.findall(r'([а-яА-ЯёЁ+]+|[^а-яА-ЯёЁ+]+)', text)
        words_to_accent = [token for token in tokens if '+' not in token and re.search(r'[а-яА-ЯёЁ]', token)]
        if not words_to_accent: return text
        text_to_accent = ' '.join(words_to_accent)
        accented_text = self.accentizer(text_to_accent)
        accented_words = iter(accented_text.split())
        result_tokens = []
        for token in tokens:
            if '+' not in token and re.search(r'[а-яА-ЯёЁ]', token):
                try: result_tokens.append(next(accented_words))
                except StopIteration: result_tokens.append(token)
            else: result_tokens.append(token)
        return "".join(result_tokens)

    def _chunk_text(self, text: str, max_chars: int = 150) -> list:
        """
        Разбиваем текст на очень мелкие чанки (до 150 символов).
        Это гарантирует, что модель не выйдет за пределы тензора.
        """
        chunks = []
        # Сначала делим по знакам препинания
        sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence: continue
            
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += (sentence + " ")
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Если само предложение длиннее max_chars, бьем его по словам
                if len(sentence) > max_chars:
                    words = sentence.split()
                    temp = ""
                    for w in words:
                        if len(temp) + len(w) <= max_chars:
                            temp += (w + " ")
                        else:
                            chunks.append(temp.strip())
                            temp = w + " "
                    current_chunk = temp
                else:
                    current_chunk = sentence + " "
                    
        if current_chunk:
            chunks.append(current_chunk.strip())
        return [c for c in chunks if c]

    def synthesize(self, text: str, voice_name: str) -> tuple[np.ndarray, int]:
        if voice_name not in self.voice_references:
            voice_name = next(iter(self.voice_references))
            
        ref_audio_proc, processed_ref_text_final = self.voice_references[voice_name]

        # 1. Ударения
        accented_text = self._apply_hybrid_accentuation(text)

        # 2. Агрессивная нарезка на куски
        text_chunks = self._chunk_text(accented_text, max_chars=85)
        
        all_waves = []
        final_sample_rate = 24000

        # 3. Синтезируем каждый чанк отдельно и собираем в список
        for i, chunk in enumerate(text_chunks):
            if self.debug:
                print(f"Синтез чанка {i+1}/{len(text_chunks)}: {chunk}")
            
            infer_kwargs = dict(
                ref_audio=ref_audio_proc,
                ref_text=processed_ref_text_final,
                gen_text=chunk,
                model_obj=self.model,
                vocoder=self.vocoder,
                nfe_step=self.nfe_steps,
                speed=self.speed
            )

            try:
                if self.debug:
                    wave, sr, _ = infer_process(**infer_kwargs)
                else:
                    with open(os.devnull, 'w') as f, \
                         contextlib.redirect_stdout(f), \
                         contextlib.redirect_stderr(f):
                        wave, sr, _ = infer_process(**infer_kwargs)
                
                if wave is not None:
                    all_waves.append(wave)
                    final_sample_rate = sr
            except Exception as e:
                print(f"Ошибка на чанке '{chunk}': {e}")
                continue

        # 4. Склеиваем аудио
        if not all_waves:
            return np.array([], dtype=np.float32), final_sample_rate

        final_wave = np.concatenate(all_waves)
        
        return final_wave, final_sample_rate