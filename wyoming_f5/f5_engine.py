# Файл: f5_engine.py (ПОЛНАЯ ВЕРСИЯ С ГИБРИДНЫМИ УДАРЕНИЯМИ)

import os
import contextlib
import sys
import torch
import numpy as np
import re  # Импортируем модуль для регулярных выражений
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

        self.voice_references = {}
        print("Предобработка референсных аудио...")
        for name, config in voice_configs.items():
            print(f"  - Обработка голоса: {name}")
            ref_audio_path = config["ref_audio"]
            ref_text = config["ref_text"]
            
            # <<< ИЗМЕНЕНИЕ: Используем новый гибридный метод
            processed_ref_text = self._apply_hybrid_accentuation(ref_text)
            
            ref_audio_proc, processed_ref_text_final = preprocess_ref_audio_text(
                ref_audio_path,
                processed_ref_text
            )
            self.voice_references[name] = (ref_audio_proc, processed_ref_text_final)
        
        self.nfe_steps = nfe_steps
        print(f"Движок F5-TTS готов. Устройство: {self.device}. Скорость: {self.speed}. Загружено голосов: {len(self.voice_references)}")

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

    def _load_accentizer(self):
        print("Загрузка модели для расстановки ударений (silero-stress)...")
        try:
            accentizer = load_accentor()
            accentizer.to('cpu')
            print("Модель silero-stress успешно загружена.")
            return accentizer
        except Exception as e:
            print(f"Не удалось загрузить модель silero-stress: {e}")
            return None

    # <<< НОВЫЙ МЕТОД: Логика гибридной расстановки ударений
    def _apply_hybrid_accentuation(self, text: str) -> str:
        """
        Применяет автоматическую расстановку ударений только к тем словам,
        в которых пользователь не указал ударение (+) вручную.
        """
        if not self.accentizer or '+' not in text:
            # Если нет accentizer'а или ручных правок, обрабатываем весь текст целиком
            return self.accentizer(text) if self.accentizer else text

        # Разделяем текст на слова (включая '+') и не-слова (пробелы, пунктуация)
        # Это позволяет сохранить исходную структуру предложения
        tokens = re.findall(r'([а-яА-ЯёЁ+]+|[^а-яА-ЯёЁ+]+)', text)
        
        words_to_accent = [token for token in tokens if '+' not in token and re.search(r'[а-яА-ЯёЁ]', token)]
        
        if not words_to_accent:
            # Если все слова уже имеют ударения, ничего не делаем
            return text

        # Объединяем слова для пакетной обработки, акцентируем и разделяем обратно
        text_to_accent = ' '.join(words_to_accent)
        accented_text = self.accentizer(text_to_accent)
        accented_words = iter(accented_text.split())

        # Собираем итоговый список токенов, заменяя слова без ударения на их акцентированные версии
        result_tokens = []
        for token in tokens:
            if '+' not in token and re.search(r'[а-яА-ЯёЁ]', token):
                try:
                    result_tokens.append(next(accented_words))
                except StopIteration:
                    # На случай, если акцентайзер вернул меньше слов, чем ожидал
                    result_tokens.append(token) 
            else:
                result_tokens.append(token)
        
        return "".join(result_tokens)

    def synthesize(self, text: str, voice_name: str) -> tuple[np.ndarray, int]:
        if voice_name not in self.voice_references:
            fallback_name = next(iter(self.voice_references))
            print(f"Внимание: голос '{voice_name}' не найден. Используется запасной голос: '{fallback_name}'.")
            voice_name = fallback_name
            
        ref_audio_proc, processed_ref_text_final = self.voice_references[voice_name]

        # <<< ИЗМЕНЕНИЕ: Используем новый гибридный метод
        processed_gen_text = self._apply_hybrid_accentuation(text)

        infer_args = (
            ref_audio_proc,
            processed_ref_text_final,
            processed_gen_text,
            self.model,
            self.vocoder,
        )
        infer_kwargs = dict(
            nfe_step=self.nfe_steps,
            speed=self.speed
        )

        if self.debug:
            final_wave, final_sample_rate, _ = infer_process(*infer_args, **infer_kwargs)
        else:
            with open(os.devnull, 'w') as f, \
                 contextlib.redirect_stdout(f), \
                 contextlib.redirect_stderr(f):
                final_wave, final_sample_rate, _ = infer_process(*infer_args, **infer_kwargs)
            
        return final_wave, final_sample_rate