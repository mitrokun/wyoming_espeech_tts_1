import argparse
import asyncio
import io
import logging
import time
import wave
import soundfile as sf

from sentence_stream import SentenceBoundaryDetector
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .f5_engine import F5_Engine
from .text_normalizer import TextNormalizer

_LOGGER = logging.getLogger(__name__)

class F5EventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        f5_engine: F5_Engine,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.f5_engine = f5_engine
        self.normalizer = TextNormalizer()

        self.sbd: SentenceBoundaryDetector | None = None
        self._synthesize: Synthesize | None = None
        self._is_streaming = False
        self._audio_started = False
        
        self._sentence_buffer: str = ""
        _LOGGER.debug("Text normalizer initialized.")

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        try:
            if self._is_streaming:
                if SynthesizeChunk.is_type(event.type):
                    await self._handle_stream_chunk(SynthesizeChunk.from_event(event))
                elif SynthesizeStop.is_type(event.type):
                    await self._handle_stream_stop()
                return True

            if SynthesizeStart.is_type(event.type) and self.cli_args.streaming:
                await self._handle_stream_start(SynthesizeStart.from_event(event))
            elif Synthesize.is_type(event.type):
                await self._handle_single_synthesize(Synthesize.from_event(event))
            
            return True

        except Exception as err:
            _LOGGER.exception("Error processing event: %s", event)
            # Отправляем событие ошибки в формате Wyoming
            await self.write_event(Error(text=str(err), code=err.__class__.__name__).event())
            self._is_streaming = False
        
        return True

    async def _handle_stream_start(self, stream_start: SynthesizeStart):
        _LOGGER.debug("Text stream started")
        self.sbd = SentenceBoundaryDetector()
        self._synthesize = Synthesize(text="", voice=stream_start.voice)
        self._is_streaming = True
        self._audio_started = False
        self._sentence_buffer = ""

    async def _process_sentence(self, sentence: str):
        sentence = sentence.strip()
        if not sentence:
            return

        if self._sentence_buffer:
            self._sentence_buffer += " " + sentence
        else:
            self._sentence_buffer = sentence

        # 15 символов - хороший порог, чтобы не синтезировать "Да." отдельно
        if len(self._sentence_buffer) >= 15:
            _LOGGER.debug(f"Buffer is long enough ({len(self._sentence_buffer)} chars). Flushing.")
            await self._flush_buffer()

    async def _flush_buffer(self):
        text_to_synthesize = self._sentence_buffer.strip()
        self._sentence_buffer = "" 

        if text_to_synthesize:
            await self._synthesize_and_stream_audio(text_to_synthesize)

    async def _handle_stream_chunk(self, stream_chunk: SynthesizeChunk):
        assert self.sbd is not None
        for sentence in self.sbd.add_chunk(stream_chunk.text):
            await self._process_sentence(sentence)

    async def _handle_stream_stop(self):
        assert self.sbd is not None
        remaining_text = self.sbd.finish()
        if remaining_text:
            await self._process_sentence(remaining_text)

        await self._flush_buffer()

        if self._audio_started:
            await self.write_event(AudioStop().event())
        
        await self.write_event(SynthesizeStopped().event())
        _LOGGER.debug("Text stream stopped")
        self._is_streaming = False

    async def _handle_single_synthesize(self, synthesize: Synthesize):
        """Обрабатывает одиночный запрос."""
        self._audio_started = False
        self._sentence_buffer = ""
        self._synthesize = synthesize
        
        sbd = SentenceBoundaryDetector()
        sentences = list(sbd.add_chunk(synthesize.text))
        final_text = sbd.finish()
        if final_text:
            sentences.append(final_text)

        if not sentences:
            if not self._audio_started:
                 # Если текста нет вообще
                 await self.write_event(SynthesizeStopped().event())
            else:
                 await self.write_event(AudioStop().event())
            return
            
        for sentence in sentences:
            await self._process_sentence(sentence)

        await self._flush_buffer()

        if self._audio_started:
            await self.write_event(AudioStop().event())
        
        await self.write_event(SynthesizeStopped().event())

    async def _synthesize_and_stream_audio(self, text: str):
        # Голос берется из сохраненного объекта self._synthesize
        if self._synthesize and self._synthesize.voice:
             voice_name = self._synthesize.voice.name
        else:
             _LOGGER.warning("No voice selected for synthesis. Using default.")
             voice_name = "default"
        
        normalized_text = self.normalizer.normalize(text)
        if not normalized_text:
            return

        # Авто-пунктуация (добавляем точку в конце, если её нет)
        if self.cli_args.auto_punctuation and normalized_text[-1] not in self.cli_args.auto_punctuation:
            normalized_text += self.cli_args.auto_punctuation[0]

        _LOGGER.debug("Synthesizing: '%s'", normalized_text)

        start_time = time.monotonic()

        loop = asyncio.get_running_loop()
        final_wave, sample_rate = await loop.run_in_executor(
            None, self.f5_engine.synthesize, normalized_text, voice_name
        )

        elapsed_time = time.monotonic() - start_time
        _LOGGER.debug(
            f"Generation finished. Chars: {len(normalized_text)}, Time: {elapsed_time:.4f}s"
        )
        
        # Подготовка аудио к отправке
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, final_wave, sample_rate, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        
        with wave.open(wav_buffer, "rb") as wav_file:
            rate, width, channels = wav_file.getframerate(), wav_file.getsampwidth(), wav_file.getnchannels()
            
            if not self._audio_started:
                await self.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
                self._audio_started = True
            
            audio_bytes = wav_file.readframes(wav_file.getnframes())
            bytes_per_chunk = width * channels * self.cli_args.samples_per_chunk
            
            for i in range(0, len(audio_bytes), bytes_per_chunk):
                chunk = audio_bytes[i : i + bytes_per_chunk]
                await self.write_event(AudioChunk(audio=chunk, rate=rate, width=width, channels=channels).event())