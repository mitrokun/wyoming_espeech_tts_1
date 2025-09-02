import argparse
import asyncio
import io  # <-- Импортируем io
import logging
import math
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
        self.sbd = SentenceBoundaryDetector()
        self._synthesize: Synthesize | None = None
        self.normalizer = TextNormalizer()
        _LOGGER.debug("Text normalizer initialized.")

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        try:
            if not self.cli_args.streaming:
                if Synthesize.is_type(event.type):
                    synthesize = Synthesize.from_event(event)
                    self._synthesize = Synthesize(text="", voice=synthesize.voice)
                    self.sbd = SentenceBoundaryDetector()
                    start_sent = False
                    for i, sentence in enumerate(self.sbd.add_chunk(synthesize.text)):
                        self._synthesize.text = sentence
                        await self._handle_synthesize(
                            self._synthesize, send_start=(i == 0), send_stop=False
                        )
                        start_sent = True
                    self._synthesize.text = self.sbd.finish()
                    if self._synthesize.text:
                        await self._handle_synthesize(
                            self._synthesize, send_start=(not start_sent), send_stop=True
                        )
                    else:
                        await self.write_event(AudioStop().event())
                    return True
                return True

            if SynthesizeStart.is_type(event.type):
                stream_start = SynthesizeStart.from_event(event)
                self.sbd = SentenceBoundaryDetector()
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
                _LOGGER.debug("Text stream started")
                return True

            if SynthesizeChunk.is_type(event.type):
                assert self._synthesize is not None
                stream_chunk = SynthesizeChunk.from_event(event)
                for sentence in self.sbd.add_chunk(stream_chunk.text):
                    _LOGGER.debug("Synthesizing stream sentence: %s", sentence)
                    self._synthesize.text = sentence
                    await self._handle_synthesize(self._synthesize)
                return True

            if SynthesizeStop.is_type(event.type):
                assert self._synthesize is not None
                self._synthesize.text = self.sbd.finish()
                if self._synthesize.text:
                    await self._handle_synthesize(self._synthesize)
                await self.write_event(SynthesizeStopped().event())
                _LOGGER.debug("Text stream stopped")
                return True

        except Exception as err:
            _LOGGER.exception("Error processing event: %s", event)
            await self.write_event(Error(text=str(err), code=err.__class__.__name__).event())
        
        return True

    async def _handle_synthesize(
        self, synthesize: Synthesize, send_start: bool = True, send_stop: bool = True
    ) -> bool:
        """Основная функция синтеза с полной нормализацией и работой в памяти."""
        raw_text = synthesize.text
        normalized_text = self.normalizer.normalize(raw_text)

        if not normalized_text:
            _LOGGER.debug("Text became empty after normalization. Skipping synthesis. Original: '%s'", raw_text)
            return True

        if self.cli_args.auto_punctuation and normalized_text and normalized_text[-1] not in self.cli_args.auto_punctuation:
            normalized_text += self.cli_args.auto_punctuation[0]

        _LOGGER.debug("Original text: '%s' | Normalized text: '%s'", raw_text, normalized_text)
        
        loop = asyncio.get_running_loop()
        final_wave, sample_rate = await loop.run_in_executor(
            None, self.f5_engine.synthesize, normalized_text
        )

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, final_wave, sample_rate, format='WAV', subtype='PCM_16')
        
        wav_buffer.seek(0)
        
        with wave.open(wav_buffer, "rb") as wav_file:
            rate, width, channels = wav_file.getframerate(), wav_file.getsampwidth(), wav_file.getnchannels()
            
            if send_start:
                await self.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
            
            audio_bytes = wav_file.readframes(wav_file.getnframes())
            bytes_per_sample = width * channels
            bytes_per_chunk = bytes_per_sample * self.cli_args.samples_per_chunk
            num_chunks = math.ceil(len(audio_bytes) / bytes_per_chunk)
            
            for i in range(num_chunks):
                offset = i * bytes_per_chunk
                chunk = audio_bytes[offset : offset + bytes_per_chunk]
                await self.write_event(AudioChunk(audio=chunk, rate=rate, width=width, channels=channels).event())
        
        if send_stop:
            await self.write_event(AudioStop().event())

        _LOGGER.debug("Finished synthesizing: '%s'", normalized_text)
        return True
