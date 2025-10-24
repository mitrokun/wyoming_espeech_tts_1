import argparse
import asyncio
import logging
import os
from functools import partial

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from . import __version__
from .f5_engine import F5_Engine
from .handler import F5EventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Точка входа."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voice",
        required=True,
        nargs=2,
        action="append",
        metavar=("WAV_PATH", "TEXT"),
        help="Задать голос. Укажите путь к WAV файлу и соответствующий ему текст. "
             "Этот аргумент можно использовать несколько раз для добавления нескольких голосов.",
    )

    parser.add_argument("--uri", default="tcp://0.0.0.0:10206", help="URI сервера")
    parser.add_argument("--streaming", action="store_true", help="Включить стриминг")
    parser.add_argument("--nfe-step", type=int, default=16, help="Количество шагов NFE")
    parser.add_argument("--speed", type=float, default=0.6, help="Скорость синтеза")
    parser.add_argument("--auto-punctuation", default=".?!", help="Авто-пунктуация")
    parser.add_argument("--samples-per-chunk", type=int, default=1024)
    parser.add_argument("--debug", action="store_true", help="Включить DEBUG логирование")
    parser.add_argument("--log-format", default=logging.BASIC_FORMAT, help="Формат логов")
    parser.add_argument("--version", action="version", version=__version__, help="Показать версию")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format)
    _LOGGER.debug(args)

    wyoming_voices = []
    engine_voice_configs = {}
    
    for i, voice_data in enumerate(args.voice):
        audio_path, text = voice_data
        stable_name = f"espeech-voice-{i+1:02d}"
        description = os.path.splitext(os.path.basename(audio_path))[0]
        
        # 1. Голос для Wyoming
        wyoming_voices.append(
            TtsVoice(
                name=stable_name,
                description=description,
                attribution=Attribution(name="", url=""),
                installed=True,
                version=__version__,
                languages=["ru-Ru"],
            )
        )
        
        # 2. Конфигурация для движка
        engine_voice_configs[stable_name] = {
            "ref_audio": audio_path,
            "ref_text": text,
        }

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="ESpeech",
                description="Wyoming server for ESpeech F5-TTS",
                attribution=Attribution(name="ESpeech", url="https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2"),
                installed=True,
                version=__version__,
                supports_synthesize_streaming=args.streaming,
                voices=wyoming_voices,
            )
        ],
    )
    
    _LOGGER.info(f"Найдено и сконфигурировано {len(wyoming_voices)} голосов.")
    _LOGGER.info("Инициализация движка F5-TTS...")
    try:
        f5_engine = F5_Engine(
            voice_configs=engine_voice_configs,
            nfe_steps=args.nfe_step,
            speed=args.speed,
            debug=args.debug
        )
    except RuntimeError as e:
        _LOGGER.fatal(e)
        return

    _LOGGER.info("Движок готов.")

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Сервер готов к работе по URI: %s", args.uri)
    
    await server.run(
        partial(
            F5EventHandler,
            wyoming_info,
            args,
            f5_engine,
        )
    )

def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    run()