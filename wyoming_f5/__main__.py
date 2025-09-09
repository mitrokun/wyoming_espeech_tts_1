import argparse
import asyncio
import logging
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
    # --- Аргументы для нашего движка ---
    parser.add_argument("--ref-audio", required=True, help="Путь к референсному WAV файлу")
    parser.add_argument("--ref-text", required=True, help="Текст, произносимый в референсном аудио")
    parser.add_argument("--nfe-step", type=int, default=7, help="Количество шагов NFE [7-48]")
    # --- Общие аргументы сервера ---
    parser.add_argument("--uri", default="tcp://0.0.0.0:10206", help="unix://<path> или tcp://<host>:<port>")
    parser.add_argument("--streaming", action="store_true", help="Включить стриминг на границах предложений")
    parser.add_argument("--auto-punctuation", default=".?!", help="Символы для автоматического добавления в конце фраз")
    parser.add_argument("--samples-per-chunk", type=int, default=1024)
    # --- Служебные аргументы ---
    parser.add_argument("--debug", action="store_true", help="Включить DEBUG логирование")
    parser.add_argument("--log-format", default=logging.BASIC_FORMAT, help="Формат логов")
    parser.add_argument("--version", action="version", version=__version__, help="Показать версию и выйти")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format)
    _LOGGER.debug(args)

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="f5-tts-wyoming",
                description="Wyoming server for ESpeech F5-TTS",
                attribution=Attribution(
                    name="ESpeech", url="https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2"
                ),
                installed=True,
                version=__version__,
                supports_synthesize_streaming=args.streaming,
                voices=[
                    TtsVoice(
                        name="espeech-voice",
                        description="ESpeech",
                        attribution=Attribution(name="", url=""),
                        installed=True,
                        version=__version__,
                        languages=["ru-Ru"], 
                    )
                ],
            )
        ],
    )
    
    _LOGGER.info("Инициализация движка F5-TTS (это может занять некоторое время)...")
    try:
        f5_engine = F5_Engine(
            ref_audio_path=args.ref_audio,
            ref_text=args.ref_text,
            nfe_steps=args.nfe_step,
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
