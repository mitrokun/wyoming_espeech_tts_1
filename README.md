Установили CUDA Toolkit, необходимые библиотеки `pip install wyoming f5-tts torch torchaudio soundfile numpy huggingface_hub num2words eng_to_ipa silero-stress` 

Подобрали 8-10 секунд референсного голоса в wav `mono 16bit 44100Hz` и его расшифровку

Пример запуска
`python -m wyoming_f5 --ref-audio "D:\sushko\007.wav" --ref-text "Первое предполагает пристальное внимание к жизни, второе – ощущение нормы, более ценное, чем жизненная прагматика" --uri 'tcp://0.0.0.0:10210' --streaming`

Наши львы: [https://huggingface.co/ESpeech](https://huggingface.co/ESpeech)
