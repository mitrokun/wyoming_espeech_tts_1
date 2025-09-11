import logging
import re
from num2words import num2words
import eng_to_ipa as ipa

log = logging.getLogger(__name__)


class _EnglishToRussianNormalizer:
    """
    Класс, инкапсулирующий всю логику для преобразования
    английских слов в русское фонетическое представление.
    Использует трехуровневый подход: словарь исключений, IPA-транскрипцию и,
    в крайнем случае, простой побуквенный транслит.
    """
    SIMPLE_ENGLISH_TO_RUSSIAN = {
        'a': 'э', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г',
        'h': 'х', 'i': 'и', 'j': 'дж', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
        'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
        'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'и', 'z': 'з'
    }

    ENGLISH_EXCEPTIONS = {
        # Бренды и имена
        "google": "гугл", "apple": "эпл", "microsoft": "микрософт",
        "samsung": "самсунг", "toyota": "тойота", "volkswagen": "фольцваген",
        "coca": "кока", "cola": "кола", "pepsi": "пэпси", "whatsapp": "вотсап",
        "telegram": "телеграм", "youtube": "ютуб", "instagram": "инстаграм",
        "facebook": "фэйсбук", "twitter": "твиттер", "iphone": "айфон",
        "tesla": "тесла", "spacex": "спэйс икс", "amazon": "амазон",
        "python": "пайтон", " AI": "эй ай",

        # Служебные слова
        "a": "э", "the": "зе", "of": "оф", "and": "энд", "for": "фо",
        "to": "ту", "in": "ин", "on": "он", "is": "из",
        # Слова, где IPA-библиотека ошибается
        "nice": "найс", "knowledge": "ноуледж", "new": "нью",
        "video": "видео", "cake": "кейк", "choose": "чуз",
        "hot": "хот",
    }

    IPA_TO_RUSSIAN_MAP = {
        # Слоговые согласные
        "əl": "л",
        # Согласные
        "p": "п", "b": "б", "t": "т", "d": "д", "k": "к", "g": "г",
        "m": "м", "n": "н", "ŋ": "нг", "tʃ": "ч", "ʧ": "ч",
        "dʒ": "дж", "ʤ": "дж", "f": "ф", "v": "в", "θ": "с",
        "ð": "з", "s": "с", "z": "з", "ʃ": "ш", "ʒ": "ж",
        "h": "х", "w": "у", "j": "й", "r": "р", "l": "л",
        # Монофтонги
        "i:": "и", "ɪ": "и", "i": "и", "e": "е", "ɛ": "е",
        "æ": "э", "ʌ": "а", "ə": "а", "a": "а", "ɑ": "а",
        "u:": "у", "ʊ": "у", "u": "у", "ɒ": "о", "ɔ:": "о",
        "ɔ": "о", "o": "о", "ɑ:": "а", "ɜ:": "ё",
        # Дифтонги
        "eɪ": "эй", "aɪ": "ай", "ɔɪ": "ой", "aʊ": "ау",
        "oʊ": "оу", "əʊ": "оу", "ɪə": "иэ", "eə": "еэ", "ʊə": "уэ",
        # R-сочетания
        "ər": "ер", "ɚ": "ер", "ɛr": "эр", "ɪr": "ир", "ɑr": "ар",
        # Другие сочетания
        "ju": "ю", "jʊ": "ю",
        # Служебные символы
        "ˈ": "", "ˌ": "", "*": "", "ː": ""
    }
    
    def __init__(self):
        self._max_ipa_key_len = max(len(key) for key in self.IPA_TO_RUSSIAN_MAP.keys())

    def _convert_ipa_to_russian(self, ipa_text: str) -> str:
        result = ""
        pos = 0
        while pos < len(ipa_text):
            found_match = False
            for length in range(self._max_ipa_key_len, 0, -1):
                chunk = ipa_text[pos:pos + length]
                if chunk in self.IPA_TO_RUSSIAN_MAP:
                    result += self.IPA_TO_RUSSIAN_MAP[chunk]
                    pos += length
                    found_match = True
                    break
            if not found_match:
                pos += 1
        return result

    def _transliterate_word(self, match):
        word = match.group(0).lower()

        # Уровень 1: Проверка словаря исключений
        if word in self.ENGLISH_EXCEPTIONS:
            log.debug(f"Replacing '{word}' from exceptions -> '{self.ENGLISH_EXCEPTIONS[word]}'")
            return self.ENGLISH_EXCEPTIONS[word]

        # Уровень 2: Фонетическая транскрипция через IPA
        try:
            ipa_transcription = re.sub(r'[/]', '', ipa.convert(word)).strip()
            if '*' in ipa_transcription: raise ValueError("IPA conversion failed.")

            russian_phonetics = self._convert_ipa_to_russian(ipa_transcription)
            russian_phonetics = re.sub(r'йй', 'й', russian_phonetics)
            russian_phonetics = re.sub(r'([чшщждж])ь', r'\1', russian_phonetics)

            log.debug(f"Phonetic replacement: '{word}' -> '{ipa_transcription}' -> '{russian_phonetics}'")
            return russian_phonetics
        except Exception:
            # Уровень 3: Простой побуквенный транслит
            log.warning(f"Could not get IPA for '{word}'. Falling back to simple transliteration.")
            return ''.join(self.SIMPLE_ENGLISH_TO_RUSSIAN.get(c, c) for c in word)

    def normalize(self, text: str) -> str:
        return re.sub(r'\b[a-zA-Z]+\b', self._transliterate_word, text)


class TextNormalizer:
    """
    Класс для полной очистки и нормализации русского текста перед TTS.
    """
    _emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF" u"\U00002600-\U000026FF" u"\U00002700-\U000027BF"
        u"\U0001F900-\U0001F9FF" u"\u200D" u"\uFE0F"
        "]+",
        flags=re.UNICODE
    )
    _chars_to_delete = "=#$“”„«»<>*\"‘’‚‹›'/"
    _map_from = "—–−\xa0"
    _map_to = "--- "
    _translation_table = str.maketrans(_map_from, _map_to, _chars_to_delete)
    _FINAL_CLEANUP_PATTERN = re.compile(r'[^а-яА-ЯёЁ.,?! ]')

    def __init__(self):
        """
        Инициализирует TextNormalizer и его компоненты.
        """
        self._eng_normalizer = _EnglishToRussianNormalizer()

    def normalize(self, text: str) -> str:
        """
        Выполняет полный цикл нормализации текста.
        """
        # Этап 1: "Умная" обработка процентов
        normalized_text = self._normalize_percentages(text)
        # Этап 2: Базовая нормализация символов, пробелов, эмодзи
        normalized_text = self._normalize_special_chars(normalized_text)
        # Этап 3: Оставшиеся числа в слова
        normalized_text = self._normalize_numbers(normalized_text)
        # Этап 4: Английские слова в русскую транслитерацию
        normalized_text = self._normalize_english(normalized_text)
        # Этап 5: Финальная очистка от неразрешенных символов
        normalized_text = self._cleanup_final_text(normalized_text)
        # Финальная чистка пробелов
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()

        return normalized_text

    def _choose_percent_form(self, number_str: str) -> str:
        if '.' in number_str or ',' in number_str: return "процента"
        try:
            number = int(number_str)
            if 10 < number % 100 < 20: return "процентов"
            last_digit = number % 10
            if last_digit == 1: return "процент"
            if last_digit in [2, 3, 4]: return "процента"
            return "процентов"
        except (ValueError, OverflowError): return "процентов"

    def _normalize_percentages(self, text: str) -> str:
        def replace_match(match):
            number_str_clean = match.group(1).replace(',', '.')
            percent_word = self._choose_percent_form(number_str_clean)
            return f" {number_str_clean} {percent_word} "
        processed_text = re.sub(r'(\d+([.,]\d+)?)\s*\%', replace_match, text)
        return processed_text.replace('%', ' процентов ')

    def _normalize_special_chars(self, text: str) -> str:
        text = self._emoji_pattern.sub(r'', text)
        text = text.translate(self._translation_table)
        text = text.replace('…', '.')
        text = re.sub(r':(?!\d)', ',', text)
        text = re.sub(r'([a-zA-Zа-яА-ЯёЁ])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Zа-яА-ЯёЁ])', r'\1 \2', text)
        text = text.replace('\n', ' ').replace('\t', ' ')
        return re.sub(r'\s+', ' ', text).strip()

    def _normalize_numbers(self, text: str) -> str:
        def replace_number(match):
            num_str = match.group(0).replace(',', '.')
            try:
                if '.' in num_str:
                    parts = num_str.split('.')
                    integer_part_str, fractional_part_str = parts[0], parts[1]
                    if not integer_part_str or not fractional_part_str:
                        valid_num_str = num_str.replace('.', '')
                        return num2words(int(valid_num_str), lang='ru') if valid_num_str.isdigit() else num_str

                    integer_part_val, fractional_part_val = int(integer_part_str), int(fractional_part_str)
                    fractional_len = len(fractional_part_str)
                    integer_words, fractional_words = num2words(integer_part_val, lang='ru'), num2words(fractional_part_val, lang='ru')
                    
                    if fractional_len == 1: return f"{integer_words} и {fractional_words}"
                    if fractional_part_val % 10 == 1 and fractional_part_val % 100 != 11:
                        if fractional_words.endswith("один"): fractional_words = fractional_words[:-4] + "одна"
                    if fractional_part_val % 10 == 2 and fractional_part_val % 100 != 12:
                        if fractional_words.endswith("два"): fractional_words = fractional_words[:-3] + "две"
                    if fractional_len == 2: return f"{integer_words} и {fractional_words} сотых"
                    if fractional_len == 3: return f"{integer_words} и {fractional_words} тысячных"
                    return f"{integer_words} точка {fractional_words}"
                else: return num2words(int(num_str), lang='ru')
            except (ValueError, OverflowError) as e:
                log.warning(f"Could not normalize number '{num_str}': {e}")
                return num_str
        return re.sub(r'\b\d+([.,]\d+)?\b', replace_number, text)

    def _normalize_english(self, text: str) -> str:
        """
        Нормализует английские слова, используя специализированный класс.
        """
        return self._eng_normalizer.normalize(text)

    def _cleanup_final_text(self, text: str) -> str:
        return self._FINAL_CLEANUP_PATTERN.sub(' ', text)
