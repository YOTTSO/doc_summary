from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk

nltk.download('punkt_tab')

LANGUAGE = "russian" # Язык для TextRank и токенизатора
DEFAULT_TEXTRANK_SENTENCES = 3 # Количество предложений для экстрактивного реферата (можно настроить)
ABSTRACTIVE_MODEL_NAME = "IlyaGusev/rut5_base_sum_gazeta" # Русская абстрактивная модель для суммаризации
DEFAULT_ABSTRACTIVE_MAX_LENGTH = 150 # Максимальное количество токенов для абстрактивного реферата (можно настроить)

try:
    abstractive_tokenizer = AutoTokenizer.from_pretrained(ABSTRACTIVE_MODEL_NAME)
    abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(ABSTRACTIVE_MODEL_NAME)
    # Определяем устройство (GPU или CPU)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    abstractive_model.to(DEVICE)
    print(f"Абстрактивная модель {ABSTRACTIVE_MODEL_NAME} загружена на {DEVICE}")
except Exception as e:
    print(f"Ошибка при загрузке абстрактивной модели {ABSTRACTIVE_MODEL_NAME}: {e}")
    abstractive_tokenizer = None
    abstractive_model = None
    DEVICE = "cpu" # На всякий случай

def summarize_extractive(text, sentences_count=DEFAULT_TEXTRANK_SENTENCES):

    if not text or not text.strip():
        return "Не удалось извлечь текст для экстрактивного суммирования."

    try:
        parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))

        try:
            stemmer = Stemmer(LANGUAGE)
        except LookupError:
            print(f"Предупреждение (TextRank): Стеммер для языка '{LANGUAGE}' не найден. Продолжаем без стеммера.")
            stemmer = None

        summarizer = TextRankSummarizer(stemmer)

        try:
            summarizer.stop_words = get_stop_words(LANGUAGE)
        except LookupError:
            print(f"Предупреждение (TextRank): Стоп-слова для языка '{LANGUAGE}' не найдены. Продолжаем без стоп-слов.")
            pass

        num_sentences_in_text = len(list(parser.document.sentences))
        print(f"DEBUG (TextRank): Распознано предложений sumy: {num_sentences_in_text}")
        actual_sentences_count = min(sentences_count, num_sentences_in_text)
        if actual_sentences_count == 0 and num_sentences_in_text > 0:
             actual_sentences_count = 1
        elif actual_sentences_count == 0 and num_sentences_in_text == 0:
             return "Не удалось разбить текст на предложения для экстрактивного суммирования."


        summary_sentences = summarizer(parser.document, actual_sentences_count)
        summary = " ".join(str(sentence) for sentence in summary_sentences)

        return summary

    except Exception as e:
        print(f"Ошибка при выполнении TextRank суммирования: {e}")
        print(f"Текст, вызвавший ошибку TextRank:\n{text[:500]}...")
        return "Произошла ошибка при экстрактивном суммировании текста."

def summarize_abstractive(text, max_length=DEFAULT_ABSTRACTIVE_MAX_LENGTH):
    """
    Выполняет абстрактивное суммирование текста с использованием Transformers модели.
    """
    if not abstractive_model or not abstractive_tokenizer:
        return "Абстрактивная модель не загружена или произошла ошибка."
    if not text or not text.strip():
        return "Не удалось извлечь текст для абстрактивного суммирования."

    try:
        # Подготовка текста для модели
        # Модели могут иметь ограничения на входную длину. Truncation=True обрезает текст.
        inputs = abstractive_tokenizer(
            text,
            max_length=512, # Большинство моделей ограничены 512 токенами, но это зависит от модели
            truncation=True,
            padding="longest", # Или "max_length" если хотите фиксированный размер
            return_tensors="pt"
        ).to(DEVICE)

        # Генерация реферата
        # num_beams > 1 часто улучшает качество за счет поиска по нескольким гипотезам
        # early_stopping=True останавливает генерацию, как только все лучи нашли конец предложения
        summary_ids = abstractive_model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=10, # Минимальная длина реферата
            num_beams=4, # Количество лучей для поиска
            early_stopping=True
        )

        # Декодирование сгенерированных токенов обратно в текст
        summary = abstractive_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

    except Exception as e:
        print(f"Ошибка при выполнении абстрактивного суммирования: {e}")
        print(f"Текст, вызвавший ошибку абстрактивного суммирования:\n{text[:500]}...")
        # Также выводим информацию об устройстве, если есть ошибка
        print(f"Устройство: {DEVICE}")
        # Выводим информацию о входных токенах, если есть
        try:
            print(f"Длина входных токенов: {inputs['input_ids'].shape[1]}")
        except:
            pass
        return "Произошла ошибка при абстрактивном суммировании текста."


def main(text):
    """
    Выполняет гибридное суммирование текста (экстрактивное + абстрактивное).
    """
    if not text or not text.strip():
        return "Не удалось извлечь текст для суммирования."

    # Выполняем экстрактивное суммирование
    extractive_summary = summarize_extractive(text, DEFAULT_TEXTRANK_SENTENCES)

    # Выполняем абстрактивное суммирование
    abstractive_summary = summarize_abstractive(text, DEFAULT_ABSTRACTIVE_MAX_LENGTH)

    # Комбинируем результаты
    # Простая комбинация: сначала экстрактивный, потом абстрактивный, с разделителями
    combined_summary = (
        f"--- Экстрактивный реферат (TextRank) ---\n"
        f"{extractive_summary}\n\n"
        f"--- Абстрактивный реферат (RuT5) ---\n"
        f"{abstractive_summary}"
    )
    print(f'{combined_summary}')
    return combined_summary

# Закомментированный код для Pegasus
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# import torch
# def main_pegasus(text):
#     # ... (код Pegasus)
#     pass