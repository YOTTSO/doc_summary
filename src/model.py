from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk

nltk.download('punkt_tab')

LANGUAGE = "russian"
DEFAULT_TEXTRANK_SENTENCES = 3
ABSTRACTIVE_MODEL_NAME = "IlyaGusev/rut5_base_sum_gazeta"
DEFAULT_ABSTRACTIVE_MAX_LENGTH = 200

try:
    abstractive_tokenizer = AutoTokenizer.from_pretrained(ABSTRACTIVE_MODEL_NAME)
    abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(ABSTRACTIVE_MODEL_NAME)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    abstractive_model.to(DEVICE)
    print(f"Абстрактивная модель {ABSTRACTIVE_MODEL_NAME} загружена на {DEVICE}")
except Exception as e:
    print(f"Ошибка при загрузке абстрактивной модели {ABSTRACTIVE_MODEL_NAME}: {e}")
    abstractive_tokenizer = None
    abstractive_model = None
    DEVICE = "cpu"

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
    if not abstractive_model or not abstractive_tokenizer:
        return "Абстрактивная модель не загружена или произошла ошибка."
    if not text or not text.strip():
        return "Не удалось извлечь текст для абстрактивного суммирования."

    try:
        inputs = abstractive_tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        ).to(DEVICE)

        summary_ids = abstractive_model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=10,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )

        summary = abstractive_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

    except Exception as e:
        print(f"Ошибка при выполнении абстрактивного суммирования: {e}")
        print(f"Текст, вызвавший ошибку абстрактивного суммирования:\n{text[:500]}...")
        print(f"Устройство: {DEVICE}")
        try:
            print(f"Длина входных токенов: {inputs['input_ids'].shape[1]}")
        except:
            pass
        return "Произошла ошибка при абстрактивном суммировании текста."


def main(text):
    if not text or not text.strip():
        return "Не удалось извлечь текст для суммирования."

    extractive_summary = summarize_extractive(text, DEFAULT_TEXTRANK_SENTENCES)

    abstractive_summary = summarize_abstractive(text, DEFAULT_ABSTRACTIVE_MAX_LENGTH)
    combined_summary = (
        f"{abstractive_summary}\n\n"
        f"{extractive_summary}"
    )
    print(f'{combined_summary}')
    return combined_summary