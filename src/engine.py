import cv2
import numpy as np
import pandas as pd
from model import main as summarizer_main
import pytesseract
import os
import re


TESSDATA_PATH = '/usr/share/tesseract-ocr/5/tessdata/'
DEBUG_IMAGE_DIR = "debug_processed_images"

if not os.path.exists(DEBUG_IMAGE_DIR):
    try:
        os.makedirs(DEBUG_IMAGE_DIR)
        print(f"Создана директория для отладочных изображений: {DEBUG_IMAGE_DIR}")
    except OSError as e:
        print(f"Ошибка при создании директории {DEBUG_IMAGE_DIR}: {e}")

def processor(fname):
    img_cv = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    if img_cv is None:
        print(f"ERROR: Не удалось загрузить изображение с помощью OpenCV: {fname}")
        return "Ошибка: не удалось загрузить изображение."

    img_cv_processed = img_cv
    block_size = 31
    C_value = 15
    try:
        binary_img_cv = cv2.adaptiveThreshold(
            img_cv_processed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C_value
        )
        print(f"DEBUG: Применена адаптивная бинаризация OpenCV (block_size={block_size}, C={C_value}).")

        try:
            base_fname = os.path.basename(fname)
            name, ext = os.path.splitext(base_fname)
            debug_fname = f"{name}_processed_opencv_binarized.png"
            debug_path = os.path.join(DEBUG_IMAGE_DIR, debug_fname)
            cv2.imwrite(debug_path, binary_img_cv)
            print(f"DEBUG: Обработанное (OpenCV бинаризованное) изображение сохранено: {debug_path}")
        except Exception as e:
            print(f"DEBUG ERROR: Не удалось сохранить обработанное изображение для отладки: {e}")

    except Exception as e:
        print(f"DEBUG ERROR: Не удалось применить адаптивную бинаризацию OpenCV: {e}")
        binary_img_cv = img_cv_processed
        print("DEBUG: Fallback на предобработанное серое изображение для OCR.")

    raw_text_from_ocr = ""
    try:
        TESSERACT_PSM = 3
        myconfig = f'--tessdata-dir "{TESSDATA_PATH}" --psm {TESSERACT_PSM} --oem 3'
        raw_text_from_ocr = pytesseract.image_to_string(
            binary_img_cv,
            config=myconfig,
            lang='rus',
        )
        print(f"DEBUG: Получен текст с помощью pytesseract.image_to_string (PSM={TESSERACT_PSM}, tessdata_dir в config).")

    except pytesseract.TesseractError as e:
        print(f"ERROR: Ошибка PyTesseract при распознавании текста: {e}")
        raw_text_from_ocr = "" # Текст будет пустым в случае ошибки
    except Exception as e:
        print(f"ERROR: Неизвестная ошибка при распознавании текста: {e}")
        raw_text_from_ocr = "" # Текст будет пустым в случае ошибки

    if raw_text_from_ocr:
        print(f"DEBUG: Сырой текст из Tesseract (image_to_string):\n{repr(raw_text_from_ocr[:2000])}") # Выводим больше текста для лучшей отладки
    else:
        print("DEBUG: Сырой текст из Tesseract пустой или произошла ошибка.")

    cleaned_text = raw_text_from_ocr

    chars_to_remove = ['|', '—', ')', '=', '©', '®', '°', '«', '*', ';', '`', '“', '”', '„', '»', '«', ':', '_'] # Добавьте символы, которые видите в мусоре
    for char in chars_to_remove:
        cleaned_text = cleaned_text.replace(char, '')

    cleaned_text = re.sub(r'-\s*\n\s*', '', cleaned_text)
    cleaned_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\n+', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'\n ', '\n', cleaned_text)
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()

    lines = cleaned_text.split('\n')
    filtered_lines = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if re.search(r'[а-яА-ЯёЁa-zA-Z0-9]', stripped_line):
            filtered_lines.append(line)

    cleaned_text = "\n".join(filtered_lines)
    cleaned_text = re.sub(r'\n\n+', '\n\n', cleaned_text)


    if cleaned_text:
        print(f"DEBUG: Очищенный текст (repr):\n{repr(cleaned_text[:2000])}")
    else:
         print("DEBUG: Очищенный текст пустой после очистки.")
    if not cleaned_text:
        return "Не удалось распознать или обработать текст для суммирования."

    word_count = len(cleaned_text.split())
    sentence_count = len(re.split(r'[.!?…]+', cleaned_text))
    print(f"DEBUG: Очищенный текст - Слов: {word_count}, Предложений (оценка): {sentence_count}")

    if word_count < 20 or sentence_count < 3:
         return "Распознанный текст слишком короткий или содержит недостаточно предложений для осмысленного суммирования."
    summary = summarizer_main(cleaned_text)
    return summary