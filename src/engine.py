import cv2
import numpy as np
import pandas as pd
from model import main as pegasus
import pytesseract
import PIL.Image
from PIL import ImageEnhance
from PIL import ImageOps
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
    # Process image using OpenCV
    # 1. Загружаем изображение с помощью OpenCV, сразу в оттенках серого
    img_cv = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    if img_cv is None:
        print(f"ERROR: Не удалось загрузить изображение с помощью OpenCV: {fname}")
        return "Ошибка: не удалось загрузить изображение."

    # 2. Улучшение качества и цветокоррекция с использованием OpenCV
    # *** АДАПТИВНАЯ БИНАРИЗАЦИЯ OpenCV ***
    # Применяем адаптивную бинаризацию для лучшей обработки неравномерного освещения
    # Экспериментируйте с block_size и C_value для вашего типа изображений!
    # block_size: Размер окрестности, по которой вычисляется порог (нечетное число > 1)
    # C_value: Константа, вычитаемая из среднего или взвешенного среднего
    block_size = 25 # Начнем с 25. Если символы "слипаются" или тонкие линии исчезают, уменьшите.
    C_value = 10  # Начнем с 10. Увеличение делает порог ниже (больше белого), уменьшение - выше (больше черного).

    try:
        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C обычно дает лучшие результаты, чем MEAN_C
        # cv2.THRESH_BINARY преобразует пиксели ярче порога в 255 (белый), темнее - в 0 (черный)
        binary_img_cv = cv2.adaptiveThreshold(
            img_cv, # Входное изображение (оттенки серого)
            255, # Значение, присваиваемое пикселям выше порога
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Метод адаптивного порога
            cv2.THRESH_BINARY, # Тип порога
            block_size,
            C_value
        )
        print(f"DEBUG: Применена адаптивная бинаризация OpenCV (block_size={block_size}, C={C_value}).")

        # --- Сохранение обработанного изображения для отладки (используем OpenCV) ---
        try:
            base_fname = os.path.basename(fname)
            name, ext = os.path.splitext(base_fname)
            # Сохраняем бинаризованное изображение
            debug_fname = f"{name}_processed_opencv_binarized.png"
            debug_path = os.path.join(DEBUG_IMAGE_DIR, debug_fname)
            cv2.imwrite(debug_path, binary_img_cv) # Сохраняем массив numpy напрямую
            print(f"DEBUG: Обработанное (OpenCV бинаризованное) изображение сохранено: {debug_path}")
        except Exception as e:
            print(f"DEBUG ERROR: Не удалось сохранить обработанное изображение для отладки: {e}")
        # --- Конец секции отладочного сохранения ---

    except Exception as e:
        print(f"DEBUG ERROR: Не удалось применить адаптивную бинаризацию OpenCV: {e}")
        # Если адаптивная бинаризация не удалась, попробуем Tesseract на исходном сером изображении
        binary_img_cv = img_cv # Fallback на серое изображение
        print("DEBUG: Fallback на серое изображение для OCR.")

    tessdata_dir_config = r'--tessdata-dir "/usr/share/tesseract-ocr/5/tessdata" --psm 1 --oem 3'
    raw_text_from_ocr = ""
    try:
        raw_text_from_ocr = pytesseract.image_to_string(
            binary_img_cv,
            config=tessdata_dir_config,
            lang='rus'

        )
        # *** Отладочный вывод raw-текста (можно оставить или убрать после отладки) ***
        print(f"DEBUG: Сырой текст из Tesseract (repr):\n{repr(raw_text_from_ocr[:1000])}")
    except pytesseract.TesseractError as e:
        print(f"ERROR: Ошибка PyTesseract при распознавании: {e}")
        # Если OCR не сработал, возвращаем ошибку вместо попытки суммирования
        return f"Ошибка распознавания текста: {e}"
    except Exception as e:
        print(f"ERROR: Неизвестная ошибка при OCR: {e}")
        return f"Неизвестная ошибка при распознавании текста: {e}"

    cleaned_text = raw_text_from_ocr
    chars_to_remove = ['|', '—', ')', '=', '©']
    for char in chars_to_remove:
        cleaned_text = cleaned_text.replace(char, '')
    cleaned_text = re.sub(r'-\s+', '', cleaned_text)
    cleaned_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\n+', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()

    print(f"DEBUG: Сырой текст из Tesseract (repr):\n{repr(cleaned_text[:1000])}")

    summary = pegasus(cleaned_text)
    return summary