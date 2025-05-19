import cv2
import numpy as np
import pandas as pd
from model import main as summarizer_main
import pytesseract
import os
import re

# Ensure TESSDATA_PREFIX is set if tessdata is not in a standard location
# For example, if you have a local TESSDATA_PATH for development:
# os.environ["TESSDATA_PREFIX"] = os.path.dirname(TESSDATA_PATH) if TESSDATA_PATH.endswith('/') else os.path.dirname(os.path.dirname(TESSDATA_PATH))

# It's often better to pass tessdata-dir directly in the config string if TESSDATA_PREFIX doesn't work reliably across systems.
TESSDATA_PATH = '/usr/share/tesseract-ocr/5/tessdata/'  # User's path
# Check if the path exists and is a directory. If not, Tesseract might fail silently or with errors.
if not os.path.isdir(TESSDATA_PATH):
    print(
        f"WARNING: TESSDATA_PATH '{TESSDATA_PATH}' does not exist or is not a directory. Pytesseract might not find language data.")
    # Fallback or alternative path could be set here if needed, or rely on system default.

DEBUG_IMAGE_DIR = "debug_processed_images"

if not os.path.exists(DEBUG_IMAGE_DIR):
    try:
        os.makedirs(DEBUG_IMAGE_DIR)
        print(f"Создана директория для отладочных изображений: {DEBUG_IMAGE_DIR}")
    except OSError as e:
        print(f"Ошибка при создании директории {DEBUG_IMAGE_DIR}: {e}")


def processor(fname):
    img_cv_original = cv2.imread(fname)  # Load in color first for layout analysis if we were to use it
    if img_cv_original is None:
        print(f"ERROR: Не удалось загрузить изображение с помощью OpenCV: {fname}")
        return "Ошибка: не удалось загрузить изображение."

    img_cv_gray = cv2.cvtColor(img_cv_original, cv2.COLOR_BGR2GRAY)
    # Using grayscale for current OCR pipeline

    # Adaptive thresholding (current approach)
    # Parameters might need tuning depending on image characteristics
    block_size = 31  # Should be odd
    C_value = 15
    binary_img_cv = cv2.adaptiveThreshold(
        img_cv_gray,
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
        debug_fname_binarized = f"{name}_processed_opencv_binarized.png"
        debug_path_binarized = os.path.join(DEBUG_IMAGE_DIR, debug_fname_binarized)
        cv2.imwrite(debug_path_binarized, binary_img_cv)
        print(f"DEBUG: Обработанное (OpenCV бинаризованное) изображение сохранено: {debug_path_binarized}")
    except Exception as e:
        print(f"DEBUG ERROR: Не удалось сохранить обработанное бинаризованное изображение для отладки: {e}")

    raw_text_from_ocr = ""
    try:
        # Using pytesseract.image_to_data to get detailed OCR output including bounding boxes and structure
        # PSM 3: Auto page segmentation with OSD.
        # PSM 6: Assume a single uniform block of text. Might be better if the image IS a single table.
        # PSM 11: Sparse text.
        # Let's stick with PSM 3 for general documents.
        # The --oem 3 uses the LSTM OCR engine.

        rus_traineddata_path = os.path.join(TESSDATA_PATH, 'rus.traineddata')
        if not os.path.exists(rus_traineddata_path):
            print(f"WARNING: rus.traineddata not found at {rus_traineddata_path}. OCR will likely fail for Russian.")

        ocr_config = f'--tessdata-dir "{TESSDATA_PATH}" --psm 3 --oem 3'

        print(f"DEBUG: Pytesseract config: {ocr_config}")

        ocr_data_df = pytesseract.image_to_data(
            binary_img_cv,  # Using the binarized image
            config=ocr_config,
            lang='rus',
            output_type=pytesseract.Output.DATAFRAME
        )
        print(f"DEBUG: OCR data obtained using image_to_data (PSM=3). DataFrame shape: {ocr_data_df.shape}")

        if not ocr_data_df.empty:
            # Filter out low confidence words and empty text entries
            ocr_data_df = ocr_data_df[ocr_data_df.conf > 30]  # Confidence threshold (0-100)
            ocr_data_df.dropna(subset=['text'], inplace=True)
            if 'text' in ocr_data_df.columns:  # Ensure 'text' column exists after dropna
                ocr_data_df['text'] = ocr_data_df['text'].astype(str).str.strip()
                ocr_data_df = ocr_data_df[ocr_data_df['text'] != '']
            else:
                ocr_data_df = pd.DataFrame()  # Empty dataframe if 'text' column was removed

            if not ocr_data_df.empty:
                print(f"DEBUG: Filtered OCR DataFrame shape: {ocr_data_df.shape}")
                # Reconstruct text respecting Tesseract's layout structure (block, paragraph, line)
                paragraphs_content = []
                # Group by block_num, then par_num to reconstruct paragraphs
                # sort=False to keep Tesseract's original order as much as possible
                for (block_num, par_num), par_group_df in ocr_data_df.groupby(['block_num', 'par_num'], sort=False):
                    lines_in_par = []
                    # Within each paragraph, group by line_num
                    for line_num, line_group_df in par_group_df.groupby('line_num', sort=False):
                        # Sort words by word_num within the line for correct order
                        line_group_sorted_df = line_group_df.sort_values('word_num')
                        lines_in_par.append(' '.join(line_group_sorted_df['text']))
                    # Join lines within a paragraph with a space (forming continuous text for the paragraph)
                    paragraphs_content.append(' '.join(lines_in_par))
                # Join paragraphs with double newlines
                raw_text_from_ocr = '\n\n'.join(paragraphs_content)
                print(f"DEBUG: Reconstructed text from OCR data. Length: {len(raw_text_from_ocr)}")
            else:
                print("DEBUG: OCR DataFrame is empty after filtering.")
        else:
            print("DEBUG: OCR DataFrame was initially empty.")

    except pytesseract.TesseractError as e:
        print(f"ERROR: PyTesseract error during image_to_data: {e}")
        try:
            print("DEBUG: Attempting fallback to pytesseract.image_to_string due to image_to_data error.")
            fallback_config = '--psm 3 --oem 3'
            if os.path.isdir(TESSDATA_PATH) and os.path.exists(os.path.join(TESSDATA_PATH, 'rus.traineddata')):
                fallback_config = f'--tessdata-dir "{TESSDATA_PATH}" --psm 3 --oem 3'

            raw_text_from_ocr = pytesseract.image_to_string(
                binary_img_cv,
                config=fallback_config,
                lang='rus',
            )
            print("DEBUG: Fallback to image_to_string succeeded.")
        except Exception as fallback_e:
            print(f"ERROR: PyTesseract fallback to image_to_string also failed: {fallback_e}")
            raw_text_from_ocr = ""
    except Exception as e:
        print(f"ERROR: Unknown error during OCR processing: {e}")
        raw_text_from_ocr = ""

    if raw_text_from_ocr:
        print(f"DEBUG: Сырой текст из OCR (первые 500 символов):\n{raw_text_from_ocr[:500]}")
    else:
        print("DEBUG: Сырой текст из OCR пустой или произошла ошибка.")
        return "Не удалось распознать текст на изображении для суммирования."

    # Text Cleaning
    cleaned_text = raw_text_from_ocr

    # 1. Remove specific unwanted characters
    chars_to_remove = ['|', ')', '=', '©', '®', '°', '«', '*', ';', '`', '“', '”', '„', '»', '«', ':', '_']
    for char in chars_to_remove:
        cleaned_text = cleaned_text.replace(char, '')

    # 2. De-hyphenation: handles words broken across lines (e.g., "сло-\n\nво" -> "слово")
    cleaned_text = re.sub(r'-\s*(\n\s*){1,2}\s*', '', cleaned_text)

    # 3. Normalize multiple spaces/tabs to a single space
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)

    # 4. Normalize multiple newlines to a maximum of two (paragraph separator)
    # First, ensure \n\n style, then reduce excessive newlines.
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

    # 5. Strip leading/trailing whitespace from the whole text
    cleaned_text = cleaned_text.strip()

    # 6. Paragraph-wise cleaning: filter out paragraphs that are empty or contain no significant content.
    if cleaned_text:
        paragraphs = cleaned_text.split('\n\n')
        final_paragraphs = []
        for p_text in paragraphs:
            # For each paragraph, further clean and check if it's substantial
            # Remove internal excessive spacing within a paragraph that might have been missed
            p_cleaned = re.sub(r'\s+', ' ', p_text).strip()
            if p_cleaned and re.search(r'[а-яА-ЯёЁa-zA-Z0-9]', p_cleaned):  # Check for alphanumeric
                final_paragraphs.append(p_cleaned)
        cleaned_text = "\n\n".join(final_paragraphs)

    # 7. Final trim and space normalization (again, just in case)
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text).strip()

    if cleaned_text:
        print(f"DEBUG: Очищенный текст (repr, первые 500 символов):\n{repr(cleaned_text[:500])}")
    else:
        print("DEBUG: Очищенный текст пустой после очистки.")
        return "Не удалось извлечь содержательный текст из изображения после очистки."

    word_count = len(cleaned_text.split())
    sentence_count = len(re.split(r'[.!?…]+', cleaned_text))  # Simple regex sentence count
    print(f"DEBUG: Очищенный текст - Слов: {word_count}, Предложений (оценка regex): {sentence_count}")

    MIN_WORDS_FOR_SUMMARY = 20
    MIN_SENTENCES_FOR_SUMMARY = 2

    if word_count < MIN_WORDS_FOR_SUMMARY or sentence_count < MIN_SENTENCES_FOR_SUMMARY:
        print(f"DEBUG: Текст слишком короткий для суммирования (слов: {word_count}, предложений: {sentence_count}).")
        if cleaned_text:  # Return the short text if it's not empty
            return f"Распознанный текст слишком короткий для полноценного суммирования:\n\n{cleaned_text}"
        else:  # This case should be covered by earlier "empty after cleaning"
            return "Распознанный текст слишком короткий или содержит недостаточно предложений для осмысленного суммирования."

    print("DEBUG: Передача очищенного текста в модуль суммирования.")
    summary = summarizer_main(cleaned_text)
    return summary

# --- Future Enhancements Section (Conceptual) ---
# To truly "understand" tables and inner images beyond OCRing their text:
#
# 1. Layout Analysis:
#    - Use libraries like `layoutparser` with models (e.g., Detectron2, PaddleOCR based)
#      to identify distinct regions: text blocks, tables, figures/images, titles, lists.
#    - Example:
#      ```python
#      # import layoutparser as lp
#      # # Ensure image is in RGB if model expects it
#      # # image_rgb = cv2.cvtColor(img_cv_original, cv2.COLOR_BGR2RGB)
#      # model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
#      #                                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})
#      # layout_result = model.detect(img_cv_original) # Pass BGR or RGB based on model
#      # for block in layout_result:
#      #     cropped_image = block.crop_image(img_cv_original)
#      #     if block.type == "Table":
#      #         # Process table_image with table-specific OCR or structure recognition
#      #     elif block.type == "Figure":
#      #         # Process figure_image with image captioning or OCR if it contains text
#      #     elif block.type == "Text":
#      #         # OCR text_image (perhaps with PSM_SINGLE_BLOCK)
#      ```
#
# 2. Table Structure Recognition and Textualization:
#    - For regions identified as tables:
#      - Use specialized table OCR tools or models (e.g., Table Transformer, Microsoft TR).
#      - Convert the structured table data into a natural language description.
#
# 3. Inner Image Captioning:
#    - For regions identified as figures/images (that are not primarily text):
#      - Use an image captioning model (e.g., from Hugging Face Transformers: BLIP, GIT).
#      - `from transformers import pipeline`
#      - `captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")`
#      - `caption = captioner(figure_image_pil)[0]['generated_text']` (convert cv2 image to PIL)
#      - Prepend/append this caption to the main text.
#
# 4. Aggregation:
#    - Combine OCRed text from text blocks, textualized table data, and image captions
#      in a logical order (e.g., based on their position on the page).
#    - Feed this richer, structured text into the summarization models.
#
# These enhancements require significant additional libraries (like layoutparser, detectron2/paddleocr, transformers for captioning),
# models, and processing logic, potentially increasing complexity and resource requirements.