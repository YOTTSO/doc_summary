from flask import Flask, render_template, send_from_directory, url_for, redirect, request # Добавлены redirect и request
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField

from engine import processor
import os

app = Flask(__name__)
app.config["SECRET_KEY"] = "askjdhfjskd"
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"
app.config["RESULTS_DEST"] = "results"

if not os.path.exists(app.config["UPLOADED_PHOTOS_DEST"]):
    os.makedirs(app.config["UPLOADED_PHOTOS_DEST"])
if not os.path.exists(app.config["RESULTS_DEST"]):
    os.makedirs(app.config["RESULTS_DEST"])


photos = UploadSet("photos", IMAGES)
configure_uploads(app, photos)

class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, "Only images are allowed"), FileRequired("File field should not be empty")])
    submit = SubmitField("Upload")

@app.route('/uploads/<filename>')
def get_file(filename):
    # send_from_directory безопасен, но убедитесь, что filename не содержит символов для обхода пути
    return send_from_directory(app.config["UPLOADED_PHOTOS_DEST"], filename)

@app.route('/results/<filename>')
def get_result_file(filename):
    return send_from_directory(app.config["RESULTS_DEST"], filename)


# Основной маршрут для загрузки
@app.route("/", methods=["GET", "POST"])
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        try:
            filename = photos.save(form.photo.data)
            absolute_path_image = os.path.join(app.config["UPLOADED_PHOTOS_DEST"], filename)

            # Обработка изображения и получение результата
            result_text = processor(absolute_path_image)

            # Сохранение результата в файл
            # Используем базовое имя файла изображения с расширением .txt
            result_filename = f"{os.path.splitext(filename)[0]}.txt"
            absolute_path_result = os.path.join(app.config["RESULTS_DEST"], result_filename)

            try:
                with open(absolute_path_result, "w", encoding="utf-8") as f:
                    f.write(result_text)
                print(f"DEBUG: Result saved to {absolute_path_result}")
            except Exception as e:
                print(f"ERROR: Could not save result file {absolute_path_result}: {e}")
                pass

            # Перенаправление на новую страницу с результатом
            # Передаем имя файла изображения, чтобы на новой странице знать, какой результат и изображение показать
            return redirect(url_for("show_summary", filename=filename))

        except Exception as e:
            # Обработка ошибок при загрузке или обработке
            print(f"ERROR: An error occurred during upload or processing: {e}")
            # Можно добавить flash message или передать ошибку в шаблон
            # Для простоты пока просто рендерим форму с сообщением об ошибке
            form.photo.errors.append(f"Произошла ошибка при обработке изображения: {e}")


    # Для GET запросов или при ошибках валидации просто отображаем форму
    return render_template("index.html", form=form)


# Новый маршрут для отображения результата
@app.route("/summary/<filename>")
def show_summary(filename):
    # Генерируем URL для отображения загруженного изображения
    image_url = url_for("get_file", filename=filename)

    # Формируем путь к файлу с результатом
    result_filename = f"{os.path.splitext(filename)[0]}.txt"
    absolute_path_result = os.path.join(app.config["RESULTS_DEST"], result_filename)

    # Читаем результат из файла
    result_text = "Ошибка: Не удалось загрузить результат." # Дефолтное сообщение об ошибке
    try:
        with open(absolute_path_result, "r", encoding="utf-8") as f:
            result_text = f.read()
        print(f"DEBUG: Result loaded from {absolute_path_result}")
    except FileNotFoundError:
        print(f"ERROR: Result file not found for {filename}")
        result_text = "Ошибка: Результат не найден. Возможно, файл был удален или произошла ошибка при обработке."
    except Exception as e:
         print(f"ERROR: Could not read result file {absolute_path_result}: {e}")
         result_text = f"Ошибка при чтении файла результата: {e}"

    # Рендерим новый шаблон, передавая URL изображения и текст результата
    return render_template("result.html", image_url=image_url, result=result_text)


if __name__ == "__main__":
    app.run(debug=True)
