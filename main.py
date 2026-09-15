import os
import io
import json
import time
import logging
from pathlib import Path

import requests
from PIL import Image
from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

YANDEX_TOKEN = os.getenv("YANDEX_TOKEN")
YANDEX_FOLDER = os.getenv(
    "YANDEX_FOLDER",
    "disk:/Нейрофото"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN"
)

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID"
)

# 2 часа
INTERVAL_SECONDS = 2 * 60 * 60

# Gemini
GEMINI_MODEL = "gemini-3.6-flash"

# HTTP
HTTP_TIMEOUT = 120

# Максимальное число попыток Gemini
GEMINI_RETRIES = 3

# Паузы
RETRY_DELAYS = [5, 15, 30]

# Разрешённые изображения
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ============================================================
# CHECK CONFIG
# ============================================================

required_settings = {
    "YANDEX_TOKEN": YANDEX_TOKEN,
    "YANDEX_FOLDER": YANDEX_FOLDER,
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
    "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
}

missing_settings = [
    name
    for name, value in required_settings.items()
    if not value
]

if missing_settings:
    raise RuntimeError(
        "Не заданы переменные окружения:\n"
        + "\n".join(
            f"- {item}"
            for item in missing_settings
        )
    )


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(
    "yandex-gemini-worker"
)


# ============================================================
# GEMINI
# ============================================================

gemini = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# YANDEX DISK
# ============================================================

YANDEX_API = (
    "https://cloud-api.yandex.net/v1/disk"
)


def yandex_headers():

    return {
        "Authorization": (
            f"OAuth {YANDEX_TOKEN}"
        )
    }


def yandex_get_folder():

    url = (
        f"{YANDEX_API}/resources"
    )

    params = {
        "path": YANDEX_FOLDER,
        "limit": 100,
        "sort": "path"
    }

    response = requests.get(
        url,
        headers=yandex_headers(),
        params=params,
        timeout=HTTP_TIMEOUT
    )

    response.raise_for_status()

    return response.json()


def get_first_image():

    data = yandex_get_folder()

    embedded = data.get(
        "_embedded",
        {}
    )

    items = embedded.get(
        "items",
        []
    )

    images = []

    for item in items:

        if item.get(
            "type"
        ) != "file":
            continue

        path = item.get(
            "path"
        )

        if not path:
            continue

        extension = (
            Path(path)
            .suffix
            .lower()
        )

        if extension not in IMAGE_EXTENSIONS:
            continue

        images.append(item)

    if not images:
        return None

    # Сортируем по имени/пути,
    # чтобы обработка была предсказуемой
    images.sort(
        key=lambda item: item.get(
            "path",
            ""
        )
    )

    return images[0]


def yandex_download_file(path):

    url = (
        f"{YANDEX_API}/resources/download"
    )

    params = {
        "path": path
    }

    response = requests.get(
        url,
        headers=yandex_headers(),
        params=params,
        timeout=HTTP_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    href = data.get(
        "href"
    )

    if not href:
        raise RuntimeError(
            "Яндекс.Диск не вернул ссылку "
            "на скачивание."
        )

    file_response = requests.get(
        href,
        timeout=HTTP_TIMEOUT,
        allow_redirects=True
    )

    file_response.raise_for_status()

    return file_response.content


def yandex_delete_file(path):

    url = (
        f"{YANDEX_API}/resources"
    )

    params = {
        "path": path,
        "permanently": "true"
    }

    response = requests.delete(
        url,
        headers=yandex_headers(),
        params=params,
        timeout=HTTP_TIMEOUT
    )

    response.raise_for_status()

    logger.info(
        "Файл удалён с Яндекс.Диска: %s",
        path
    )


# ============================================================
# IMAGE
# ============================================================

def normalize_image(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    logger.info(
        "Исходное изображение: "
        "%s %s",
        image.format,
        image.size
    )

    rgb = image.convert(
        "RGB"
    )

    buffer = io.BytesIO()

    rgb.save(
        buffer,
        format="JPEG",
        quality=95
    )

    return buffer.getvalue()


# ============================================================
# GEMINI ANALYSIS
# ============================================================

ANALYSIS_PROMPT = """
Ты — профессиональный аналитик изображений
и prompt engineer.

Проанализируй прикреплённое изображение максимально подробно.

Главная задача:
описать именно то, что находится на фотографии,
чтобы другой генератор изображения мог
максимально точно воспроизвести сцену.

Не придумывай элементов,
которых нет на изображении.

Особенно подробно проанализируй:

- сюжет;
- главный объект;
- количество людей;
- положение человека в кадре;
- кадрирование;
- ракурс;
- перспективу;
- позу;
- положение головы;
- направление взгляда;
- руки и пальцы;
- ноги;
- волосы;
- выражение лица;
- одежду;
- обувь;
- аксессуары;
- фон;
- предметы;
- освещение;
- направление света;
- тени;
- глубину резкости;
- цветовую палитру;
- цветокоррекцию;
- визуальный стиль;
- предполагаемую камеру;
- объектив;
- предполагаемые параметры съёмки;
- текст или надписи.

Если точные параметры камеры неизвестны,
укажи реалистичные предположения.

Если на изображении есть текст,
обязательно опиши его.

Верни строго JSON.

Структура:

{
  "photo_title": "",
  "photo_style": "",
  "camera_and_settings": "",
  "shot_type_and_pose_intro": "",
  "hairstyle_and_makeup": "",
  "outfit": "",
  "pose_details": "",
  "lighting": "",
  "background": "",
  "color_grading_and_style": "",
  "quality_and_style_tags": "",
  "hashtags": "",
  "text_in_image": ""
}
"""


def analyze_photo(image_bytes):

    last_error = None

    mime_type = "image/jpeg"

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type
    )

    for attempt in range(
        GEMINI_RETRIES
    ):

        try:

            logger.info(
                "Gemini: анализ "
                "попытка %s/%s",
                attempt + 1,
                GEMINI_RETRIES
            )

            response = (
                gemini.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        image_part,
                        ANALYSIS_PROMPT
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type=(
                            "application/json"
                        )
                    )
                )
            )

            raw = (
                response.text or ""
            ).strip()

            if not raw:
                raise RuntimeError(
                    "Gemini вернул пустой ответ."
                )

            # Если вдруг модель добавила markdown
            if raw.startswith("```"):

                lines = raw.splitlines()

                if (
                    lines
                    and lines[0].startswith("```")
                ):
                    lines = lines[1:]

                if (
                    lines
                    and lines[-1].strip() == "```"
                ):
                    lines = lines[:-1]

                raw = "\n".join(
                    lines
                ).strip()

            result = json.loads(
                raw
            )

            logger.info(
                "Gemini: анализ успешно получен."
            )

            return result

        except Exception as e:

            last_error = e

            logger.exception(
                "Ошибка Gemini"
            )

            if attempt < (
                GEMINI_RETRIES - 1
            ):

                time.sleep(
                    RETRY_DELAYS[attempt]
                )

    raise RuntimeError(
        "Gemini не смог обработать "
        f"изображение: {last_error}"
    )


# ============================================================
# STANDARD PROMPT
# ============================================================

def build_standard_prompt(data):

    return f"""📌 Промпт для генерации:

Внешность должна полностью соответствовать
прикреплённому референсу:
идентичные черты лица, возраст, рост,
форма лица, цвет глаз, цвет и длина волос,
причёска, телосложение, пропорции, макияж,
выражение лица и общее визуальное впечатление.

Любые изменения внешности,
стилизация под другого человека
или искажение типажа недопустимы.

Сюжет:
{data.get("photo_title", "")}

Стиль:
{data.get("photo_style", "")}

Камера и параметры:
{data.get("camera_and_settings", "")}

Ракурс и положение:
{data.get("shot_type_and_pose_intro", "")}

Причёска и макияж:
{data.get("hairstyle_and_makeup", "")}

Одежда:
{data.get("outfit", "")}

Поза:
{data.get("pose_details", "")}

Освещение:
{data.get("lighting", "")}

Фон:
{data.get("background", "")}

Цветокоррекция:
{data.get("color_grading_and_style", "")}

Текст на изображении:
{data.get("text_in_image", "")}

Качество:
{data.get(
    "quality_and_style_tags",
    "Максимальный фотореализм, "
    "высокая детализация, естественная "
    "анатомия, реалистичная кожа, "
    "без CGI, без мультяшности, "
    "без артефактов."
)}
""".strip()


# ============================================================
# TELEGRAM
# ============================================================

def telegram_send_message(text):

    url = (
        "https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True
    }

    response = requests.post(
        url,
        json=payload,
        timeout=HTTP_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("ok"):
        raise RuntimeError(
            f"Telegram API error: {data}"
        )


# ============================================================
# PROCESS ONE PHOTO
# ============================================================

def process_one_photo():

    item = get_first_image()

    if not item:

        logger.info(
            "В папке нет новых изображений."
        )

        return False

    path = item.get(
        "path"
    )

    name = item.get(
        "name",
        Path(path).name
    )

    logger.info(
        "Выбрано фото: %s",
        path
    )

    # --------------------------------------------------------
    # Telegram: начало
    # --------------------------------------------------------

    telegram_send_message(
        "🚀 Начинаю обработку фото\n\n"
        f"📁 Файл: {name}\n"
        f"📂 Папка: {YANDEX_FOLDER}\n\n"
        "🔍 Шаг 1/4 — скачивание и анализ..."
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    image_bytes = (
        yandex_download_file(
            path
        )
    )

    normalized_bytes = (
        normalize_image(
            image_bytes
        )
    )

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    analysis = analyze_photo(
        normalized_bytes
    )

    prompt = build_standard_prompt(
        analysis
    )

    title = analysis.get(
        "photo_title",
        "Без названия"
    )

    hashtags = analysis.get(
        "hashtags",
        ""
    )

    # --------------------------------------------------------
    # Telegram result
    # --------------------------------------------------------

    text = (
        "✅ Фото успешно обработано\n\n"
        f"📁 Файл: {name}\n"
        f"📌 Название: {title}\n\n"
        "🔍 Анализ Gemini 3.6 завершён\n"
        "📝 Стандартный промпт создан\n\n"
        "🏷 Хэштеги:\n"
        f"{hashtags}\n\n"
        "📂 Источник:\n"
        f"{path}\n\n"
        "🗑 Исходный файл будет удалён "
        "с Яндекс.Диска после успешного "
        "завершения обработки."
    )

    telegram_send_message(
        text
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    telegram_send_message(
        "📝 Готовый промпт:\n\n"
        + prompt
    )

    # --------------------------------------------------------
    # DELETE
    # --------------------------------------------------------

    yandex_delete_file(
        path
    )

    telegram_send_message(
        "🎉 Готово!\n\n"
        f"✅ Обработано: {name}\n"
        "✅ Gemini-анализ создан\n"
        "✅ Промпт создан\n"
        "✅ Исходник удалён с Яндекс.Диска\n\n"
        "📌 VK пока не используется — "
        "подключим публикацию после решения "
        "вопроса с VK API."
    )

    return True


# ============================================================
# WORKER
# ============================================================

def main():

    logger.info(
        "================================================"
    )

    logger.info(
        "🚀 Yandex → Gemini worker запущен"
    )

    logger.info(
        "Папка: %s",
        YANDEX_FOLDER
    )

    logger.info(
        "Интервал: 2 часа"
    )

    logger.info(
        "За один запуск: 1 фото"
    )

    logger.info(
        "Gemini: %s",
        GEMINI_MODEL
    )

    logger.info(
        "================================================"
    )

    while True:

        started_at = time.time()

        try:

            process_one_photo()

        except Exception as e:

            logger.exception(
                "Ошибка обработки"
            )

            try:

                telegram_send_message(
                    "❌ Автоматическая обработка "
                    "завершилась ошибкой.\n\n"
                    f"{e}\n\n"
                    "⚠️ Исходный файл НЕ удалён."
                )

            except Exception:

                logger.exception(
                    "Не удалось отправить "
                    "ошибку в Telegram"
                )

        elapsed = (
            time.time() - started_at
        )

        sleep_seconds = max(
            1,
            INTERVAL_SECONDS - elapsed
        )

        logger.info(
            "Следующая проверка через "
            "%s минут.",
            round(
                sleep_seconds / 60,
                1
            )
        )

        time.sleep(
            sleep_seconds
        )


if __name__ == "__main__":
    main()
