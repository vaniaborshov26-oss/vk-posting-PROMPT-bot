import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from google import genai
from google.genai import types


# ============================================================
# НАСТРОЙКИ
# ============================================================

# -------------------------
# VK
# -------------------------

VK_API_VERSION = "5.131"
VK_GROUP_ID = os.getenv("VK_GROUP_ID")
VK_ACCESS_TOKEN = os.getenv("VK_ACCESS_TOKEN")


# -------------------------
# YANDEX DISK
# -------------------------

YANDEX_TOKEN = os.getenv("YANDEX_TOKEN")

YANDEX_FOLDER = os.getenv(
    "YANDEX_FOLDER",
    "disk:/Нейрофото"
)


# -------------------------
# GEMINI
# -------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = "gemini-3.6-flash"


# -------------------------
# TELEGRAM
# -------------------------

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# -------------------------
# ПРОЧИЕ НАСТРОЙКИ
# -------------------------

CHECK_INTERVAL = 2 * 60 * 60  # 2 часа

YANDEX_API = "https://cloud-api.yandex.net/v1/disk"

TELEGRAM_API = (
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
)


# ============================================================
# ЛОГИ
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# ПРОВЕРКА ПЕРЕМЕННЫХ
# ============================================================

def check_env():

    required = {
        "YANDEX_TOKEN": YANDEX_TOKEN,
        "YANDEX_FOLDER": YANDEX_FOLDER,
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
        "VK_ACCESS_TOKEN": VK_ACCESS_TOKEN,
        "VK_GROUP_ID": VK_GROUP_ID,
    }

    missing = [
        name
        for name, value in required.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Не заданы переменные окружения: "
            + ", ".join(missing)
        )

    logger.info("Все необходимые переменные окружения найдены.")


# ============================================================
# TELEGRAM
# ============================================================

def telegram_send(text: str):

    url = f"{TELEGRAM_API}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "disable_web_page_preview": True,
        },
        timeout=30,
    )

    # В случае ошибки Telegram выводим его настоящий ответ
    # в лог, чтобы было понятно, что именно произошло.
    if not response.ok:
        logger.error(
            "Telegram error: HTTP %s | %s",
            response.status_code,
            response.text,
        )

    response.raise_for_status()


# ============================================================
# VK — ТЕСТ АВТОРИЗАЦИИ
# ============================================================

def test_vk():

    logger.info("")
    logger.info("======================================")
    logger.info("          VK AUTH TEST")
    logger.info("======================================")

    if not VK_ACCESS_TOKEN:
        logger.error("VK_ACCESS_TOKEN не задан.")
        return False

    if not VK_GROUP_ID:
        logger.error("VK_GROUP_ID не задан.")
        return False

    logger.info("VK_GROUP_ID: %s", VK_GROUP_ID)
    logger.info("VK API VERSION: %s", VK_API_VERSION)
    logger.info("Проверяем доступ к VK...")

    url = "https://api.vk.com/method/groups.getById"

    params = {
        "group_id": VK_GROUP_ID,
        "access_token": VK_ACCESS_TOKEN,
        "v": VK_API_VERSION,
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=30,
        )

        logger.info(
            "VK HTTP STATUS: %s",
            response.status_code,
        )

        logger.info(
            "VK RESPONSE: %s",
            response.text,
        )

        if not response.ok:
            logger.error(
                "VK вернул HTTP ошибку."
            )
            return False

        data = response.json()

        # VK API вернул ошибку
        if "error" in data:

            error = data["error"]

            logger.error(
                "VK ERROR CODE: %s",
                error.get("error_code"),
            )

            logger.error(
                "VK ERROR MESSAGE: %s",
                error.get("error_msg"),
            )

            logger.error(
                "Полная ошибка VK: %s",
                error,
            )

            logger.info(
                "======================================"
            )

            return False

        # Успешный ответ
        if "response" in data:

            logger.info(
                "VK AUTH OK"
            )

            logger.info(
                "Доступ к группе получен."
            )

            logger.info(
                "VK RESPONSE DATA: %s",
                data["response"],
            )

            logger.info(
                "======================================"
            )

            return True

        logger.warning(
            "VK вернул неожиданный ответ: %s",
            data,
        )

        logger.info(
            "======================================"
        )

        return False

    except Exception as e:

        logger.exception(
            "Ошибка при проверке VK: %s",
            e,
        )

        logger.info(
            "======================================"
        )

        return False


# ============================================================
# YANDEX DISK
# ============================================================

def yandex_headers():

    return {
        "Authorization": f"OAuth {YANDEX_TOKEN}"
    }


def get_yandex_files():

    url = f"{YANDEX_API}/resources"

    response = requests.get(
        url,
        headers=yandex_headers(),
        params={
            "path": YANDEX_FOLDER,
            "limit": 100,
            "sort": "name",
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    return data.get(
        "_embedded",
        {}
    ).get(
        "items",
        []
    )


def select_image(files):

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".heic",
    }

    for item in files:

        if item.get("type") != "file":
            continue

        name = item.get("name", "")

        extension = Path(
            name
        ).suffix.lower()

        if extension in image_extensions:
            return item

    return None


def download_yandex_file(path: str) -> bytes:

    response = requests.get(
        f"{YANDEX_API}/resources/download",
        headers=yandex_headers(),
        params={
            "path": path
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    download_url = data["href"]

    file_response = requests.get(
        download_url,
        timeout=120,
    )

    file_response.raise_for_status()

    return file_response.content


def delete_yandex_file(path: str):

    response = requests.delete(
        f"{YANDEX_API}/resources",
        headers=yandex_headers(),
        params={
            "path": path,
            "permanently": "true",
        },
        timeout=30,
    )

    response.raise_for_status()


# ============================================================
# IMAGE
# ============================================================

def normalize_image(file_bytes: bytes) -> bytes:

    image = Image.open(
        io.BytesIO(file_bytes)
    )

    if image.mode not in (
        "RGB",
        "RGBA",
    ):
        image = image.convert("RGB")

    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=95,
    )

    return output.getvalue()


# ============================================================
# GEMINI
# ============================================================

def analyze_image(
    image_bytes: bytes
) -> dict[str, Any]:

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    prompt = """
Ты профессиональный аналитик изображений
и специалист по созданию фотореалистичных
промптов для генерации изображений.

Внимательно проанализируй предоставленную фотографию.

Определи:

1. Кто изображён на фотографии.
2. Пол и примерный возраст.
3. Внешность человека.
4. Форму лица.
5. Волосы и причёску.
6. Цвет глаз.
7. Одежду.
8. Аксессуары.
9. Положение тела.
10. Положение головы.
11. Выражение лица.
12. Направление взгляда.
13. Положение рук.
14. Окружение.
15. Фон.
16. Передний и задний план.
17. Освещение.
18. Направление света.
19. Тени.
20. Цветовую гамму.
21. Композицию.
22. Ракурс камеры.
23. Предполагаемое фокусное расстояние.
24. Глубину резкости.
25. Размытие фона.
26. Атмосферу.
27. Фотографический стиль.
28. Качество изображения.

После анализа создай подробный
фотореалистичный промпт на русском языке.

Особенно важно сохранить:

- внешность;
- черты лица;
- возраст;
- цвет глаз;
- цвет и длину волос;
- причёску;
- телосложение;
- пропорции;
- выражение лица;
- положение головы;
- позу;
- одежду;
- композицию;
- свет;
- тени;
- атмосферу.

Промпт должен начинаться строго словами:

"Внешность должна полностью соответствовать прикреплённому референсу:"

Не придумывай изменения внешности человека.

Ответ верни строго в JSON.
"""

    schema = {
        "type": "object",
        "properties": {

            "summary": {
                "type": "string"
            },

            "appearance": {
                "type": "string"
            },

            "clothing": {
                "type": "string"
            },

            "pose": {
                "type": "string"
            },

            "environment": {
                "type": "string"
            },

            "lighting": {
                "type": "string"
            },

            "camera": {
                "type": "string"
            },

            "atmosphere": {
                "type": "string"
            },

            "prompt": {
                "type": "string"
            },

            "hashtags": {
                "type": "string"
            },

        },

        "required": [
            "summary",
            "appearance",
            "clothing",
            "pose",
            "environment",
            "lighting",
            "camera",
            "atmosphere",
            "prompt",
            "hashtags",
        ],
    }

    response = client.models.generate_content(
        model=GEMINI_MODEL,

        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            ),

            prompt,
        ],

        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    if not response.text:

        raise RuntimeError(
            "Gemini вернул пустой ответ."
        )

    return json.loads(
        response.text
    )


# ============================================================
# ОБРАБОТКА ОДНОГО ФОТО
# ============================================================

def process_one_photo():

    logger.info(
        "Проверяем Яндекс.Диск..."
    )

    files = get_yandex_files()

    logger.info(
        "Файлов найдено: %s",
        len(files),
    )

    image = select_image(files)

    if not image:

        logger.info(
            "Фотографий для обработки нет."
        )

        telegram_send(
            "ℹ️ Проверка завершена\n\n"
            f"📁 Папка: {YANDEX_FOLDER}\n"
            "📷 Новых фотографий нет."
        )

        return

    file_name = image["name"]
    file_path = image["path"]

    logger.info(
        "Найдена фотография: %s",
        file_name,
    )

    telegram_send(
        "🚀 Начинаю обработку фото\n\n"
        f"📁 Папка: {YANDEX_FOLDER}\n"
        f"📷 Файл: {file_name}\n\n"
        "🔎 Шаг 1/4 — скачивание и анализ..."
    )

    # -----------------------------------
    # Скачивание
    # -----------------------------------

    original_bytes = download_yandex_file(
        file_path
    )

    # -----------------------------------
    # Нормализация
    # -----------------------------------

    image_bytes = normalize_image(
        original_bytes
    )

    # -----------------------------------
    # Gemini
    # -----------------------------------

    result = analyze_image(
        image_bytes
    )

    logger.info(
        "Gemini анализ завершён."
    )

    # -----------------------------------
    # Telegram
    # -----------------------------------

    first_message = (
        "✅ Фото успешно обработано\n\n"

        f"📁 Файл:\n"
        f"{file_name}\n\n"

        "📌 Название:\n"
        f"{result['summary']}\n\n"

        "🔎 Анализ Gemini 3.6 завершён\n"
        "📝 Стандартный промпт создан\n\n"

        "🏷 Хэштеги:\n"
        f"{result['hashtags']}\n\n"

        "📂 Источник:\n"
        f"{file_path}\n\n"

        "🗑 Исходный файл будет удалён "
        "после успешного завершения обработки."
    )

    telegram_send(
        first_message
    )

    # -----------------------------------
    # Готовый промпт
    # -----------------------------------

    prompt_message = (
        "📝 ГОТОВЫЙ ПРОМПТ\n\n"
        "📌 Промпт для генерации:\n\n"
        f"{result['prompt']}"
    )

    max_length = 4000

    if len(prompt_message) <= max_length:

        telegram_send(
            prompt_message
        )

    else:

        telegram_send(
            "📝 ГОТОВЫЙ ПРОМПТ\n\n"
            "Промпт длинный, поэтому отправляю "
            "его несколькими сообщениями."
        )

        text = result["prompt"]

        for start in range(
            0,
            len(text),
            max_length,
        ):

            telegram_send(
                text[
                    start:start + max_length
                ]
            )

    # -----------------------------------
    # Удаление исходника
    # -----------------------------------

    delete_yandex_file(
        file_path
    )

    logger.info(
        "Исходный файл удалён: %s",
        file_path,
    )

    telegram_send(
        "🎉 Готово!\n\n"

        f"✅ Обработано: {file_name}\n"
        "✅ Gemini-анализ создан\n"
        "✅ Промпт создан\n"
        "✅ Исходник удалён с Яндекс.Диска\n\n"

        "📌 VK пока не используется — "
        "сейчас проверяем VK API."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    check_env()

    # --------------------------------------------------------
    # ПРОВЕРКА VK
    # --------------------------------------------------------

    vk_ok = test_vk()

    if vk_ok:

        logger.info(
            "✅ VK: авторизация и доступ к группе работают."
        )

        telegram_send(
            "✅ Проверка VK пройдена\n\n"
            f"🏠 Группа ID: {VK_GROUP_ID}\n"
            "🔑 Доступ к VK API подтверждён.\n\n"
            "📌 Публикация пока не включена."
        )

    else:

        logger.error(
            "❌ VK: проверка не пройдена."
        )

        telegram_send(
            "⚠️ Проверка VK не пройдена\n\n"
            "Автоматизация Яндекс.Диск → "
            "Gemini → Telegram продолжит работу.\n\n"
            "📌 Публикация в VK пока отключена."
        )

    # --------------------------------------------------------
    # ЗАПУСК АВТОМАТИЗАЦИИ
    # --------------------------------------------------------

    telegram_send(
        "🚀 Автоматизация запущена\n\n"

        f"📁 Папка: {YANDEX_FOLDER}\n"
        "⏱ Интервал: 2 часа\n"
        "📷 За один запуск: 1 фотография\n"
        "🤖 Gemini: включён\n"
        f"🔵 VK: {'доступ есть' if vk_ok else 'проверка не пройдена'}\n\n"

        "⏳ Ожидаю фотографии..."
    )

    # --------------------------------------------------------
    # ОСНОВНОЙ ЦИКЛ
    # --------------------------------------------------------

    while True:

        try:

            process_one_photo()

        except Exception as e:

            logger.exception(
                "Ошибка обработки."
            )

            try:

                telegram_send(
                    "❌ Ошибка автоматизации\n\n"
                    f"{type(e).__name__}: {e}"
                )

            except Exception:

                logger.exception(
                    "Не удалось отправить сообщение "
                    "об ошибке в Telegram."
                )

        logger.info(
            "Следующая проверка через 2 часа..."
        )

        time.sleep(
            CHECK_INTERVAL
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
