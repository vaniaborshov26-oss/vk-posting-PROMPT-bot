import os
import io
import json
import time
import traceback

import telebot
from PIL import Image
from google import genai
from google.genai import types

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-3.6-flash")
MAX_ANALYSIS_ATTEMPTS = int(os.getenv("MAX_ANALYSIS_ATTEMPTS", "3"))
RETRY_DELAYS = [3, 7, 15]
TG_CHUNK = 3800

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN")
if not GEMINI_API_KEY:
    raise RuntimeError("Не задан GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
telebot.apihelper.RETRY_ON_ERROR = True
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)


def clean_json(text):
    if not text:
        raise ValueError("Gemini вернул пустой ответ")
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_json(text):
    try:
        data = json.loads(clean_json(text))
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini вернул некорректный JSON:\n\n{clean_json(text)[:4000]}") from e
    if not isinstance(data, dict):
        raise ValueError("JSON должен быть объектом")
    return data


def normalize_image(data):
    try:
        img = Image.open(io.BytesIO(data))
        print("[IMAGE]", img.format, img.size)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except Exception as e:
        raise ValueError(f"Не удалось подготовить изображение: {e}") from e


def aspect_ratio(data):
    try:
        w, h = Image.open(io.BytesIO(data)).size
        r = w / h
        candidates = {
            "1:1": 1.0, "4:5": .8, "3:4": .75, "2:3": .6667,
            "9:16": .5625, "5:4": 1.25, "4:3": 1.3333,
            "3:2": 1.5, "16:9": 1.7778
        }
        return min(candidates, key=lambda k: abs(candidates[k] - r))
    except Exception:
        return "не удалось определить"


def mime(data):
    try:
        fmt = (Image.open(io.BytesIO(data)).format or "JPEG").upper()
        return {"JPEG":"image/jpeg", "JPG":"image/jpeg", "PNG":"image/png", "WEBP":"image/webp", "GIF":"image/gif"}.get(fmt, "image/jpeg")
    except Exception:
        return "image/jpeg"


def v(d, *keys, default=""):
    x = d
    for k in keys:
        if not isinstance(x, dict): return default
        x = x.get(k)
    return default if x is None else str(x)


def hashtags(raw):
    raw = raw or "#нейрофото #промпт #нейросеть"
    out = []
    for x in str(raw).replace(",", " ").split():
        x = x if x.startswith("#") else "#" + x.lstrip("#")
        out.append(x)
    return " ".join(out[:12]) or "#нейрофото #промпт #нейросеть"


def send_long(chat_id, text):
    if len(text) <= TG_CHUNK:
        bot.send_message(chat_id, text)
        return
    paragraphs = text.split("\n\n")
    parts, cur = [], ""
    for p in paragraphs:
        candidate = p if not cur else cur + "\n\n" + p
        if len(candidate) <= TG_CHUNK:
            cur = candidate
        else:
            if cur: parts.append(cur)
            while len(p) > TG_CHUNK:
                parts.append(p[:TG_CHUNK]); p = p[TG_CHUNK:]
            cur = p
    if cur: parts.append(cur)
    for i, part in enumerate(parts, 1):
        bot.send_message(chat_id, (f"Часть {i}/{len(parts)}\n\n" if len(parts) > 1 else "") + part)


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




if __name__ == "__main__":
    print(f"🤖 Бот запущен. TEXT_MODEL={TEXT_MODEL}")
    bot.infinity_polling(skip_pending=True)
