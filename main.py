import os
import io
import json
import time
import traceback

import telebot
from PIL import Image
from google import genai
from google.genai import types


# ============================================================
# НАСТРОЙКИ
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-3.6-flash")
MAX_ANALYSIS_ATTEMPTS = int(os.getenv("MAX_ANALYSIS_ATTEMPTS", "3"))

# Паузы между повторными запросами Gemini
RETRY_DELAYS = [3, 7, 15]

# Ограничение Telegram на одно сообщение.
TELEGRAM_CHUNK_SIZE = 3800

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN")

if not GEMINI_API_KEY:
    raise RuntimeError("Не задан GEMINI_API_KEY")


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# TELEGRAM
# ============================================================

telebot.apihelper.RETRY_ON_ERROR = True

bot = telebot.TeleBot(
    TELEGRAM_BOT_TOKEN
)


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def clean_json_response(raw_text):
    """Удаляет markdown-обёртку ```json ... ``` из ответа Gemini."""

    if not raw_text:
        raise Exception("Gemini вернул пустой ответ.")

    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


def parse_json(text):
    """Парсит JSON и выдаёт понятную ошибку."""

    cleaned = clean_json_response(text)

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError as error:
        raise Exception(
            "Gemini вернул некорректный JSON.\n\n"
            f"{cleaned[:5000]}"
        ) from error


def get_image_mime_type(image_bytes):
    """Определяет MIME изображения через PIL."""

    try:
        image = Image.open(io.BytesIO(image_bytes))
        fmt = (image.format or "JPEG").upper()

        mapping = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
            "GIF": "image/gif",
        }

        return mapping.get(fmt, "image/jpeg")

    except Exception:
        return "image/jpeg"


def normalize_image_for_ai(image_bytes):
    """Приводит изображение к JPEG для стабильного анализа."""

    try:
        image = Image.open(io.BytesIO(image_bytes))

        print("[IMAGE] Исходный формат:", image.format)
        print("[IMAGE] Размер:", image.size)

        rgb = image.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=95)

        result = buffer.getvalue()

        print("[IMAGE] JPEG:", len(result), "байт")

        return result

    except Exception as error:
        raise Exception(
            f"Не удалось подготовить изображение: {error}"
        ) from error


def get_aspect_ratio(image_bytes):
    """Определяет ближайшее реальное соотношение сторон."""

    try:
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        ratio = width / height

        candidates = {
            "1:1": 1.0,
            "4:5": 0.8,
            "3:4": 0.75,
            "2:3": 0.6667,
            "9:16": 0.5625,
            "5:4": 1.25,
            "4:3": 1.3333,
            "3:2": 1.5,
            "16:9": 1.7778,
        }

        closest = min(
            candidates.items(),
            key=lambda item: abs(item[1] - ratio)
        )

        return closest[0]

    except Exception:
        return "не удалось определить"


def safe_value(data, *keys, default=""):
    """Безопасное получение вложенного значения."""

    current = data

    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    if current is None:
        return default

    return current


def normalize_hashtags(value):
    """Приводит хэштеги к удобному виду."""

    if not value:
        return "#нейрофото #промпт #нейросеть"

    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = str(value).replace(",", " ").split()

    result = []

    for item in raw_items:
        item = str(item).strip()
        if not item:
            continue

        if not item.startswith("#"):
            item = "#" + item

        result.append(item)

    return " ".join(result[:12]) or "#нейрофото #промпт #нейросеть"


def split_for_telegram(text, max_length=TELEGRAM_CHUNK_SIZE):
    """Делит длинный текст на сообщения Telegram."""

    if len(text) <= max_length:
        return [text]

    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for paragraph in paragraphs:
        candidate = (
            paragraph
            if not current
            else current + "\n\n" + paragraph
        )

        if len(candidate) <= max_length:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        while len(paragraph) > max_length:
            chunks.append(paragraph[:max_length])
            paragraph = paragraph[max_length:]

        current = paragraph

    if current:
        chunks.append(current)

    return chunks


def send_long_message(chat_id, text):
    """Отправляет текст с учётом ограничения Telegram."""

    chunks = split_for_telegram(text)

    for index, chunk in enumerate(chunks, start=1):
        prefix = ""

        if len(chunks) > 1:
            prefix = f"Часть {index}/{len(chunks)}\n\n"

        bot.send_message(
            chat_id,
            prefix + chunk
        )


# ============================================================
# GEMINI 3.6
# АНАЛИЗ РЕФЕРЕНСА
# ============================================================

def analyze_reference(image_bytes):

    aspect_ratio = get_aspect_ratio(image_bytes)

    prompt = f"""
Ты — профессиональный visual analyst и prompt engineer
для фотореалистичной генерации изображений.

Проанализируй прикреплённый референс максимально подробно.

Главная задача:
не придумать новое изображение,
а восстановить максимально точное техническое описание
того, ЧТО УЖЕ находится в этом конкретном референсе.

ОЧЕНЬ ВАЖНО:
все параметры камеры и съёмки должны формироваться
исходя из анализа ИМЕННО ЭТОГО ФОТО.

Нельзя использовать одни и те же параметры камеры
для всех фотографий.

Определи по визуальному результату конкретного фото
наиболее вероятные:
- тип камеры;
- объектив / фокусное расстояние;
- диафрагму;
- ISO;
- выдержку;
- глубину резкости.

Если точные параметры невозможно определить,
укажи реалистичное предположение и не выдавай его
за достоверный факт.

Фактическое соотношение сторон исходного файла:
{aspect_ratio}

Особенно внимательно анализируй:
1. Композицию.
2. Кадрирование.
3. Соотношение сторон.
4. Размер и положение главного объекта.
5. Положение человека/людей.
6. Направление взгляда.
7. Положение головы.
8. Положение корпуса.
9. Положение рук.
10. Положение пальцев.
11. Положение ног.
12. Причёску.
13. Черты внешнего образа без выдумывания новых деталей.
14. Макияж.
15. Одежду.
16. Материалы и текстуры одежды.
17. Аксессуары.
18. Фон.
19. Предметы вокруг.
20. Перспективу.
21. Глубину резкости.
22. Освещение.
23. Направление света.
24. Тени.
25. Цветовую палитру.
26. Цветокоррекцию.
27. Атмосферу.
28. Предполагаемую камеру и объектив.
29. Возможные параметры съёмки.
30. Любой текст, логотипы или надписи на изображении.

Не придумывай детали, которых нет в фотографии.

Если в изображении присутствует текст,
обязательно укажи его содержание,
язык, расположение и визуальное оформление.

Верни ТОЛЬКО JSON.

Структура:

{{
  "photo_title": "",
  "composition": {{
    "aspect_ratio": "",
    "shot_type": "",
    "framing": "",
    "camera_angle": "",
    "subject_position": "",
    "perspective": ""
  }},
  "subject": {{
    "count": 0,
    "description": "",
    "position": "",
    "scale_in_frame": ""
  }},
  "face_and_expression": {{
    "head_position": "",
    "gaze": "",
    "expression": "",
    "makeup": "",
    "skin": ""
  }},
  "hair": {{
    "color": "",
    "length": "",
    "style": "",
    "details": ""
  }},
  "outfit": {{
    "description": "",
    "colors": "",
    "materials": "",
    "shoes": "",
    "accessories": ""
  }},
  "pose": {{
    "body": "",
    "head": "",
    "left_arm": "",
    "right_arm": "",
    "left_hand": "",
    "right_hand": "",
    "legs": "",
    "feet": ""
  }},
  "environment": {{
    "location": "",
    "background": "",
    "foreground": "",
    "objects": ""
  }},
  "lighting": {{
    "type": "",
    "source": "",
    "direction": "",
    "hardness": "",
    "shadows": "",
    "rim_light": ""
  }},
  "camera": {{
    "camera_type": "",
    "lens": "",
    "aperture": "",
    "iso": "",
    "shutter_speed": "",
    "depth_of_field": ""
  }},
  "color": {{
    "palette": "",
    "grading": "",
    "contrast": "",
    "saturation": "",
    "white_balance": ""
  }},
  "text_in_image": {{
    "present": false,
    "language": "",
    "content": "",
    "position": "",
    "style": ""
  }},
  "style": "",
  "quality": "",
  "hashtags": ""
}}
"""

    mime_type = get_image_mime_type(image_bytes)

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type
    )

    last_error = None

    for attempt in range(MAX_ANALYSIS_ATTEMPTS):
        try:
            print(
                f"[Gemini {TEXT_MODEL}] Анализ: "
                f"{attempt + 1}/{MAX_ANALYSIS_ATTEMPTS}"
            )

            response = client.models.generate_content(
                model=TEXT_MODEL,
                contents=[
                    image_part,
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            data = parse_json(response.text)

            if not isinstance(data.get("composition"), dict):
                data["composition"] = {}

            # Фактическое соотношение сторон файла имеет приоритет.
            data["composition"]["aspect_ratio"] = aspect_ratio

            print("[Gemini] JSON анализа получен.")

            return data

        except Exception as error:
            last_error = error

            print(
                "[Gemini] Ошибка анализа:",
                error
            )

            if attempt < MAX_ANALYSIS_ATTEMPTS - 1:
                time.sleep(RETRY_DELAYS[attempt])

    raise Exception(
        "Не удалось проанализировать референс: "
        f"{last_error}"
    )


# ============================================================
# СТАНДАРТНЫЙ ПРОМПТ
# ============================================================

def build_standard_prompt(data):

    composition = data.get("composition", {})
    subject = data.get("subject", {})
    face = data.get("face_and_expression", {})
    hair = data.get("hair", {})
    outfit = data.get("outfit", {})
    pose = data.get("pose", {})
    environment = data.get("environment", {})
    lighting = data.get("lighting", {})
    camera = data.get("camera", {})
    color = data.get("color", {})
    text_info = data.get("text_in_image", {})

    prompt = f"""[📌](https://vk.ru/emoji/e/f09f938c.png) Промпт для генерации:

Внешность должна полностью соответствовать прикреплённому референсу: идентичные черты лица, возраст, рост, форма лица, цвет глаз, цвет и длина волос, причёска, телосложение, пропорции, макияж, выражение лица и общее визуальное впечатление.

Любые изменения внешности, стилизация под другого человека или искажение типажа недопустимы.

Стиль: {safe_value(data, "style")}

Камера и параметры: {safe_value(camera, "camera_type")}, {safe_value(camera, "lens")}, {safe_value(camera, "aperture")}, {safe_value(camera, "shutter_speed")}, ISO {safe_value(camera, "iso")}, глубина резкости: {safe_value(camera, "depth_of_field")}

{safe_value(composition, "shot_type")}, {safe_value(composition, "aspect_ratio")} кадр, {safe_value(composition, "camera_angle")}. {safe_value(subject, "description")} Положение в кадре: {safe_value(subject, "position")}. Перспектива: {safe_value(composition, "perspective")}.

Причёска, кожа и макияж: волосы {safe_value(hair, "color")}, {safe_value(hair, "length")}, укладка {safe_value(hair, "style")}. Детали волос: {safe_value(hair, "details")}. Кожа: {safe_value(face, "skin")}. Макияж: {safe_value(face, "makeup")}.

Образ: {safe_value(outfit, "description")}. Цвета: {safe_value(outfit, "colors")}. Материалы и фактуры: {safe_value(outfit, "materials")}. Обувь: {safe_value(outfit, "shoes")}. Аксессуары: {safe_value(outfit, "accessories")}.

Поза: корпус — {safe_value(pose, "body")}; голова — {safe_value(pose, "head")}; левая рука — {safe_value(pose, "left_arm")}; правая рука — {safe_value(pose, "right_arm")}; левая кисть — {safe_value(pose, "left_hand")}; правая кисть — {safe_value(pose, "right_hand")}; ноги — {safe_value(pose, "legs")}; стопы — {safe_value(pose, "feet")}.

Освещение: {safe_value(lighting, "type")}. Источник: {safe_value(lighting, "source")}. Направление: {safe_value(lighting, "direction")}. Жёсткость: {safe_value(lighting, "hardness")}. Тени: {safe_value(lighting, "shadows")}. Контровой свет: {safe_value(lighting, "rim_light")}.

Фон: локация — {safe_value(environment, "location")}; фон — {safe_value(environment, "background")}; передний план — {safe_value(environment, "foreground")}; предметы — {safe_value(environment, "objects")}.

Цветокор: палитра — {safe_value(color, "palette")}; грейдинг — {safe_value(color, "grading")}; контраст — {safe_value(color, "contrast")}; насыщенность — {safe_value(color, "saturation")}; баланс белого — {safe_value(color, "white_balance")}.

Текст на изображении: наличие — {safe_value(text_info, "present")}; язык — {safe_value(text_info, "language")}; содержание — {safe_value(text_info, "content")}; положение — {safe_value(text_info, "position")}; стиль — {safe_value(text_info, "style")}.

Сохрани оригинальную композицию, геометрию, ракурс, кадрирование, расположение и масштаб объектов, позу, освещение, фон и цветовую логику референса.

Не добавляй элементы, которых нет в референсе.
Не удаляй важные элементы референса.

Качество и реализм:
Максимальный фотореализм. Высокая детализация. Естественная анатомия. Реалистичная кожа. Натуральная текстура кожи. Реалистичные волосы. Реалистичные глаза. Естественный свет. Реалистичные тени.

Негативный промпт:
Без CGI. Без мультяшности. Без пластиковой кожи. Без деформаций. Без лишних пальцев. Без лишних конечностей. Без анатомических ошибок."""

    return prompt.strip()


# ============================================================
# WALL POST TEXT
# ЛОГИКА И ОФОРМЛЕНИЕ ИЗ ИСХОДНОГО ДОКУМЕНТА
# ============================================================

def build_wall_post_text(data):

    title = data.get(
        "photo_title",
        "Нейрофотосессия"
    )

    hashtags = normalize_hashtags(
        data.get(
            "hashtags",
            "#нейрофото #промпт #нейросеть"
        )
    )

    return f"""✨ {title}

ИНСТРУКЦИЯ

КАК СОЗДАТЬ ФОТО С ПОМОЩЬЮ БОТОВ 🖤🖤🖤🖤

🔹 БОТ 1 ВК — GPTron Nano Banana Pro🍌✅
1️⃣ Переходим в бот ➡️https://vk.com/write-236453790?ref=pp53aacd7d52

🔹 БОТ 2 ВК — Lexy Nano Banana Pro🍌✅
Переходим в бот ➡️https://vk.com/write-233546714?ref=84372609_add

Отправляем своё фото.

Перед отправкой вставляем нужный промт в комментариях.

ВАЖНО ПРОЧИТАТЬ

❗ Промт всегда можно и нужно менять под себя:
цвет волос, глаз, одежду, позу, настроение и т.д.

👇 Забирай готовый промпт для генерации в комментариях к этому посту!

{hashtags}"""


# ============================================================
# ИТОГОВОЕ СООБЩЕНИЕ В TELEGRAM
# ============================================================

def send_final_result(message, analysis_json, standard_prompt):

    title = analysis_json.get(
        "photo_title",
        "Нейрофотосессия"
    )

    wall_text = build_wall_post_text(
        analysis_json
    )

    comment_text = standard_prompt

    json_text = json.dumps(
        analysis_json,
        ensure_ascii=False,
        indent=2
    )

    # 1. Заголовок результата
    bot.send_message(
        message.chat.id,
        "🎉 ГОТОВО!\n\n"
        f"✨ {title}\n\n"
        "Ниже всё подготовлено для ручной публикации в VK.\n"
        "Фото оставляешь исходное, текст поста копируешь из блока №2,\n"
        "промпт — из блока №4."
    )

    # 2. Пост VK
    bot.send_message(
        message.chat.id,
        "📝 БЛОК 1 — ГОТОВЫЙ ПОСТ VK"
    )
    send_long_message(
        message.chat.id,
        wall_text
    )

    # 3. JSON
    bot.send_message(
        message.chat.id,
        "📦 БЛОК 2 — JSON-АНАЛИЗ"
    )
    send_long_message(
        message.chat.id,
        json_text
    )

    # 4. Промпт
    bot.send_message(
        message.chat.id,
        "📌 БЛОК 3 — ГОТОВЫЙ ПРОМПТ ДЛЯ КОММЕНТАРИЯ VK"
    )
    send_long_message(
        message.chat.id,
        comment_text
    )

    # 5. Финальная инструкция
    bot.send_message(
        message.chat.id,
        "✅ ГОТОВО\n\n"
        "1️⃣ Берёшь исходное фото.\n"
        "2️⃣ Публикуешь его в своей группе VK.\n"
        "3️⃣ Копируешь БЛОК 1 в текст поста.\n"
        "4️⃣ Копируешь БЛОК 3 в комментарий.\n\n"
        "🤖 Бот ничего не публикует в VK автоматически."
    )


# ============================================================
# TELEGRAM START
# ============================================================

@bot.message_handler(commands=["start"])
def send_welcome(message):

    bot.reply_to(
        message,
        "👋 Привет!\n\n"
        "Отправь мне понравившееся фото.\n\n"
        "Я:\n"
        "1️⃣ Изучу референс\n"
        "2️⃣ Создам JSON-анализ\n"
        "3️⃣ Сформирую стандартный промпт\n"
        "4️⃣ Подготовлю готовый текст поста VK\n"
        "5️⃣ Подготовлю готовый промпт для комментария\n\n"
        "📌 VK автоматически не публикую.\n"
        "Ты сам вставляешь фото и готовый текст в свою группу."
    )


# ============================================================
# TELEGRAM PHOTO HANDLER
# ============================================================

@bot.message_handler(
    content_types=[
        "photo",
        "document"
    ]
)
def handle_photo(message):

    status_msg = bot.reply_to(
        message,
        "⏳ Получил референс.\n"
        "Начинаю глубокий анализ..."
    )

    try:
        # ----------------------------------------------------
        # 1. Telegram file_id
        # ----------------------------------------------------

        if message.photo:
            file_id = message.photo[-1].file_id

        elif message.document:
            file_id = message.document.file_id

        else:
            raise Exception("Фотография не найдена.")

        # ----------------------------------------------------
        # 2. Download
        # ----------------------------------------------------

        file_info = bot.get_file(file_id)

        original_bytes = bot.download_file(
            file_info.file_path
        )

        if not original_bytes:
            raise Exception(
                "Не удалось скачать фото из Telegram."
            )

        print(
            "[Telegram] Получено:",
            len(original_bytes),
            "байт"
        )

        # ----------------------------------------------------
        # 3. Нормализация
        # ----------------------------------------------------

        reference_bytes = normalize_image_for_ai(
            original_bytes
        )

        # ----------------------------------------------------
        # 4. ANALYSIS
        # ----------------------------------------------------

        bot.edit_message_text(
            "🔍 Анализирую конкретный референс через "
            f"{TEXT_MODEL}...",
            chat_id=message.chat.id,
            message_id=status_msg.message_id
        )

        analysis_json = analyze_reference(
            reference_bytes
        )

        print(
            "[SYSTEM] JSON анализа:\n",
            json.dumps(
                analysis_json,
                ensure_ascii=False,
                indent=2
            )
        )

        # ----------------------------------------------------
        # 5. STANDARD PROMPT
        # ----------------------------------------------------

        bot.edit_message_text(
            "📝 JSON готов.\n"
            "Формирую стандартный промпт и пост VK...",
            chat_id=message.chat.id,
            message_id=status_msg.message_id
        )

        current_prompt = build_standard_prompt(
            analysis_json
        )

        # ----------------------------------------------------
        # 6. FINAL RESULT
        # ----------------------------------------------------

        bot.delete_message(
            message.chat.id,
            status_msg.message_id
        )

        send_final_result(
            message,
            analysis_json,
            current_prompt
        )

    except Exception as error:

        error_text = str(error)

        print(
            "\n================ ERROR ================\n"
        )
        print(error_text)
        traceback.print_exc()
        print(
            "========================================\n"
        )

        try:
            bot.edit_message_text(
                "❌ Произошла ошибка:\n\n"
                f"{error_text}",
                chat_id=message.chat.id,
                message_id=status_msg.message_id
            )

        except Exception:
            bot.reply_to(
                message,
                "❌ Произошла ошибка:\n\n"
                f"{error_text}"
            )


# ============================================================
# ЗАПУСК
# ============================================================

print(
    "=================================================="
)
print("✅ TELEGRAM БОТ ЗАПУЩЕН")
print(f"Gemini Text: {TEXT_MODEL}")
print("VK API: ОТКЛЮЧЁН — ручная публикация")
print("Генерация изображений: ОТКЛЮЧЕНА")
print("Ожидаю референс...")
print(
    "=================================================="
)


# ============================================================
# TELEGRAM POLLING
# ============================================================

while True:
    try:
        bot.infinity_polling(
            timeout=120,
            long_polling_timeout=120,
            skip_pending=True
        )

    except Exception as error:
        print(
            "[Telegram] Ошибка polling:",
            error
        )
        time.sleep(5)
