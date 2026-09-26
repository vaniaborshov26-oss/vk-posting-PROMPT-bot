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


ANALYSIS_SCHEMA = r'''
{
  "photo_title": "",
  "composition": {"aspect_ratio":"","shot_type":"","framing":"","camera_angle":"","subject_position":"","perspective":""},
  "subject": {"count":0,"description":"","position":"","scale_in_frame":""},
  "face_and_expression": {"head_position":"","gaze":"","expression":"","makeup":"","skin":""},
  "hair": {"color":"","length":"","style":"","details":""},
  "outfit": {"description":"","colors":"","materials":"","shoes":"","accessories":""},
  "pose": {"body":"","head":"","left_arm":"","right_arm":"","left_hand":"","right_hand":"","legs":"","feet":""},
  "environment": {"location":"","background":"","foreground":"","objects":""},
  "lighting": {"type":"","source":"","direction":"","hardness":"","shadows":"","rim_light":""},
  "camera": {"camera_type":"","lens":"","aperture":"","iso":"","shutter_speed":"","depth_of_field":""},
  "color": {"palette":"","grading":"","contrast":"","saturation":"","white_balance":""},
  "text_in_image": {"present":false,"language":"","content":"","position":"","style":""},
  "style":"",
  "quality":"",
  "hashtags":""
}
'''


def analyze(data, ratio):
    prompt = f"""
Ты — профессиональный visual analyst и prompt engineer.
Проанализируй ИМЕННО прикреплённое фото максимально подробно. Не придумывай отсутствующие детали.
Главная задача — восстановить максимально точное описание того, что уже находится в референсе.

КРИТИЧЕСКИ ВАЖНО: параметры камеры и съёмки определяй ИНДИВИДУАЛЬНО ПО ЭТОЙ ФОТОГРАФИИ.
Не используй универсальные значения. Проанализируй визуальный результат и укажи наиболее вероятные
тип камеры, объектив/фокусное расстояние, диафрагму, ISO, выдержку и глубину резкости.
Если значение нельзя определить точно, укажи вероятное предположение.
Фактическое соотношение сторон файла: {ratio}.

Анализируй: композицию, кадрирование, план, ракурс, положение и масштаб объекта, перспективу,
голову, взгляд, выражение, волосы, макияж, кожу, одежду, материалы, обувь, аксессуары,
корпус, руки, кисти, пальцы, ноги, стопы, фон, передний план, предметы, свет, направление света,
жёсткость, тени, контровой свет, глубину резкости, палитру, цветокоррекцию, контраст,
насыщенность, баланс белого, атмосферу, стиль и весь текст/логотипы на изображении.

Верни ТОЛЬКО JSON следующей структуры:
{ANALYSIS_SCHEMA}
"""
    part = types.Part.from_bytes(data=data, mime_type=mime(data))
    last = None
    for attempt in range(MAX_ANALYSIS_ATTEMPTS):
        try:
            print(f"[GEMINI] Анализ {attempt+1}/{MAX_ANALYSIS_ATTEMPTS}")
            response = client.models.generate_content(
                model=TEXT_MODEL,
                contents=[part, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            result = parse_json(response.text)
            result.setdefault("composition", {})["aspect_ratio"] = ratio
            return result
        except Exception as e:
            last = e
            print("[GEMINI] Ошибка:", e)
            if attempt < MAX_ANALYSIS_ATTEMPTS - 1:
                time.sleep(RETRY_DELAYS[attempt])
    raise RuntimeError(f"Не удалось проанализировать референс: {last}")


def build_prompt(a):
    c, s, f, h, o, p, e, l, cam, col, txt = [a.get(k, {}) for k in (
        "composition","subject","face_and_expression","hair","outfit","pose","environment","lighting","camera","color","text_in_image")]
    return f"""Внешность должна полностью соответствовать прикреплённому референсу: идентичные черты лица, возраст, рост, форма лица, цвет глаз, цвет и длина волос, причёска, телосложение, пропорции, макияж, выражение лица и общее визуальное впечатление.

Любые изменения внешности, стилизация под другого человека или искажение типажа недопустимы.

Стиль: {v(a,'style')}
Камера и параметры: {v(cam,'camera_type')}, {v(cam,'lens')}, {v(cam,'aperture')}, {v(cam,'shutter_speed')}, ISO {v(cam,'iso')}, глубина резкости: {v(cam,'depth_of_field')}
{v(c,'shot_type')}, {v(c,'aspect_ratio')} кадр, {v(c,'camera_angle')}. {v(s,'description')} Положение: {v(s,'position')}. Перспектива: {v(c,'perspective')}.
Причёска, кожа и макияж: волосы {v(h,'color')}, {v(h,'length')}, укладка {v(h,'style')}. Детали: {v(h,'details')}. Кожа: {v(f,'skin')}. Макияж: {v(f,'makeup')}.
Образ: {v(o,'description')}. Цвета: {v(o,'colors')}. Материалы: {v(o,'materials')}. Обувь: {v(o,'shoes')}. Аксессуары: {v(o,'accessories')}.
Поза: корпус — {v(p,'body')}; голова — {v(p,'head')}; левая рука — {v(p,'left_arm')}; правая рука — {v(p,'right_arm')}; левая кисть — {v(p,'left_hand')}; правая кисть — {v(p,'right_hand')}; ноги — {v(p,'legs')}; стопы — {v(p,'feet')}.
Освещение: {v(l,'type')}. Источник: {v(l,'source')}. Направление: {v(l,'direction')}. Жёсткость: {v(l,'hardness')}. Тени: {v(l,'shadows')}. Контровой свет: {v(l,'rim_light')}.
Фон: {v(e,'location')}; фон — {v(e,'background')}; передний план — {v(e,'foreground')}; предметы — {v(e,'objects')}.
Цветокор: {v(col,'palette')}; грейдинг — {v(col,'grading')}; контраст — {v(col,'contrast')}; насыщенность — {v(col,'saturation')}; баланс белого — {v(col,'white_balance')}.
Текст на изображении: {v(txt,'content')} ({v(txt,'position')}).

Сохрани оригинальную композицию, геометрию, ракурс, кадрирование, расположение и масштаб объектов, позу, освещение, фон и цветовую логику референса.
Не добавляй элементы, которых нет в референсе.
Не удаляй важные элементы референса.

Качество и реализм:
Максимальный фотореализм. Высокая детализация. Естественная анатомия. Реалистичная кожа. Натуральная текстура кожи. Реалистичные волосы. Реалистичные глаза. Естественный свет. Реалистичные тени.

Негативный промпт:
Без CGI. Без мультяшности. Без пластиковой кожи. Без деформаций. Без лишних пальцев. Без лишних конечностей. Без анатомических ошибок.""".strip()


def build_vk_post(a):
    title = v(a, "photo_title", default="Нейрофотосессия")
    tags = hashtags(v(a, "hashtags", default="#нейрофото #промпт #нейросеть"))
    return f"""⚠️Берешь промпт — обязательно ставь лайк ❤️ на пост и делись результатом вашей генерации в комментарии!

КАК СОЗДАТЬ ФОТО С ПОМОЩЬЮ БОТОВ 🖤

🔹 БОТ 1 ВК — GPTron Nano Banana Pro 🍌✅
1️⃣ Переходим в бот:
https://vk.com/write-236453790?ref=pp53aacd7d52

🔹 БОТ 2 ВК — Lexy Nano Banana Pro 🍌✅
Переходим в бот:
https://vk.com/write-233546714?ref=84372609_add

Отправляем своё фото.
Выбираем модель генерации NANA BANANA PRO
Перед отправкой вставляем нужный промт в комментариях.

❗️ Промт всегда можно и нужно менять под себя:
цвет волос, глаз, одежду, позу, настроение и т.д.

👇 Забирай готовый промпт для генерации в комментариях к этому посту!

{tags}"""


def build_vk_comment(prompt):
    return "[📌](https://vk.ru/emoji/e/f09f938c.png) Промпт для генерации:\n\n" + prompt


@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(message, "👋 Привет!\n\nОтправь фото. Я проанализирую референс и пришлю:\n✅ готовый текст поста VK\n✅ готовый промпт для комментария\n\nФото и публикацию в VK ты размещаешь сам.")


@bot.message_handler(content_types=["photo", "document"])
def handle_photo(message):
    status = bot.reply_to(message, "⏳ Получил фото. Начинаю подробный анализ...")
    try:
        file_id = message.photo[-1].file_id if message.photo else message.document.file_id
        file_info = bot.get_file(file_id)
        original = bot.download_file(file_info.file_path)
        ratio = aspect_ratio(original)
        ref = normalize_image(original)

        analysis = analyze(ref, ratio)
        prompt = build_prompt(analysis)
        post = build_vk_post(analysis)
        comment = build_vk_comment(prompt)
        title = v(analysis, "photo_title", default="Нейрофотосессия")

        bot.edit_message_text("✅ Анализ завершён. Отправляю готовые материалы...", message.chat.id, status.message_id)
        bot.send_message(message.chat.id, f"✅ Готово!\n\n📌 {title}\n\nФото публикуешь сам вручную.")
        send_long(message.chat.id, "📝 ГОТОВЫЙ ТЕКСТ ПОСТА VK\n\n" + post)
        send_long(message.chat.id, "💬 ГОТОВЫЙ КОММЕНТАРИЙ VK\n\n" + comment)
        bot.edit_message_text("🎉 Готово!\n\n✅ Фото проанализировано\n✅ Промпт создан\n✅ Текст поста создан\n✅ Комментарий создан\n\n📌 Теперь бери фото и эти тексты и публикуй их вручную в группе VK.", message.chat.id, status.message_id)

    except Exception as e:
        print("\n========== ERROR ==========")
        print(str(e))
        traceback.print_exc()
        print("===========================\n")
        try:
            bot.edit_message_text(f"❌ Произошла ошибка:\n\n{e}", message.chat.id, status.message_id)
        except Exception:
            bot.reply_to(message, f"❌ Произошла ошибка:\n\n{e}")


if __name__ == "__main__":
    print(f"🤖 Бот запущен. TEXT_MODEL={TEXT_MODEL}")
    bot.infinity_polling(skip_pending=True)
