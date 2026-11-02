"""Safe Yandex Disk → Gemini → VK autoposter.

No publication occurs unless an administrator explicitly sends /post_now or
sets SCHEDULER_ENABLED=true. Both RUN_ON_START and the scheduler are off by
default.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
import telebot
from flask import Flask, abort, request
from google import genai
from google.genai import types
from PIL import Image

LOG = logging.getLogger("vk_autopost")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
VK_AUTHORIZE, VK_TOKEN, VK_API = "https://id.vk.ru/authorize", "https://id.vk.ru/oauth2/auth", "https://api.vk.com/method/"
YANDEX = "https://cloud-api.yandex.net/v1/disk"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
POST_LOCK = threading.Lock()


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


def required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Не задана обязательная переменная {name}")
    return value


def scope_list(name: str) -> tuple[str, ...]:
    return tuple(value for value in os.getenv(name, "").split() if value)


@dataclass(frozen=True)
class Settings:
    telegram_token: str; chat_id: int; gemini_key: str; yandex_token: str; yandex_folder: str
    client_id: str; redirect_uri: str; group_id: int; community_token: str
    oauth_scopes: tuple[str, ...]; required_scopes: tuple[str, ...]; confidential: bool; service_token: str
    api_version: str; interval_hours: float; scheduler_enabled: bool; run_on_start: bool
    data_dir: Path; port: int; text_model: str; timeout: int; max_file_bytes: int


def load_settings() -> Settings:
    return Settings(
        required("TELEGRAM_BOT_TOKEN"), int(required("TELEGRAM_CHAT_ID")), required("GEMINI_API_KEY"), required("YANDEX_TOKEN"), os.getenv("YANDEX_FOLDER", "disk:/Нейрофото"),
        os.getenv("VK_CLIENT_ID", "54775563"), required("VK_REDIRECT_URI"), int(required("VK_GROUP_ID")), os.getenv("VK_COMMUNITY_TOKEN", os.getenv("VK_ACCESS_TOKEN", "")),
        scope_list("VK_OAUTH_SCOPES"), scope_list("VK_REQUIRED_SCOPES"), env_bool("VK_APP_CONFIDENTIAL"), os.getenv("VK_SERVICE_TOKEN", ""),
        os.getenv("VK_API_VERSION", "5.199"), float(os.getenv("POST_INTERVAL_HOURS", "2")), env_bool("SCHEDULER_ENABLED"), env_bool("RUN_ON_START"),
        Path(os.getenv("DATA_DIR", "/app/data")), int(os.getenv("PORT", "8080")), os.getenv("TEXT_MODEL", "gemini-3.6-flash"), int(os.getenv("REQUEST_TIMEOUT", "90")), int(os.getenv("MAX_YANDEX_FILE_BYTES", "26214400")),
    )


def state_path(s: Settings, name: str) -> Path:
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s.data_dir / name


def load_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return fallback
    except (OSError, json.JSONDecodeError):
        LOG.warning("State %s is unreadable; using safe default", path.name)
        return fallback


def save_json_atomic(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(temp, 0o600)
    except OSError:
        pass
    os.replace(temp, path)


def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    return verifier, challenge


def normalise_scopes(value: Any) -> set[str]:
    return set(value.split()) if isinstance(value, str) else {str(x) for x in value} if isinstance(value, list) else set()


def aspect_ratio(raw: bytes) -> str:
    with Image.open(io.BytesIO(raw)) as image:
        ratio = image.width / image.height
    variants = {"1:1": 1, "4:5": .8, "3:4": .75, "2:3": .6667, "9:16": .5625, "5:4": 1.25, "4:3": 1.3333, "3:2": 1.5, "16:9": 1.7778}
    return min(variants, key=lambda candidate: abs(variants[candidate] - ratio))


def ai_bytes(raw: bytes) -> bytes:
    with Image.open(io.BytesIO(raw)) as image:
        output = io.BytesIO(); image.convert("RGB").save(output, format="JPEG", quality=95)
    return output.getvalue()


def json_text(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()[1:]
        if lines and lines[-1].strip() == "```": lines.pop()
        raw = "\n".join(lines).strip()
    return raw


def analyse(s: Settings, image: bytes) -> dict[str, Any]:
    ratio = aspect_ratio(image)
    instruction = f'''Проанализируй конкретное фото. Верни строго JSON, все описания на русском. Не выдумывай деталей; параметры камеры указывай как вероятную оценку, не EXIF-факт. Соотношение сторон {ratio}. Схема: {{"photo_title":"","style":"","hashtags":"","composition":{{"aspect_ratio":"{ratio}","shot_type":"","framing":"","camera_angle":"","subject_position":"","perspective":""}},"subject":{{"description":"","position":"","scale_in_frame":""}},"face_and_expression":{{"makeup":"","skin":""}},"hair":{{"color":"","length":"","style":"","details":""}},"outfit":{{"description":"","colors":"","materials":"","shoes":"","accessories":""}},"pose":{{"body":"","head":"","left_arm":"","right_arm":"","left_hand":"","right_hand":"","legs":"","feet":""}},"environment":{{"location":"","background":"","foreground":"","objects":""}},"lighting":{{"type":"","source":"","direction":"","hardness":"","shadows":"","rim_light":""}},"camera":{{"camera_type":"","lens":"","aperture":"","iso":"","shutter_speed":"","depth_of_field":""}},"color":{{"palette":"","grading":"","contrast":"","saturation":"","white_balance":""}}}}'''
    client = genai.Client(api_key=s.gemini_key)
    response = client.models.generate_content(model=s.text_model, contents=[types.Part.from_bytes(data=ai_bytes(image), mime_type="image/jpeg"), instruction], config=types.GenerateContentConfig(response_mime_type="application/json"))
    result = json.loads(json_text(response.text))
    if not isinstance(result, dict): raise RuntimeError("Gemini вернул JSON не в виде объекта")
    result.setdefault("composition", {})["aspect_ratio"] = ratio
    return result


def val(data: dict[str, Any], *keys: str) -> str:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict): return ""
        current = current.get(key, "")
    return str(current or "")


def build_standard_prompt(data: dict[str, Any]) -> str:
    c, f, h, o, p, e, l, z = (data.get(k, {}) for k in ("camera", "face_and_expression", "hair", "outfit", "pose", "environment", "lighting", "color"))
    q = data.get("composition", {})
    return f'''[📌](https://vk.ru/emoji/e/f09f938c.png) Промпт для генерации:

Внешность должна полностью соответствовать прикреплённому референсу: идентичные черты лица, возраст, рост, форма лица, цвет глаз, цвет и длина волос, причёска, телосложение, пропорции, макияж, выражение лица и общее визуальное впечатление.

Любые изменения внешности, стилизация под другого человека или искажение типажа недопустимы.

Стиль: {val(data, "style")}

Камера и параметры: {val(c,"camera_type")}, объектив {val(c,"lens")}, {val(c,"aperture")}, ISO {val(c,"iso")}, выдержка {val(c,"shutter_speed")}, глубина резкости — {val(c,"depth_of_field")}. {val(q,"framing")}. {val(q,"shot_type")}, кадр {val(q,"aspect_ratio")}, ракурс {val(q,"camera_angle")}. Положение объекта: {val(q,"subject_position")}. Перспектива: {val(q,"perspective")}.

Причёска, кожа и макияж: волосы {val(h,"color")}, {val(h,"length")}, укладка {val(h,"style")}; {val(h,"details")}. Кожа: {val(f,"skin")}. Макияж: {val(f,"makeup")}.

Образ: {val(o,"description")}. Цвета: {val(o,"colors")}. Материалы: {val(o,"materials")}. Обувь: {val(o,"shoes")}. Аксессуары: {val(o,"accessories")}.

Поза: корпус — {val(p,"body")}; голова — {val(p,"head")}; левая рука — {val(p,"left_arm")}; правая рука — {val(p,"right_arm")}; левая кисть — {val(p,"left_hand")}; правая кисть — {val(p,"right_hand")}; ноги — {val(p,"legs")}; стопы — {val(p,"feet")}.

Освещение: {val(l,"type")}. Источник: {val(l,"source")}. Направление: {val(l,"direction")}. Жёсткость: {val(l,"hardness")}. Тени: {val(l,"shadows")}. Контровой свет: {val(l,"rim_light")}.

Фон: локация {val(e,"location")}. {val(e,"background")}. Передний план: {val(e,"foreground")}. Предметы: {val(e,"objects")}.

Цветокор: палитра {val(z,"palette")}. Грейдинг: {val(z,"grading")}. Контраст: {val(z,"contrast")}. Насыщенность: {val(z,"saturation")}. Баланс белого: {val(z,"white_balance")}.

Сохраняй композицию, геометрию, ракурс, кадрирование, положение и масштаб объектов, позу, освещение, фон и цветовую логику референса.

Максимальный фотореализм. Высокая детализация. Естественная анатомия. Реалистичная кожа. Натуральная текстура кожи. Реалистичные волосы. Реалистичные глаза. Естественный свет. Реалистичные тени.

Негативный промпт: Без CGI. Без мультяшности. Без пластиковой кожи. Без деформаций. Без лишних пальцев. Без лишних конечностей. Без анатомических ошибок.'''.strip()


def wall_text(data: dict[str, Any]) -> str:
    return f'''✨ {val(data,"photo_title") or "Нейрофотосессия"}

ИНСТРУКЦИЯ

КАК СОЗДАТЬ ФОТО С ПОМОЩЬЮ БОТОВ 🖤🖤🖤🖤

🔹 БОТ 1 ВК — GPTron Nano Banana Pro🍌✅
1️⃣ Переходим в бот ➡️https://vk.com/write-236453790?ref=pp53aacd7d52

🔹 БОТ 2 ВК — Lexy Nano Banana Pro🍌✅
Переходим в бот ➡️https://vk.com/write-233546714?ref=84372609_add

Отправляем своё фото.

Перед отправкой вставляем нужный промт в комментариях.

ВАЖНО ПРОЧИТАТЬ

❗ Промт всегда можно и нужно менять под себя: цвет волос, глаз, одежду, позу, настроение и т.д.

👇 Забирай готовый промпт для генерации в комментариях к этому посту!

{val(data,"hashtags") or "#нейрофото #промпт #нейросеть"}'''


class Poster:
    def __init__(self, s: Settings): self.s = s
    def tokens(self) -> dict[str, Any]: return load_json(state_path(self.s, "vk_tokens.json"), {})
    def save_tokens(self, values: dict[str, Any]) -> None: save_json_atomic(state_path(self.s, "vk_tokens.json"), values)

    def login_url(self, chat_id: int) -> str:
        state, (verifier, challenge) = secrets.token_urlsafe(32), pkce_pair()
        path, flows = state_path(self.s, "vk_oauth_flows.json"), load_json(state_path(self.s, "vk_oauth_flows.json"), {})
        flows[state] = {"chat_id": chat_id, "code_verifier": verifier, "created_at": time.time()}; save_json_atomic(path, flows)
        query: dict[str, str] = {"response_type":"code", "client_id":self.s.client_id, "redirect_uri":self.s.redirect_uri, "state":state, "code_challenge":challenge, "code_challenge_method":"S256", "prompt":"consent"}
        if self.s.oauth_scopes: query["scope"] = " ".join(self.s.oauth_scopes)
        return VK_AUTHORIZE + "?" + urlencode(query)

    def callback(self, code: str, state: str, device_id: str) -> set[str]:
        path, flows = state_path(self.s, "vk_oauth_flows.json"), load_json(state_path(self.s, "vk_oauth_flows.json"), {})
        flow = flows.pop(state, None); save_json_atomic(path, flows)  # consume before external request: replay-safe
        if not flow or time.time() - float(flow.get("created_at", 0)) > 600: raise RuntimeError("OAuth state отсутствует, уже использован или истёк")
        payload = {"grant_type":"authorization_code", "code":code, "code_verifier":flow["code_verifier"], "redirect_uri":self.s.redirect_uri, "client_id":self.s.client_id, "device_id":device_id, "state":state}
        if self.s.confidential: payload["service_token"] = required("VK_SERVICE_TOKEN")
        result = requests.post(VK_TOKEN, data=payload, timeout=self.s.timeout); result.raise_for_status(); token = result.json()
        if token.get("state") and token["state"] != state: raise RuntimeError("VK ID вернул несовпадающий state")
        if not token.get("access_token") or not token.get("refresh_token"): raise RuntimeError("VK ID не выдал access_token и refresh_token")
        granted, missing = normalise_scopes(token.get("scope")), set(self.s.required_scopes) - normalise_scopes(token.get("scope"))
        if missing: raise RuntimeError("VK ID не выдал обязательные разрешения: " + ", ".join(sorted(missing)))
        token["device_id"], token["expires_at"] = device_id, int(time.time()) + int(token.get("expires_in", 0)); self.save_tokens(token)
        return granted

    def user_token(self) -> str:
        old = self.tokens()
        if not old.get("access_token"): raise RuntimeError("VK ID не подключён: выполните /vk_login")
        if int(old.get("expires_at", 0)) > int(time.time()) + 60: return str(old["access_token"])
        payload = {"grant_type":"refresh_token", "refresh_token":old.get("refresh_token", ""), "client_id":self.s.client_id, "device_id":old.get("device_id", ""), "state":secrets.token_urlsafe(32)}
        if self.s.oauth_scopes: payload["scope"] = " ".join(self.s.oauth_scopes)
        if self.s.confidential: payload["service_token"] = required("VK_SERVICE_TOKEN")
        response = requests.post(VK_TOKEN, data=payload, timeout=self.s.timeout); response.raise_for_status(); new = response.json()
        if not new.get("access_token") or not new.get("refresh_token"): raise RuntimeError("VK ID не обновил токены")
        missing = set(self.s.required_scopes) - normalise_scopes(new.get("scope"))
        if missing: raise RuntimeError("После обновления VK ID не выдал обязательные разрешения: " + ", ".join(sorted(missing)))
        new["device_id"], new["expires_at"] = old["device_id"], int(time.time()) + int(new.get("expires_in", 0)); self.save_tokens(new)
        return str(new["access_token"])

    def api(self, method: str, token: str, **params: Any) -> dict[str, Any]:
        response = requests.post(VK_API + method, data={**params, "access_token":token, "v":self.s.api_version}, timeout=self.s.timeout); response.raise_for_status(); body = response.json()
        if "error" in body: raise RuntimeError(f"VK API {method}: {body['error'].get('error_msg', 'неизвестная ошибка')}")
        return body["response"]

    def preflight(self) -> str:
        data = self.api("photos.getWallUploadServer", self.user_token(), group_id=self.s.group_id)
        if not data.get("upload_url"): raise RuntimeError("VK не вернул upload_url")
        return "VK preflight пройден: upload_url получен; фото и пост не создавались."

    def yandex_next(self) -> tuple[str, bytes]:
        done = set(load_json(state_path(self.s, "processed_files.json"), {"files":[]}).get("files", [])); headers = {"Authorization":f"OAuth {self.s.yandex_token}"}
        listing = requests.get(YANDEX + "/resources", headers=headers, params={"path":self.s.yandex_folder, "limit":100}, timeout=self.s.timeout); listing.raise_for_status()
        for item in sorted(listing.json().get("_embedded", {}).get("items", []), key=lambda x: x.get("name", "")):
            path = item.get("path", "")
            if path in done or item.get("type") != "file" or Path(item.get("name", "")).suffix.lower() not in EXTENSIONS: continue
            meta = requests.get(YANDEX + "/resources/download", headers=headers, params={"path":path}, timeout=self.s.timeout); meta.raise_for_status()
            file = requests.get(meta.json()["href"], timeout=self.s.timeout); file.raise_for_status()
            if len(file.content) > self.s.max_file_bytes: raise RuntimeError("Файл превышает MAX_YANDEX_FILE_BYTES")
            return path, file.content
        raise RuntimeError("В Яндекс Диске нет нового подходящего изображения")

    def upload(self, image: bytes) -> str:
        token = self.user_token(); server = self.api("photos.getWallUploadServer", token, group_id=self.s.group_id)
        raw = requests.post(server["upload_url"], files={"photo":("source.jpg", image, "image/jpeg")}, timeout=self.s.timeout); raw.raise_for_status(); data = raw.json()
        photo = self.api("photos.saveWallPhoto", token, group_id=self.s.group_id, server=data["server"], photo=data["photo"], hash=data["hash"])[0]
        return f"photo{photo['owner_id']}_{photo['id']}"

    def mark_done(self, source: str) -> None:
        path, data = state_path(self.s, "processed_files.json"), load_json(state_path(self.s, "processed_files.json"), {"files":[]})
        data.setdefault("files", []).append(source); data["files"] = list(dict.fromkeys(data["files"])); save_json_atomic(path, data)

    def post_one(self) -> str:
        if not self.s.community_token: raise RuntimeError("Не задан VK_COMMUNITY_TOKEN (или совместимый VK_ACCESS_TOKEN)")
        with POST_LOCK:
            pending_path, pending = state_path(self.s, "vk_pending_post.json"), load_json(state_path(self.s, "vk_pending_post.json"), {})
            if pending:
                self.api("wall.createComment", self.s.community_token, owner_id=-self.s.group_id, post_id=pending["post_id"], message=pending["comment"]); self.mark_done(pending["source_path"]); save_json_atomic(pending_path, {})
                return "Комментарий добавлен к уже созданному посту; дубликат поста не создан."
            source, image = self.yandex_next(); data = analyse(self.s, image); attachment = self.upload(image)
            post = self.api("wall.post", self.s.community_token, owner_id=-self.s.group_id, from_group=1, message=wall_text(data), attachments=attachment)
            pending = {"source_path":source, "post_id":post["post_id"], "comment":build_standard_prompt(data)}; save_json_atomic(pending_path, pending)
            self.api("wall.createComment", self.s.community_token, owner_id=-self.s.group_id, post_id=post["post_id"], message=pending["comment"]); self.mark_done(source); save_json_atomic(pending_path, {})
            return f"Опубликован один файл: {Path(source).name}"


def build_runtime() -> tuple[Settings, Poster, telebot.TeleBot, Flask]:
    s, app, poster = load_settings(), Flask(__name__), None
    poster, bot = Poster(s), telebot.TeleBot(s.telegram_token)
    def is_admin(message: Any) -> bool:
        if message.chat.id == s.chat_id: return True
        bot.reply_to(message, "Нет прав для управления этим ботом."); return False
    @app.get("/health")
    def health() -> tuple[dict[str, Any], int]: return {"status":"ok", "scheduler_enabled":s.scheduler_enabled}, 200
    @app.get("/vk/callback")
    def callback() -> tuple[str, int]:
        code, state, device_id = request.args.get("code", ""), request.args.get("state", ""), request.args.get("device_id", "")
        if not code or not state or not device_id: abort(400, "VK ID callback requires code, state and device_id")
        try: poster.callback(code, state, device_id); return "VK ID подключён. Вернитесь в Telegram и выполните /vk_status.", 200
        except Exception as error: LOG.warning("OAuth callback failed: %s", error); return "Не удалось подключить VK ID. Начните /vk_login заново.", 400
    @bot.message_handler(commands=["start"])
    def start(m: Any) -> None:
        if is_admin(m): bot.reply_to(m, "Готово. Команды: /vk_login, /vk_status, /vk_check. Публикация — только /post_now.")
    @bot.message_handler(commands=["vk_login"])
    def login(m: Any) -> None:
        if is_admin(m): bot.reply_to(m, "Откройте ссылку, войдите в VK ID и подтвердите разрешения:\n" + poster.login_url(m.chat.id))
    @bot.message_handler(commands=["vk_status"])
    def status(m: Any) -> None:
        if is_admin(m):
            token = poster.tokens(); grants = ", ".join(sorted(normalise_scopes(token.get("scope")))) or "нет"
            bot.reply_to(m, f"VK ID: {'подключён' if token.get('access_token') else 'не подключён'}\nВыданные scopes: {grants}\nТокен сообщества: {'задан' if s.community_token else 'не задан'}\nРасписание: {'включено' if s.scheduler_enabled else 'выключено'}")
    @bot.message_handler(commands=["vk_check"])
    def check(m: Any) -> None:
        if is_admin(m):
            try: bot.reply_to(m, poster.preflight())
            except Exception as error: bot.reply_to(m, f"VK preflight не пройден: {error}")
    @bot.message_handler(commands=["post_now"])
    def post(m: Any) -> None:
        if is_admin(m):
            try: bot.reply_to(m, poster.post_one())
            except Exception as error: LOG.exception("Manual post failed"); bot.reply_to(m, f"Публикация не выполнена: {error}")
    return s, poster, bot, app


def scheduled(poster: Poster, s: Settings) -> None:
    if s.run_on_start:
        try: LOG.info(poster.post_one())
        except Exception: LOG.exception("Startup posting failed")
    while True:
        time.sleep(s.interval_hours * 3600)
        try: LOG.info(poster.post_one())
        except Exception: LOG.exception("Scheduled posting failed")


def main() -> None:
    s, poster, bot, app = build_runtime()
    threading.Thread(target=lambda: bot.infinity_polling(timeout=120, long_polling_timeout=120, skip_pending=True), daemon=True).start()
    if s.scheduler_enabled: threading.Thread(target=scheduled, args=(poster, s), daemon=True).start()
    LOG.info("Service started; scheduler_enabled=%s", s.scheduler_enabled); app.run(host="0.0.0.0", port=s.port)


if __name__ == "__main__": main()

