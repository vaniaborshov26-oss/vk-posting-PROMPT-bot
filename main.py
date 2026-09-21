import base64
import hashlib
import io
import json
import logging
import os
import random
import secrets
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from flask import Flask, redirect, request, render_template_string
from PIL import Image
from google import genai
from google.genai import types


# ============================================================
# НАСТРОЙКИ
# ============================================================

# ------------------------------------------------------------
# VK
# ------------------------------------------------------------

VK_API_VERSION = "5.131"

VK_GROUP_ID = os.getenv("VK_GROUP_ID")
VK_ACCESS_TOKEN = os.getenv("VK_ACCESS_TOKEN")

# VK ID OAuth 2.1 + PKCE
VK_CLIENT_ID = os.getenv("VK_CLIENT_ID")
VK_REDIRECT_URI = os.getenv(
    "VK_REDIRECT_URI",
    "https://bot-1789502779-5051-vania.bothost.tech/vk/callback"
)

# Запрашиваемые права пользовательского токена.
VK_OAUTH_SCOPE = os.getenv(
    "VK_OAUTH_SCOPE",
    "wall,photos,groups"
)

VK_TIMEOUT = 120


# ------------------------------------------------------------
# ЯНДЕКС.ДИСК
# ------------------------------------------------------------

YANDEX_TOKEN = os.getenv("YANDEX_TOKEN")

YANDEX_FOLDER = os.getenv(
    "YANDEX_FOLDER",
    "disk:/Нейрофото"
)

YANDEX_API = "https://cloud-api.yandex.net/v1/disk"


# ------------------------------------------------------------
# GEMINI
# ------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

TEXT_MODEL = "gemini-3.6-flash"

# Резервные модели Gemini для временных ошибок 503/429/5xx.
# Все модели ниже поддерживают входные изображения и структурированный JSON.
GEMINI_FALLBACK_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
]


# ------------------------------------------------------------
# TELEGRAM
# ------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TELEGRAM_API = (
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
)


# ------------------------------------------------------------
# ИНТЕРВАЛ ПУБЛИКАЦИЙ
# ------------------------------------------------------------

# По умолчанию публикация каждые 2 часа.
#
# За один цикл:
# 1 фото -> 1 анализ -> 1 пост VK
#
# После завершения цикла программа ждёт
# POST_INTERVAL_HOURS часов.
#
# Например:
# POST_INTERVAL_HOURS=2
#
# Первый цикл выполняется сразу после запуска,
# затем следующий через 2 часа.
# ------------------------------------------------------------

POST_INTERVAL_HOURS = float(
    os.getenv("POST_INTERVAL_HOURS", "2")
)

POST_INTERVAL_SECONDS = int(
    POST_INTERVAL_HOURS * 60 * 60
)


# ------------------------------------------------------------
# ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ
# ------------------------------------------------------------

TELEGRAM_MAX_LENGTH = 3900

# Повторы при временных сбоях Gemini.
# Используем увеличивающуюся задержку + небольшой jitter.
GEMINI_RETRY_DELAYS = [30, 60, 120, 300]

# Сколько попыток делать для каждой модели.
GEMINI_ATTEMPTS_PER_MODEL = len(GEMINI_RETRY_DELAYS) + 1

VK_RETRY_DELAYS = [3, 7, 15]

# Постоянное хранилище BotHost.
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
os.makedirs(DATA_DIR, exist_ok=True)

VK_TOKEN_FILE = os.path.join(
    DATA_DIR,
    "vk_tokens.json"
)

VK_PENDING_FILE = os.path.join(
    DATA_DIR,
    "vk_oauth_pending.json"
)

VK_FLASK_SECRET_FILE = os.path.join(
    DATA_DIR,
    "flask_secret.txt"
)

WEB_PORT = int(
    os.getenv("PORT", "3000")
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
# GEMINI CLIENT
# ============================================================

if GEMINI_API_KEY:

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

else:

    client = None


# ============================================================
# ПРОВЕРКА ENV
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
        "VK_CLIENT_ID": VK_CLIENT_ID,
        "VK_REDIRECT_URI": VK_REDIRECT_URI,
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

    logger.info(
        "Все необходимые переменные окружения найдены."
    )

    logger.info(
        "Папка Яндекс.Диска: %s",
        YANDEX_FOLDER
    )

    logger.info(
        "Интервал публикаций: %.2f часа",
        POST_INTERVAL_HOURS
    )

    logger.info(
        "За один цикл: 1 фотография"
    )

    logger.info(
        "Gemini модели: %s -> %s",
        TEXT_MODEL,
        ", ".join(GEMINI_FALLBACK_MODELS)
    )


# ============================================================
# VK ID OAUTH 2.1 + PKCE
# ============================================================

VK_ID_AUTHORIZE_URL = "https://id.vk.ru/authorize"
VK_ID_TOKEN_URL = "https://id.vk.ru/oauth2/auth"


def _load_or_create_flask_secret():
    if os.path.exists(VK_FLASK_SECRET_FILE):

        value = Path(
            VK_FLASK_SECRET_FILE
        ).read_text(
            encoding="utf-8"
        ).strip()

        if value:
            return value

    value = secrets.token_urlsafe(48)

    Path(
        VK_FLASK_SECRET_FILE
    ).write_text(
        value,
        encoding="utf-8"
    )

    return value


app = Flask(__name__)
app.secret_key = _load_or_create_flask_secret()


def _save_json_file(
    path: str,
    data: dict
):

    temp_path = path + ".tmp"

    Path(
        temp_path
    ).write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    os.replace(
        temp_path,
        path
    )


def _load_json_file(
    path: str
) -> dict:

    if not os.path.exists(path):
        return {}

    try:

        return json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

    except Exception:

        logger.exception(
            "Не удалось прочитать JSON: %s",
            path
        )

        return {}


def _generate_code_verifier():

    return secrets.token_urlsafe(64)


def _generate_code_challenge(
    code_verifier: str
):

    digest = hashlib.sha256(
        code_verifier.encode("ascii")
    ).digest()

    return base64.urlsafe_b64encode(
        digest
    ).decode(
        "ascii"
    ).rstrip("=")


def _save_pending_oauth(
    state: str,
    code_verifier: str
):

    _save_json_file(
        VK_PENDING_FILE,
        {
            "state": state,
            "code_verifier": code_verifier,
            "created_at": time.time()
        }
    )


def _load_pending_oauth():

    return _load_json_file(
        VK_PENDING_FILE
    )


def _clear_pending_oauth():

    try:

        if os.path.exists(
            VK_PENDING_FILE
        ):

            os.remove(
                VK_PENDING_FILE
            )

    except Exception:

        logger.exception(
            "Не удалось удалить pending OAuth."
        )


def _load_vk_tokens():

    return _load_json_file(
        VK_TOKEN_FILE
    )


def _save_vk_tokens(
    tokens: dict
):

    _save_json_file(
        VK_TOKEN_FILE,
        tokens
    )


def vk_login_url():

    # Не создаём новую OAuth-сессию при каждом обращении.
    # Это важно, чтобы ссылка из Telegram не становилась недействительной
    # через несколько секунд из-за нового state.
    pending = _load_pending_oauth()

    pending_created_at = float(
        pending.get(
            "created_at",
            0
        )
    )

    if (
        pending.get("state")
        and pending.get("code_verifier")
        and time.time() - pending_created_at < 600
    ):

        state = pending["state"]
        code_verifier = pending["code_verifier"]

    else:

        state = secrets.token_urlsafe(
            32
        )

        code_verifier = (
            _generate_code_verifier()
        )

        _save_pending_oauth(
            state,
            code_verifier
        )

    code_challenge = (
        _generate_code_challenge(
            code_verifier
        )
    )

    params = {
        "lang_id": 0,
        "scheme": "light",
        "code_challenge": code_challenge,
        "code_challenge_method": "s256",
        "client_id": str(VK_CLIENT_ID),
        "response_type": "code",
        "scope": VK_OAUTH_SCOPE,
        "state": state,
        "sdk_type": "vkid",
        "app_id": str(VK_CLIENT_ID),
        "redirect_uri": VK_REDIRECT_URI,
        "prompt": "login consent",
    }

    return (
        VK_ID_AUTHORIZE_URL
        + "?"
        + urlencode(params)
    )


def _exchange_code_for_tokens(
    code: str,
    device_id: str,
    state: str,
    code_verifier: str
):

    params = {
        "grant_type": "authorization_code",
        "redirect_uri": VK_REDIRECT_URI,
        "client_id": str(VK_CLIENT_ID),
        "code_verifier": code_verifier,
        "state": state,
        "device_id": device_id,
    }

    logger.info(
        "VK ID: обмен authorization code -> token."
    )

    response = requests.post(
        VK_ID_TOKEN_URL
        + "?"
        + urlencode(params),
        data={
            "code": code
        },
        timeout=VK_TIMEOUT
    )

    try:
        data = response.json()
    except ValueError as e:
        raise RuntimeError(
            "VK ID вернул не JSON.\n"
            f"HTTP {response.status_code}\n"
            f"{response.text[:3000]}"
        ) from e

    if not response.ok:

        raise RuntimeError(
            "VK ID OAuth ошибка.\n"
            f"HTTP {response.status_code}\n"
            f"{data}"
        )

    if "error" in data:

        raise RuntimeError(
            "VK ID OAuth ошибка:\n"
            f"{data}"
        )

    returned_state = data.get(
        "state"
    )

    if (
        returned_state
        and returned_state != state
    ):

        raise RuntimeError(
            "VK ID: state в ответе не совпадает."
        )

    if not data.get(
        "access_token"
    ):

        raise RuntimeError(
            "VK ID не вернул access_token:\n"
            f"{data}"
        )

    # По документации VK ID refresh_token выдаётся вместе
    # с access_token для дальнейшего обновления.
    if not data.get(
        "refresh_token"
    ):

        raise RuntimeError(
            "VK ID не вернул refresh_token:\n"
            f"{data}"
        )

    expires_in = int(
        data.get(
            "expires_in",
            3600
        )
    )

    tokens = {
        "access_token": data[
            "access_token"
        ],

        "refresh_token": data[
            "refresh_token"
        ],

        "device_id": device_id,

        "user_id": data.get(
            "user_id"
        ),

        "scope": data.get(
            "scope",
            ""
        ),

        "expires_at": (
            time.time()
            + max(
                60,
                expires_in - 120
            )
        )
    }

    _save_vk_tokens(
        tokens
    )

    return tokens


def refresh_vk_user_token():

    old_tokens = _load_vk_tokens()

    refresh_token = old_tokens.get(
        "refresh_token"
    )

    device_id = old_tokens.get(
        "device_id"
    )

    if not refresh_token:

        raise RuntimeError(
            "VK refresh_token отсутствует. "
            "Нужно пройти /vk/login."
        )

    if not device_id:

        raise RuntimeError(
            "VK device_id отсутствует. "
            "Нужно повторно пройти авторизацию."
        )

    refresh_state = secrets.token_urlsafe(
        32
    )

    params = {
        "grant_type": "refresh_token",
        "redirect_uri": VK_REDIRECT_URI,
        "client_id": str(VK_CLIENT_ID),
        "device_id": device_id,
        "state": refresh_state,
    }

    logger.info(
        "VK ID: обновляем access token..."
    )

    response = requests.post(
        VK_ID_TOKEN_URL
        + "?"
        + urlencode(params),
        data={
            "refresh_token": refresh_token
        },
        timeout=VK_TIMEOUT
    )

    try:
        data = response.json()
    except ValueError as e:
        raise RuntimeError(
            "VK ID refresh вернул не JSON.\n"
            f"HTTP {response.status_code}\n"
            f"{response.text[:3000]}"
        ) from e

    if not response.ok:

        raise RuntimeError(
            "VK ID refresh ошибка.\n"
            f"HTTP {response.status_code}\n"
            f"{data}"
        )

    if "error" in data:

        raise RuntimeError(
            "VK ID refresh ошибка:\n"
            f"{data}"
        )

    returned_state = data.get(
        "state"
    )

    if (
        returned_state
        and returned_state != refresh_state
    ):

        raise RuntimeError(
            "VK ID refresh: state не совпадает."
        )

    if not data.get(
        "access_token"
    ):

        raise RuntimeError(
            "VK ID refresh не вернул access_token:\n"
            f"{data}"
        )

    expires_in = int(
        data.get(
            "expires_in",
            3600
        )
    )

    new_tokens = dict(
        old_tokens
    )

    new_tokens[
        "access_token"
    ] = data[
        "access_token"
    ]

    if data.get(
        "refresh_token"
    ):

        new_tokens[
            "refresh_token"
        ] = data[
            "refresh_token"
        ]

    if data.get(
        "device_id"
    ):

        new_tokens[
            "device_id"
        ] = data[
            "device_id"
        ]

    if data.get(
        "user_id"
    ):

        new_tokens[
            "user_id"
        ] = data[
            "user_id"
        ]

    if data.get(
        "scope"
    ):

        new_tokens[
            "scope"
        ] = data[
            "scope"
        ]

    new_tokens[
        "expires_at"
    ] = (
        time.time()
        + max(
            60,
            expires_in - 120
        )
    )

    _save_vk_tokens(
        new_tokens
    )

    logger.info(
        "VK ID: access token обновлён."
    )

    return new_tokens


_vk_refresh_lock = threading.Lock()


def get_vk_user_access_token(
    force_refresh: bool = False
):

    tokens = _load_vk_tokens()

    access_token = tokens.get(
        "access_token"
    )

    expires_at = float(
        tokens.get(
            "expires_at",
            0
        )
    )

    if (
        not force_refresh
        and access_token
        and time.time() < expires_at
    ):

        return access_token

    with _vk_refresh_lock:

        tokens = _load_vk_tokens()

        access_token = tokens.get(
            "access_token"
        )

        expires_at = float(
            tokens.get(
                "expires_at",
                0
            )
        )

        if (
            not force_refresh
            and access_token
            and time.time() < expires_at
        ):

            return access_token

        refreshed = refresh_vk_user_token()

        return refreshed[
            "access_token"
        ]


@app.route(
    "/",
    methods=["GET"]
)
def home_route():

    tokens = _load_vk_tokens()

    authenticated = bool(
        tokens.get("refresh_token")
    )

    return render_template_string(
        """
        <html>
        <head>
            <meta charset="utf-8">
            <title>NEURO Photo Automation</title>
        </head>
        <body style="font-family:Arial;padding:40px;line-height:1.6">
            <h1>NEURO Photo Automation</h1>

            <p>
                VK ID:
                {% if authenticated %}
                ✅ пользователь авторизован
                {% else %}
                ❌ пользователь не авторизован
                {% endif %}
            </p>

            <p>
                <a href="/vk/login">
                    🔐 Авторизоваться через VK ID
                </a>
            </p>
        </body>
        </html>
        """,
        authenticated=authenticated
    )


@app.route(
    "/vk/login",
    methods=["GET"]
)
def vk_login_route():

    try:

        if not VK_CLIENT_ID:

            return (
                "VK_CLIENT_ID не задан.",
                500
            )

        return redirect(
            vk_login_url()
        )

    except Exception as e:

        logger.exception(
            "Ошибка /vk/login"
        )

        return (
            f"<h2>Ошибка VK ID</h2><pre>{e}</pre>",
            500
        )


@app.route(
    "/vk/callback",
    methods=["GET"]
)
def vk_callback_route():

    error = request.args.get(
        "error"
    )

    if error:

        description = request.args.get(
            "error_description",
            error
        )

        return render_template_string(
            """
            <html>
            <body style="font-family:Arial;padding:40px">
                <h2>❌ VK ID: авторизация не завершена</h2>
                <p>{{ description }}</p>
                <p><a href="/vk/login">Попробовать ещё раз</a></p>
            </body>
            </html>
            """,
            description=description
        ), 400

    code = request.args.get(
        "code"
    )

    state = request.args.get(
        "state"
    )

    device_id = request.args.get(
        "device_id"
    )

    response_type = request.args.get(
        "type"
    )

    if not code:

        return (
            "VK ID не передал code.",
            400
        )

    if not state:

        return (
            "VK ID не передал state.",
            400
        )

    if not device_id:

        return (
            "VK ID не передал device_id.",
            400
        )

    pending = _load_pending_oauth()

    expected_state = pending.get(
        "state"
    )

    code_verifier = pending.get(
        "code_verifier"
    )

    if not expected_state:

        return (
            "Сессия авторизации не найдена. "
            "Откройте /vk/login ещё раз.",
            400
        )

    if state != expected_state:

        logger.error(
            "VK ID state mismatch."
        )

        return (
            "VK ID: state mismatch.",
            400
        )

    if not code_verifier:

        return (
            "code_verifier не найден.",
            400
        )

    if response_type not in (
        None,
        "",
        "code_v2"
    ):

        logger.warning(
            "VK ID вернул неожиданный type=%s",
            response_type
        )

    try:

        tokens = _exchange_code_for_tokens(
            code=code,
            device_id=device_id,
            state=state,
            code_verifier=code_verifier
        )

        _clear_pending_oauth()

        logger.info(
            "VK ID: пользовательская авторизация завершена."
        )

        try:

            telegram_send(
                "✅ VK ID авторизация завершена!\n\n"
                f"👤 VK user ID: "
                f"{tokens.get('user_id', '—')}\n"
                f"🔐 Права: "
                f"{tokens.get('scope', '—')}\n\n"
                "✅ Access token получен\n"
                "✅ Refresh token получен\n"
                "✅ Токены сохранены на BotHost\n\n"
                "Теперь можно загружать фото в VK."
            )

        except Exception:

            logger.exception(
                "Не удалось отправить Telegram OAuth уведомление."
            )

        return render_template_string(
            """
            <html>
            <body style="font-family:Arial;padding:40px;line-height:1.6">
                <h2>✅ VK ID авторизация успешна</h2>
                <p>Пользовательский токен получен.</p>
                <p>Токены сохранены на BotHost.</p>
                <p>Теперь эту страницу можно закрыть.</p>
                <hr>
                <p><b>User ID:</b> {{ user_id }}</p>
                <p><b>Scope:</b> {{ scope }}</p>
            </body>
            </html>
            """,
            user_id=tokens.get(
                "user_id",
                "—"
            ),
            scope=tokens.get(
                "scope",
                "—"
            )
        )

    except Exception as e:

        logger.exception(
            "VK ID callback error."
        )

        return render_template_string(
            """
            <html>
            <body style="font-family:Arial;padding:40px;line-height:1.6">
                <h2>❌ Ошибка авторизации VK ID</h2>
                <pre>{{ error }}</pre>
                <p><a href="/vk/login">Попробовать ещё раз</a></p>
            </body>
            </html>
            """,
            error=str(e)
        ), 500


def start_web_server():

    logger.info(
        "VK ID web server: 0.0.0.0:%s",
        WEB_PORT
    )

    app.run(
        host="0.0.0.0",
        port=WEB_PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )


def ensure_vk_user_auth():

    tokens = _load_vk_tokens()

    if tokens.get(
        "refresh_token"
    ):

        try:

            return get_vk_user_access_token()

        except Exception as e:

            logger.warning(
                "VK user token refresh failed: %s",
                e
            )

    login_url = vk_login_url()

    try:

        telegram_send(
            "🔐 Требуется авторизация VK ID\n\n"
            "Открой ссылку в браузере:\n"
            f"{login_url}\n\n"
            "После авторизации токен будет храниться "
            "на BotHost и обновляться автоматически."
        )

    except Exception:

        logger.exception(
            "Не удалось отправить ссылку VK ID в Telegram."
        )

    raise RuntimeError(
        "Требуется VK ID авторизация. "
        f"Откройте: {login_url}"
    )


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

    if not response.ok:

        logger.error(
            "Telegram error: HTTP %s | %s",
            response.status_code,
            response.text,
        )

    response.raise_for_status()


def telegram_send_long(text: str):

    if len(text) <= TELEGRAM_MAX_LENGTH:

        telegram_send(text)
        return

    for start in range(
        0,
        len(text),
        TELEGRAM_MAX_LENGTH
    ):

        chunk = text[
            start:start + TELEGRAM_MAX_LENGTH
        ]

        telegram_send(chunk)


# ============================================================
# YANDEX DISK
# ============================================================

def yandex_headers():

    return {
        "Authorization": f"OAuth {YANDEX_TOKEN}"
    }


def get_yandex_files():

    response = requests.get(
        f"{YANDEX_API}/resources",
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

    return (
        data
        .get("_embedded", {})
        .get("items", [])
    )


def select_one_image(files):

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".gif",
    }

    for item in files:

        if item.get("type") != "file":
            continue

        name = item.get(
            "name",
            ""
        )

        extension = Path(
            name
        ).suffix.lower()

        if extension in image_extensions:

            return item

    return None


def download_yandex_file(
    path: str
) -> bytes:

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


def delete_yandex_file(
    path: str
):

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

    logger.info(
        "Файл удалён с Яндекс.Диска: %s",
        path
    )


# ============================================================
# IMAGE
# ============================================================

def get_image_mime_type(
    image_bytes: bytes
):

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        fmt = (
            image.format
            or "JPEG"
        ).upper()

        mapping = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
            "GIF": "image/gif",
        }

        return mapping.get(
            fmt,
            "image/jpeg"
        )

    except Exception:

        return "image/jpeg"


def normalize_image(
    image_bytes: bytes
) -> bytes:

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        logger.info(
            "Исходный формат изображения: %s",
            image.format
        )

        logger.info(
            "Размер изображения: %s",
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

        result = buffer.getvalue()

        logger.info(
            "JPEG подготовлен: %s байт",
            len(result)
        )

        return result

    except Exception as e:

        raise RuntimeError(
            f"Не удалось подготовить изображение: {e}"
        ) from e


# ============================================================
# JSON HELPERS
# ============================================================

def clean_json_response(
    raw_text: str
):

    if not raw_text:

        raise RuntimeError(
            "Gemini вернул пустой ответ."
        )

    text = raw_text.strip()

    if text.startswith("```"):

        lines = text.splitlines()

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

        text = "\n".join(
            lines
        ).strip()

    return text


def parse_json(
    text: str
):

    cleaned = clean_json_response(
        text
    )

    try:

        return json.loads(
            cleaned
        )

    except json.JSONDecodeError as e:

        raise RuntimeError(
            "Gemini вернул некорректный JSON.\n\n"
            + cleaned[:5000]
        ) from e


# ============================================================
# GEMINI RETRY / FALLBACK
# ============================================================

def _is_transient_gemini_error(error: Exception) -> bool:
    """Определяет временные ошибки Gemini, которые безопасно повторять."""

    code = getattr(error, "code", None)

    if code in (408, 429, 500, 502, 503, 504):
        return True

    text = str(error).upper()

    transient_markers = (
        "503",
        "UNAVAILABLE",
        "SERVICE_UNAVAILABLE",
        "INTERNAL",
        "RESOURCE_EXHAUSTED",
        "TOO_MANY_REQUESTS",
        "429",
        "408",
        "TIMEOUT",
    )

    return any(marker in text for marker in transient_markers)


def _gemini_retry_wait(seconds: int):
    """Ждёт перед повтором, добавляя небольшой случайный jitter."""

    jitter = random.uniform(0, min(10, max(1, seconds * 0.15)))
    total = seconds + jitter

    logger.info(
        "Gemini: ждём %.1f сек. перед повтором...",
        total
    )

    time.sleep(total)


# ============================================================
# GEMINI
# ПОДРОБНЫЙ АНАЛИЗ РЕФЕРЕНСА
# ============================================================

def analyze_reference(
    image_bytes: bytes
) -> dict[str, Any]:

    prompt = r"""
Ты — профессиональный visual analyst и prompt engineer
для фотореалистичной генерации изображений.

Проанализируй прикреплённый референс максимально подробно.

Главная задача:
не придумать новое изображение,
а восстановить максимально точное техническое описание
того, ЧТО УЖЕ находится в референсе.

КРИТИЧЕСКИ ВАЖНО:

Если на изображении присутствует человек,
внешность должна описываться максимально точно,
без выдумывания отсутствующих деталей.

Нельзя:
- делать человека моложе или старше;
- идеализировать лицо;
- менять типаж;
- придумывать новую причёску;
- придумывать другой цвет волос;
- менять телосложение;
- добавлять несуществующий макияж;
- придумывать аксессуары;
- придумывать одежду, которой нет;
- придумывать детали фона, которых нет на фото.

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

Если точные параметры камеры неизвестны,
укажи реалистичное предположение,
но НЕ выдавай предположение за достоверный факт.

Если в изображении присутствует текст,
обязательно укажи:
- его содержание;
- язык;
- расположение;
- визуальное оформление.

Создай название фотографии.

Создай релевантные хэштеги.

Верни ТОЛЬКО JSON.

Структура JSON:

{
  "photo_title": "",

  "composition": {
    "aspect_ratio": "",
    "shot_type": "",
    "framing": "",
    "camera_angle": "",
    "subject_position": "",
    "perspective": ""
  },

  "subject": {
    "count": 0,
    "description": "",
    "position": "",
    "scale_in_frame": ""
  },

  "face_and_expression": {
    "head_position": "",
    "gaze": "",
    "expression": "",
    "makeup": "",
    "skin": ""
  },

  "hair": {
    "color": "",
    "length": "",
    "style": "",
    "details": ""
  },

  "outfit": {
    "description": "",
    "colors": "",
    "materials": "",
    "shoes": "",
    "accessories": ""
  },

  "pose": {
    "body": "",
    "head": "",
    "left_arm": "",
    "right_arm": "",
    "left_hand": "",
    "right_hand": "",
    "legs": "",
    "feet": ""
  },

  "environment": {
    "location": "",
    "background": "",
    "foreground": "",
    "objects": ""
  },

  "lighting": {
    "type": "",
    "source": "",
    "direction": "",
    "hardness": "",
    "shadows": "",
    "rim_light": ""
  },

  "camera": {
    "camera_type": "",
    "lens": "",
    "aperture": "",
    "iso": "",
    "shutter_speed": "",
    "depth_of_field": ""
  },

  "color": {
    "palette": "",
    "grading": "",
    "contrast": "",
    "saturation": "",
    "white_balance": ""
  },

  "text_in_image": {
    "present": false,
    "language": "",
    "content": "",
    "position": "",
    "style": ""
  },

  "style": "",
  "quality": "",
  "hashtags": ""
}
"""

    mime_type = get_image_mime_type(
        image_bytes
    )

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type
    )

    last_error = None

    # Основная модель + резервные модели.
    # При временной перегрузке делаем несколько попыток,
    # затем переключаемся на следующую модель.
    models_to_try = [
        TEXT_MODEL,
        *GEMINI_FALLBACK_MODELS,
    ]

    for model_index, model_name in enumerate(models_to_try):

        for attempt in range(GEMINI_ATTEMPTS_PER_MODEL):

            try:

                logger.info(
                    "Gemini анализ: модель=%s, попытка %s/%s",
                    model_name,
                    attempt + 1,
                    GEMINI_ATTEMPTS_PER_MODEL
                )

                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        image_part,
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )

                data = parse_json(
                    response.text
                )

                logger.info(
                    "Gemini JSON анализа получен. Модель: %s",
                    model_name
                )

                return data

            except Exception as e:

                last_error = e

                transient = _is_transient_gemini_error(e)

                logger.error(
                    "Gemini ошибка анализа (%s): %s",
                    "временная" if transient else "неповторяемая",
                    e
                )

                if not transient:
                    raise RuntimeError(
                        "Не удалось проанализировать референс: "
                        f"{e}"
                    ) from e

                # Если это последняя попытка текущей модели,
                # сразу переходим к следующей модели.
                if attempt == GEMINI_ATTEMPTS_PER_MODEL - 1:

                    if model_index < len(models_to_try) - 1:

                        next_model = models_to_try[model_index + 1]

                        logger.warning(
                            "Gemini: модель %s временно недоступна. "
                            "Переключаемся на %s.",
                            model_name,
                            next_model
                        )

                    break

                wait_seconds = GEMINI_RETRY_DELAYS[attempt]
                _gemini_retry_wait(wait_seconds)

    raise RuntimeError(
        "Gemini временно недоступен на всех резервных моделях. "
        f"Последняя ошибка: {last_error}"
    ) from last_error


# ============================================================
# ФОРМИРОВАНИЕ СТАНДАРТНОГО ПРОМПТА
# ============================================================

def build_standard_prompt(
    data: dict[str, Any]
) -> str:

    composition = data.get(
        "composition",
        {}
    )

    subject = data.get(
        "subject",
        {}
    )

    face = data.get(
        "face_and_expression",
        {}
    )

    hair = data.get(
        "hair",
        {}
    )

    outfit = data.get(
        "outfit",
        {}
    )

    pose = data.get(
        "pose",
        {}
    )

    environment = data.get(
        "environment",
        {}
    )

    lighting = data.get(
        "lighting",
        {}
    )

    camera = data.get(
        "camera",
        {}
    )

    color = data.get(
        "color",
        {}
    )

    text_info = data.get(
        "text_in_image",
        {}
    )

    prompt = f"""
Внешность должна полностью соответствовать
прикреплённому референсу.

Сохрани индивидуальный визуальный образ человека:
форму лица, пропорции, глаза, нос, губы, кожу,
возрастное впечатление, волосы, причёску,
мимику и общее визуальное восприятие.

СТРОГО ЗАПРЕЩЕНО:
изменять идентичность человека,
менять черты лица,
омолаживать или состаривать человека,
изменять телосложение и пропорции,
менять цвет и длину волос,
изменять причёску,
изменять выражение лица без необходимости,
идеализировать лицо,
делать кожу неестественно гладкой,
добавлять новые черты внешности.

Не идеализируй лицо.
Не меняй типаж.
Не делай человека моложе или старше.
Не добавляй новые черты.

КЛЮЧЕВАЯ ЗАДАЧА:
максимально точно воспроизвести референс,
а не создать просто похожую сцену.

СЮЖЕТ:
{data.get("photo_title", "")}

КОМПОЗИЦИЯ:

Соотношение сторон:
{composition.get("aspect_ratio", "")}

Тип кадра:
{composition.get("shot_type", "")}

Кадрирование:
{composition.get("framing", "")}

Ракурс камеры:
{composition.get("camera_angle", "")}

Положение объекта:
{composition.get("subject_position", "")}

Перспектива:
{composition.get("perspective", "")}


ГЛАВНЫЙ ОБЪЕКТ:

Количество объектов/людей:
{subject.get("count", "")}

Описание:
{subject.get("description", "")}

Положение:
{subject.get("position", "")}

Размер в кадре:
{subject.get("scale_in_frame", "")}


ЛИЦО И ВЫРАЖЕНИЕ:

Положение головы:
{face.get("head_position", "")}

Взгляд:
{face.get("gaze", "")}

Выражение лица:
{face.get("expression", "")}

Макияж:
{face.get("makeup", "")}

Кожа:
{face.get("skin", "")}


ВОЛОСЫ:

Цвет:
{hair.get("color", "")}

Длина:
{hair.get("length", "")}

Причёска:
{hair.get("style", "")}

Детали:
{hair.get("details", "")}


ОДЕЖДА:

{outfit.get("description", "")}

Цвета:
{outfit.get("colors", "")}

Материалы и фактура:
{outfit.get("materials", "")}

Обувь:
{outfit.get("shoes", "")}

Аксессуары:
{outfit.get("accessories", "")}


ПОЗА:

Корпус:
{pose.get("body", "")}

Голова:
{pose.get("head", "")}

Левая рука:
{pose.get("left_arm", "")}

Правая рука:
{pose.get("right_arm", "")}

Левая кисть:
{pose.get("left_hand", "")}

Правая кисть:
{pose.get("right_hand", "")}

Ноги:
{pose.get("legs", "")}

Стопы:
{pose.get("feet", "")}


ОКРУЖЕНИЕ:

Локация:
{environment.get("location", "")}

Фон:
{environment.get("background", "")}

Передний план:
{environment.get("foreground", "")}

Предметы:
{environment.get("objects", "")}


ОСВЕЩЕНИЕ:

Тип света:
{lighting.get("type", "")}

Источник:
{lighting.get("source", "")}

Направление:
{lighting.get("direction", "")}

Жёсткость:
{lighting.get("hardness", "")}

Тени:
{lighting.get("shadows", "")}

Контровой свет:
{lighting.get("rim_light", "")}


КАМЕРА:

Камера:
{camera.get("camera_type", "")}

Объектив:
{camera.get("lens", "")}

Диафрагма:
{camera.get("aperture", "")}

ISO:
{camera.get("iso", "")}

Выдержка:
{camera.get("shutter_speed", "")}

Глубина резкости:
{camera.get("depth_of_field", "")}


ЦВЕТОКОРРЕКЦИЯ:

Палитра:
{color.get("palette", "")}

Грейдинг:
{color.get("grading", "")}

Контраст:
{color.get("contrast", "")}

Насыщенность:
{color.get("saturation", "")}

Баланс белого:
{color.get("white_balance", "")}


ТЕКСТ НА ИЗОБРАЖЕНИИ:

Наличие:
{text_info.get("present", False)}

Язык:
{text_info.get("language", "")}

Содержание:
{text_info.get("content", "")}

Положение:
{text_info.get("position", "")}

Стиль текста:
{text_info.get("style", "")}


СТИЛЬ:

{data.get("style", "")}


КАЧЕСТВО:

{data.get("quality", "")}


СОХРАНИ:

оригинальную композицию,
геометрию,
ракурс,
кадрирование,
расположение объектов,
позу,
положение головы,
руки и пальцы,
масштаб объекта в кадре,
освещение,
фон,
цветовую логику,
атмосферу
и визуальный характер референса.

Не добавляй элементы,
которых нет в референсе.

Не удаляй важные элементы
референса.

Максимальный фотореализм.
Естественная анатомия.
Реалистичная кожа.
Естественная текстура кожи.
Высокая детализация.
Натуральные волосы.
Реалистичные глаза.
Реалистичный свет.
Реалистичные тени.

Без CGI.
Без мультяшности.
Без пластиковой кожи.
Без деформаций.
Без лишних пальцев.
Без лишних конечностей.
Без анатомических ошибок.
"""

    return prompt.strip()


# ============================================================
# VK HELPERS
# ============================================================

def safe_json_response(
    response,
    source_name: str
):

    try:

        return response.json()

    except ValueError:

        raise RuntimeError(
            f"{source_name} вернул не JSON.\n"
            f"HTTP: {response.status_code}\n"
            f"Ответ: {response.text[:3000]}"
        )


def vk_call(
    method_name: str,
    params=None,
    data=None
):

    clean_method = (
        method_name
        .strip()
        .split("/")[-1]
        .replace(".json", "")
    )

    url = (
        "https://api.vk.com/method/"
        + clean_method
    )

    last_error = None

    for attempt in range(3):

        try:

            if data is not None:

                response = requests.post(
                    url,
                    data=data,
                    timeout=VK_TIMEOUT
                )

            else:

                response = requests.get(
                    url,
                    params=params or {},
                    timeout=VK_TIMEOUT
                )

            result = safe_json_response(
                response,
                f"VK {clean_method}"
            )

            if "error" in result:

                logger.error(
                    "VK %s error: %s",
                    clean_method,
                    result["error"]
                )

            return result

        except (
            requests.exceptions.Timeout,
            requests.exceptions.RequestException
        ) as e:

            last_error = e

            logger.error(
                "VK HTTP ошибка %s, попытка %s/3: %s",
                clean_method,
                attempt + 1,
                e
            )

            if attempt < 2:

                time.sleep(
                    VK_RETRY_DELAYS[
                        attempt
                    ]
                )

    raise RuntimeError(
        f"Ошибка HTTP VK "
        f"{clean_method}: {last_error}"
    )


# ============================================================
# VK POST
# ============================================================

def post_to_vk(
    image_bytes: bytes,
    wall_text: str,
    comment_text: str
) -> str:

    group_id = int(
        str(VK_GROUP_ID)
        .replace("-", "")
    )

    # --------------------------------------------------------
    # 1. Получаем upload server
    # --------------------------------------------------------

    logger.info(
        "VK: получаем upload server..."
    )

    user_token = get_vk_user_access_token()

    server_res = vk_call(
        "photos.getWallUploadServer",
        params={
            "group_id": group_id,
            "access_token": user_token.strip(),
            "v": VK_API_VERSION
        }
    )

    if "error" in server_res:

        error_code = server_res["error"].get(
            "error_code"
        )

        # Если токен внезапно стал недействительным,
        # пробуем обновить его через refresh_token один раз.
        if error_code in (5, 7, 15):

            logger.warning(
                "VK user token rejected (error %s). "
                "Пробуем обновить token.",
                error_code
            )

            user_token = get_vk_user_access_token(
                force_refresh=True
            )

            server_res = vk_call(
                "photos.getWallUploadServer",
                params={
                    "group_id": group_id,
                    "access_token": user_token.strip(),
                    "v": VK_API_VERSION
                }
            )

        if "error" in server_res:

            raise RuntimeError(
                "VK getWallUploadServer: "
                f"{server_res['error']}"
            )

    if (
        "response" not in server_res
        or "upload_url"
        not in server_res["response"]
    ):

        raise RuntimeError(
            "VK не вернул upload_url:\n"
            f"{server_res}"
        )

    upload_url = (
        server_res["response"]["upload_url"]
    )

    # --------------------------------------------------------
    # 2. Загружаем фотографию
    # --------------------------------------------------------

    logger.info(
        "VK: загружаем фотографию..."
    )

    files = {
        "photo": (
            "photo.jpg",
            image_bytes,
            "image/jpeg"
        )
    }

    upload_response = requests.post(
        upload_url,
        files=files,
        timeout=VK_TIMEOUT
    )

    upload_res = safe_json_response(
        upload_response,
        "VK upload server"
    )

    logger.info(
        "[VK] Upload response: %s",
        upload_res
    )

    if "error" in upload_res:

        raise RuntimeError(
            f"VK upload error: "
            f"{upload_res['error']}"
        )

    server = upload_res.get(
        "server"
    )

    photo = upload_res.get(
        "photo"
    )

    vk_hash = upload_res.get(
        "hash"
    )

    if not server:

        raise RuntimeError(
            "VK upload не вернул server:\n"
            f"{upload_res}"
        )

    if not vk_hash:

        raise RuntimeError(
            "VK upload не вернул hash:\n"
            f"{upload_res}"
        )

    if (
        photo is None
        or str(photo).strip() == ""
        or photo == "[]"
        or photo == "undefined"
        or str(photo).lower() == "null"
    ):

        raise RuntimeError(
            "VK upload не вернул корректный photo.\n"
            f"Ответ:\n{upload_res}"
        )

    # --------------------------------------------------------
    # 3. Сохраняем фотографию VK
    # --------------------------------------------------------

    logger.info(
        "VK: сохраняем фотографию..."
    )

    save_res = vk_call(
        "photos.saveWallPhoto",
        data={
            "group_id": group_id,
            "server": server,
            "photo": photo,
            "hash": vk_hash,
            "access_token": user_token.strip(),
            "v": VK_API_VERSION
        }
    )

    logger.info(
        "[VK] saveWallPhoto: %s",
        save_res
    )

    if "error" in save_res:

        raise RuntimeError(
            "VK saveWallPhoto: "
            f"{save_res['error']}"
        )

    try:

        photo_info = (
            save_res["response"][0]
        )

        owner_id = photo_info[
            "owner_id"
        ]

        photo_id = photo_info[
            "id"
        ]

    except Exception as e:

        raise RuntimeError(
            "VK не вернул данные "
            "сохранённой фотографии:\n"
            f"{save_res}"
        ) from e

    # --------------------------------------------------------
    # 4. Создаём пост
    # --------------------------------------------------------

    logger.info(
        "VK: создаём пост..."
    )

    post_res = vk_call(
        "wall.post",
        data={
            "owner_id": -group_id,
            "from_group": 1,
            "message": wall_text,
            "attachments": (
                f"photo{owner_id}_{photo_id}"
            ),
            "access_token": VK_ACCESS_TOKEN.strip(),
            "v": VK_API_VERSION
        }
    )

    logger.info(
        "[VK] wall.post: %s",
        post_res
    )

    if "error" in post_res:

        raise RuntimeError(
            f"VK wall.post: "
            f"{post_res['error']}"
        )

    post_id = (
        post_res
        ["response"]
        ["post_id"]
    )

    # --------------------------------------------------------
    # 5. Публикуем промпт в комментарии
    # --------------------------------------------------------

    logger.info(
        "VK: публикуем промпт в комментарии..."
    )

    comment_res = vk_call(
        "wall.createComment",
        data={
            "owner_id": -group_id,
            "post_id": post_id,
            "from_group": group_id,
            "message": comment_text,
            "access_token": VK_ACCESS_TOKEN.strip(),
            "v": VK_API_VERSION
        }
    )

    if "error" in comment_res:

        logger.warning(
            "VK комментарий с from_group=%s не прошёл.",
            group_id
        )

        # Повторяем так же, как в старом рабочем скрипте.
        comment_res = vk_call(
            "wall.createComment",
            data={
                "owner_id": -group_id,
                "post_id": post_id,
                "from_group": 1,
                "message": comment_text,
                "access_token": VK_ACCESS_TOKEN.strip(),
                "v": VK_API_VERSION
            }
        )

    if "error" in comment_res:

        raise RuntimeError(
            "Пост VK создан, "
            "но промпт не удалось разместить "
            "в комментарии:\n"
            f"{comment_res['error']}"
        )

    logger.info(
        "VK: комментарий с промптом опубликован."
    )

    post_link = (
        f"https://vk.com/wall-"
        f"{group_id}_{post_id}"
    )

    logger.info(
        "VK POST LINK: %s",
        post_link
    )

    return post_link


# ============================================================
# ТЕКСТ ПОСТА VK
# ============================================================

def build_wall_post_text(
    data: dict[str, Any]
) -> str:

    hashtags = data.get(
        "hashtags",
        "#нейрофото #промпт #нейросеть"
    )

    return f"""⚠️ Берешь промпт — обязательно ставь лайк ❤️ на пост и делись результатом вашей генерации в комментарии!

КАК СОЗДАТЬ ФОТО С ПОМОЩЬЮ БОТОВ 🖤

🔹 БОТ 1 ВК — GPTron Nano Banana Pro 🍌✅
1️⃣ Переходим в бот:
https://vk.com/write-236453790?ref=pp53aacd7d52

🔹 БОТ 2 ВК — Lexy Nano Banana Pro 🍌✅
Переходим в бот:
https://vk.com/write-233546714?ref=84372609_add

Отправляем своё фото.
Выбираем модель генерации NANA BANANA PRO.
Перед отправкой вставляем нужный промт в комментариях.

❗️ Промт всегда можно и нужно менять под себя:
цвет волос, глаз, одежду, позу, настроение и т.д.

👇 Забирай готовый промпт для генерации в комментариях к этому посту!

{hashtags}"""


# ============================================================
# ОБРАБОТКА ОДНОЙ ФОТОГРАФИИ
# ============================================================

def process_one_photo():

    logger.info("")
    logger.info("==========================================")
    logger.info("НАЧАЛО НОВОГО ЦИКЛА")
    logger.info("==========================================")

    logger.info(
        "Папка: %s",
        YANDEX_FOLDER
    )

    # --------------------------------------------------------
    # 1. Получаем список файлов
    # --------------------------------------------------------

    files = get_yandex_files()

    logger.info(
        "Всего объектов в папке: %s",
        len(files)
    )

    image = select_one_image(
        files
    )

    # --------------------------------------------------------
    # Если фотографии нет
    # --------------------------------------------------------

    if not image:

        logger.info(
            "Фотографий для публикации нет."
        )

        telegram_send(
            "ℹ️ Проверка завершена\n\n"
            f"📁 Папка: {YANDEX_FOLDER}\n"
            "📷 Новых фотографий нет.\n\n"
            f"⏱ Следующая проверка через "
            f"{POST_INTERVAL_HOURS:g} часа."
        )

        return False

    # --------------------------------------------------------
    # 2. Выбираем РОВНО одну фотографию
    # --------------------------------------------------------

    file_name = image[
        "name"
    ]

    file_path = image[
        "path"
    ]

    logger.info(
        "Выбрано фото: %s",
        file_name
    )

    logger.info(
        "За этот цикл будет обработано "
        "ровно 1 фото."
    )

    # До запуска Gemini проверяем, что VK ID user token готов.
    # Если авторизации нет, фото остаётся на Яндекс.Диске.
    try:

        ensure_vk_user_auth()

    except Exception as e:

        logger.warning(
            "VK ID user auth не готов: %s",
            e
        )

        telegram_send(
            "⏸ Публикация отложена\n\n"
            f"📷 Файл: {file_name}\n"
            "🔐 Требуется авторизация VK ID.\n\n"
            "Фото НЕ удалено с Яндекс.Диска.\n\n"
            "🔗 Открой ссылку для авторизации:\n"
            f"{vk_login_url()}"
        )

        return False

    telegram_send(
        "🚀 Начинаю обработку фотографии\n\n"

        f"📷 Файл:\n"
        f"{file_name}\n\n"

        f"📁 Папка:\n"
        f"{YANDEX_FOLDER}\n\n"

        "🔎 Шаг 1/4 — скачивание и анализ..."
    )

    # --------------------------------------------------------
    # 3. Скачиваем фото
    # --------------------------------------------------------

    original_bytes = (
        download_yandex_file(
            file_path
        )
    )

    # --------------------------------------------------------
    # 4. Подготавливаем изображение
    # --------------------------------------------------------

    image_bytes = normalize_image(
        original_bytes
    )

    # --------------------------------------------------------
    # 5. Gemini анализ
    # --------------------------------------------------------

    telegram_send(
        "🔎 Шаг 2/4 — Gemini анализирует "
        "референс..."
    )

    analysis_json = (
        analyze_reference(
            image_bytes
        )
    )

    # --------------------------------------------------------
    # 6. Создаём стандартный промпт
    # --------------------------------------------------------

    current_prompt = (
        build_standard_prompt(
            analysis_json
        )
    )

    logger.info(
        "Стандартный промпт сформирован."
    )

    # --------------------------------------------------------
    # 7. Формируем пост VK
    # --------------------------------------------------------

    wall_text = (
        build_wall_post_text(
            analysis_json
        )
    )

    # --------------------------------------------------------
    # 8. Комментарий = ГОТОВЫЙ ПРОМПТ
    # --------------------------------------------------------

    comment_text = (
        "📌 ПРОМПТ ДЛЯ ГЕНЕРАЦИИ\n\n"
        + current_prompt
    )

    # --------------------------------------------------------
    # Telegram — промежуточный статус
    # --------------------------------------------------------

    telegram_send(
        "✅ Gemini-анализ завершён\n"
        "✅ Промпт сформирован\n\n"
        "🚀 Шаг 3/4 — публикую фото в VK..."
    )

    # --------------------------------------------------------
    # 9. Публикуем в VK
    # --------------------------------------------------------

    post_link = post_to_vk(
        image_bytes=image_bytes,
        wall_text=wall_text,
        comment_text=comment_text
    )

    # --------------------------------------------------------
    # 10. И ТОЛЬКО ПОСЛЕ УСПЕШНОГО VK
    #     удаляем исходник с Яндекс.Диска
    # --------------------------------------------------------

    telegram_send(
        "✅ VK публикация завершена\n\n"
        f"🔗 {post_link}\n\n"
        "🗑 Шаг 4/4 — удаляю исходник "
        "с Яндекс.Диска..."
    )

    delete_yandex_file(
        file_path
    )

    # --------------------------------------------------------
    # 11. Финальный отчёт
    # --------------------------------------------------------

    hashtags = analysis_json.get(
        "hashtags",
        ""
    )

    title = analysis_json.get(
        "photo_title",
        "Нейрофотосессия"
    )

    telegram_send_long(
        "🎉 ГОТОВО!\n\n"

        f"📷 Обработано:\n"
        f"{file_name}\n\n"

        f"✨ Название:\n"
        f"{title}\n\n"

        "✅ Gemini-анализ создан\n"
        "✅ Промпт создан\n"
        "✅ Фото опубликовано в VK\n"
        "✅ Промпт опубликован "
        "в комментарии VK\n"
        "✅ Ссылка на пост получена\n"
        "✅ Исходник удалён "
        "с Яндекс.Диска\n\n"

        f"🔗 ПОСТ VK:\n"
        f"{post_link}\n\n"

        f"🏷 Хэштеги:\n"
        f"{hashtags}"
    )

    logger.info(
        "Цикл успешно завершён."
    )

    logger.info(
        "Пост VK: %s",
        post_link
    )

    return True


# ============================================================
# ОСНОВНОЙ ЦИКЛ
# ============================================================

def main():

    check_env()

    # --------------------------------------------------------
    # WEB SERVER VK ID
    # --------------------------------------------------------

    web_thread = threading.Thread(
        target=start_web_server,
        name="vk-web-server",
        daemon=True
    )

    web_thread.start()

    # --------------------------------------------------------
    # Стартовое сообщение
    # --------------------------------------------------------

    telegram_send(
        "🚀 Автоматизация запущена\n\n"

        f"📁 Папка:\n"
        f"{YANDEX_FOLDER}\n\n"

        f"⏱ Интервал публикаций:\n"
        f"каждые {POST_INTERVAL_HOURS:g} часа\n\n"

        "📷 За один цикл:\n"
        "ровно 1 фотография\n\n"

        "🤖 Gemini:\n"
        "включён\n\n"

        "🔵 VK:\n"
        "включён\n\n"

        "📌 Первый цикл запускается сейчас."
    )

    # Если user authorization ещё отсутствует,
    # отправляем ссылку сразу, но бот не останавливаем.
    try:

        ensure_vk_user_auth()

    except Exception as e:

        logger.warning(
            "VK ID user authorization ещё не выполнена: %s",
            e
        )

    # --------------------------------------------------------
    # БЕСКОНЕЧНЫЙ ЦИКЛ
    # --------------------------------------------------------

    while True:

        cycle_start = time.time()

        try:

            process_one_photo()

        except Exception as e:

            logger.exception(
                "ОШИБКА ЦИКЛА"
            )

            error_text = (
                "❌ ОШИБКА АВТОМАТИЗАЦИИ\n\n"
                f"{type(e).__name__}:\n"
                f"{e}\n\n"

                "⚠️ Исходный файл "
                "не удалён, если публикация "
                "не завершилась успешно."
            )

            try:

                telegram_send(
                    error_text
                )

            except Exception:

                logger.exception(
                    "Не удалось отправить ошибку Telegram."
                )

        # ----------------------------------------------------
        # Вычисляем время до следующего запуска
        # ----------------------------------------------------

        elapsed = (
            time.time()
            - cycle_start
        )

        sleep_seconds = max(
            0,
            POST_INTERVAL_SECONDS
        )

        logger.info(
            "Цикл занял: %.1f секунд.",
            elapsed
        )

        logger.info(
            "Следующий запуск через %.2f часа.",
            sleep_seconds / 3600
        )

        try:

            telegram_send(
                "⏳ Цикл завершён.\n\n"
                f"Следующая проверка через "
                f"{POST_INTERVAL_HOURS:g} часа."
            )

        except Exception:

            logger.exception(
                "Не удалось отправить "
                "сообщение о следующем запуске."
            )

        # ----------------------------------------------------
        # ВАЖНО:
        #
        # Отсчёт 2 часов начинается ПОСЛЕ завершения
        # текущего цикла.
        #
        # То есть:
        #
        # публикация
        #     ↓
        # ожидание 2 часа
        #     ↓
        # следующая публикация
        # ----------------------------------------------------

        time.sleep(
            sleep_seconds
        )


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        logger.info(
            "Бот остановлен вручную."
        )

    except Exception as e:

        logger.exception(
            "Критическая ошибка запуска: %s",
            e
        )

        try:

            telegram_send(
                "💥 КРИТИЧЕСКАЯ ОШИБКА\n\n"
                f"{type(e).__name__}:\n"
                f"{e}"
            )

        except Exception:

            pass
