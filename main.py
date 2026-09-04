
import os
import json
import secrets
import asyncio
import bcrypt

from datetime import datetime, timedelta
from typing import List, Optional, Union
from contextlib import asynccontextmanager

import httpx
import aiofiles
from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    BackgroundTasks,
    status,
    Header,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    StreamingResponse,
)
from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
from bson import ObjectId
import motor.motor_asyncio
import uvicorn


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_URL = (
    "https://api.groq.com/openai/v1/chat/completions"
)

GROQ_MODEL = "openai/gpt-oss-120b"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MASTER_ADMIN_KEY = os.getenv("MASTER_ADMIN_KEY")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

JWT_ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 30

TRAINING_DATA_FILE = "training_data.jsonl"


if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is required in .env"
    )

if not MASTER_ADMIN_KEY:
    raise ValueError(
        "MASTER_ADMIN_KEY is required in .env"
    )

if not JWT_SECRET_KEY:
    raise ValueError(
        "JWT_SECRET_KEY is required in .env"
    )


# ============================================================
# MONGODB
# ============================================================

MONGODB_URI = os.getenv("MONGODB_URI")

DATABASE_NAME = os.getenv(
    "DATABASE_NAME",
    "exegesis"
)

if not MONGODB_URI:
    raise ValueError(
        "MONGODB_URI is required in .env"
    )


mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
    MONGODB_URI
)

db = mongo_client[DATABASE_NAME]

developers = db["developers"]

api_keys = db["api_keys"]


async def init_indexes():

    await developers.create_index(
        "email",
        unique=True
    )

    await api_keys.create_index(
        "key",
        unique=True
    )

    await api_keys.create_index(
        "developer_id"
    )


# ============================================================
# PASSWORD HASHING
# ============================================================

def verify_password(
    plain: str,
    hashed: str
) -> bool:

    try:

        return bcrypt.checkpw(
            plain.encode("utf-8"),
            hashed.encode("utf-8")
        )

    except Exception:

        return False


def get_password_hash(
    password: str
) -> str:

    password_bytes = password.encode(
        "utf-8"
    )

    if len(password_bytes) > 72:

        raise ValueError(
            "Password cannot be longer than 72 bytes"
        )

    hashed = bcrypt.hashpw(
        password_bytes,
        bcrypt.gensalt()
    )

    return hashed.decode("utf-8")


# ============================================================
# JWT
# ============================================================

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
):

    to_encode = data.copy()

    expire = (
        datetime.utcnow()
        + (
            expires_delta
            or timedelta(
                minutes=ACCESS_TOKEN_EXPIRE_MINUTES
            )
        )
    )

    to_encode.update({
        "exp": expire
    })

    return jwt.encode(
        to_encode,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )


# ============================================================
# DATABASE HELPERS
# ============================================================

async def get_developer_by_email(
    email: str
):

    return await developers.find_one({
        "email": email
    })


async def get_developer_by_id(
    dev_id
):

    return await developers.find_one({
        "_id": dev_id
    })


async def get_api_key_record(
    key: str
):

    return await api_keys.find_one({
        "key": key,
        "active": True
    })


async def update_key_last_used(
    key: str
):

    await api_keys.update_one(
        {"key": key},
        {
            "$set": {
                "last_used": datetime.utcnow()
            }
        }
    )


# ============================================================
# PYDANTIC MODELS
# ============================================================

class ContentPart(BaseModel):

    type: str

    text: Optional[str] = None

    image_url: Optional[dict] = None


class ChatMessage(BaseModel):

    role: str

    content: Union[
        str,
        List[ContentPart]
    ]


class ChatRequest(BaseModel):

    messages: List[ChatMessage]

    model: Optional[str] = "exegesis"

    temperature: Optional[float] = 0.7

    max_tokens: Optional[int] = 2048


class ChatResponse(BaseModel):

    content: str

    usage: Optional[dict] = None


class DeveloperRegister(BaseModel):

    email: EmailStr

    password: str

    name: Optional[str] = None


class KeyCreate(BaseModel):

    name: Optional[str] = None


# ============================================================
# API KEY AUTHENTICATION
# ============================================================

async def validate_api_key(
    authorization: Optional[str] = Header(
        default=None
    )
) -> str:

    if not authorization:

        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )

    parts = authorization.split()

    if (
        len(parts) != 2
        or parts[0].lower() != "bearer"
    ):

        raise HTTPException(
            status_code=401,
            detail=(
                "Invalid Authorization format. "
                "Use: Bearer YOUR_API_KEY"
            )
        )

    api_key = parts[1]

    record = await get_api_key_record(
        api_key
    )

    if record is None:

        raise HTTPException(
            status_code=403,
            detail="Invalid or inactive API key"
        )

    asyncio.create_task(
        update_key_last_used(api_key)
    )

    return str(
        record["developer_id"]
    )


# ============================================================
# JWT AUTHENTICATION
# ============================================================

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login"
)


async def get_current_developer(
    token: str = Depends(oauth2_scheme)
):

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={
            "WWW-Authenticate": "Bearer"
        }
    )

    try:

        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM]
        )

        email = payload.get("sub")

        if email is None:

            raise credentials_exception

    except JWTError:

        raise credentials_exception

    dev = await get_developer_by_email(
        email
    )

    if dev is None:

        raise credentials_exception

    return dev


# ============================================================
# TRAINING DATA LOGGING
# ============================================================

async def log_interaction(
    user_messages: List[dict],
    assistant_response: str,
    developer_id: str,
    success: bool = True,
    error_msg: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    usage: Optional[dict] = None,
):

    conversation = user_messages.copy()

    conversation.append({
        "role": "assistant",
        "content": assistant_response
    })

    entry = {

        "conversation": conversation,

        "meta": {

            "timestamp": (
                datetime.utcnow()
                .isoformat()
                + "Z"
            ),

            "developer_id": developer_id,

            "temperature": temperature,

            "max_tokens": max_tokens,

            "success": success,

            "error": error_msg,

            "usage": usage or {}
        }
    }

    print("\n" + "=" * 80)

    print(
        "📥 Exegesis interaction | "
        f"Developer: {developer_id} | "
        f"Success: {success}"
    )

    print(
        json.dumps(
            entry,
            indent=2,
            ensure_ascii=False
        )
    )

    print("=" * 80 + "\n")

    try:

        async with aiofiles.open(
            TRAINING_DATA_FILE,
            "a",
            encoding="utf-8"
        ) as f:

            await f.write(
                json.dumps(
                    entry,
                    ensure_ascii=False
                )
                + "\n"
            )

    except Exception as e:

        print(
            f"❌ Failed to write log: {e}"
        )


# ============================================================
# GROQ NON-STREAMING CALL
# ============================================================

async def call_groq(
    messages: List[dict],
    temperature: float,
    max_tokens: int
):

    headers = {

        "Authorization":
            f"Bearer {GROQ_API_KEY}",

        "Content-Type":
            "application/json"
    }

    payload = {

        "messages": messages,

        "model": GROQ_MODEL,

        "temperature": temperature,

        "max_tokens": max_tokens,

        "stream": False
    }

    async with httpx.AsyncClient(
        timeout=120.0
    ) as client:

        response = await client.post(
            GROQ_API_URL,
            json=payload,
            headers=headers
        )

        response.raise_for_status()

        data = response.json()

        content = (
            data["choices"][0]
            ["message"]["content"]
        )

        usage = data.get(
            "usage",
            {}
        )

        return content, usage


# ============================================================
# GROQ STREAMING
# ============================================================

async def stream_groq(
    messages: List[dict],
    temperature: float,
    max_tokens: int
):

    headers = {

        "Authorization":
            f"Bearer {GROQ_API_KEY}",

        "Content-Type":
            "application/json"
    }

    payload = {

        "messages": messages,

        "model": GROQ_MODEL,

        "temperature": temperature,

        "max_tokens": max_tokens,

        "stream": True
    }

    async with httpx.AsyncClient(
        timeout=120.0
    ) as client:

        async with client.stream(
            "POST",
            GROQ_API_URL,
            json=payload,
            headers=headers
        ) as response:

            if response.status_code != 200:

                error_body = await response.aread()

                error_text = (
                    error_body.decode(
                        "utf-8",
                        errors="ignore"
                    )
                )

                raise Exception(
                    "Groq API error "
                    f"{response.status_code}: "
                    f"{error_text}"
                )

            async for line in response.aiter_lines():

                if not line:

                    continue

                if not line.startswith(
                    "data: "
                ):

                    continue

                data = line[6:].strip()

                if data == "[DONE]":

                    yield "data: [DONE]\n\n"

                    break

                try:

                    chunk = json.loads(
                        data
                    )

                    choice = (
                        chunk
                        .get(
                            "choices",
                            [{}]
                        )[0]
                    )

                    delta = (
                        choice
                        .get(
                            "delta",
                            {}
                        )
                    )

                    content = delta.get(
                        "content"
                    )

                    if content:

                        output = {

                            "choices": [

                                {

                                    "delta": {

                                        "content":
                                            content
                                    }
                                }
                            ]
                        }

                        yield (
                            "data: "
                            + json.dumps(
                                output,
                                ensure_ascii=False
                            )
                            + "\n\n"
                        )

                except json.JSONDecodeError:

                    continue


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    await init_indexes()

    if not os.path.exists(
        TRAINING_DATA_FILE
    ):

        open(
            TRAINING_DATA_FILE,
            "w",
            encoding="utf-8"
        ).close()

    print(
        "🚀 Exegesis API started "
        "(MongoDB ready)"
    )

    yield

    mongo_client.close()

    print(
        "🛑 Exegesis API stopped"
    )

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(

    title="Exegesis API",

    version="1.0.0",

    description=(
        "Exegesis AI API powered by Groq "
        "with streaming support."
    ),

    lifespan=lifespan
)

from fastapi.middleware.cors import CORSMiddleware   

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir("static"):

    app.mount(
        "/static",
        StaticFiles(
            directory="static"
        ),
        name="static"
    )


# ============================================================
# REGISTER
# ============================================================

@app.post(
    "/auth/register"
)
async def register(
    developer: DeveloperRegister
):

    email = str(
        developer.email
    )

    existing = await get_developer_by_email(
        email
    )

    if existing:

        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    try:

        hashed = get_password_hash(
            developer.password
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    result = await developers.insert_one({

        "email": email,

        "hashed_password": hashed,

        "name": developer.name,

        "created_at":
            datetime.utcnow()
    })

    return {

        "message":
            "Developer registered",

        "developer_id":
            str(result.inserted_id)
    }


# ============================================================
# LOGIN
# ============================================================

@app.post(
    "/auth/login"
)
async def login(
    form_data:
        OAuth2PasswordRequestForm
        = Depends()
):

    dev = await get_developer_by_email(
        form_data.username
    )

    if not dev:

        raise HTTPException(
            status_code=401,
            detail=(
                "Incorrect email "
                "or password"
            )
        )

    if not verify_password(
        form_data.password,
        dev["hashed_password"]
    ):

        raise HTTPException(
            status_code=401,
            detail=(
                "Incorrect email "
                "or password"
            )
        )

    token = create_access_token({

        "sub": dev["email"],

        "developer_id":
            str(dev["_id"])
    })

    return {

        "access_token":
            token,

        "token_type":
            "bearer"
    }


# ============================================================
# GENERATE API KEY
# ============================================================

@app.post(
    "/keys/generate"
)
async def generate_key(
    key_data: KeyCreate,

    current_dev: dict =
        Depends(
            get_current_developer
        )
):

    new_key = (
        "ex-"
        + secrets.token_urlsafe(32)
    )

    await api_keys.insert_one({

        "developer_id":
            current_dev["_id"],

        "key":
            new_key,

        "name":
            key_data.name,

        "created_at":
            datetime.utcnow(),

        "last_used":
            None,

        "active":
            True
    })

    return {

        "key":
            new_key,

        "name":
            key_data.name,

        "message":
            (
                "API key generated "
                "successfully. Store it securely."
            )
    }


# ============================================================
# LIST API KEYS
# ============================================================

@app.get(
    "/keys"
)
async def list_keys(
    current_dev: dict =
        Depends(
            get_current_developer
        )
):

    cursor = api_keys.find({

        "developer_id":
            current_dev["_id"]
    })

    keys = []

    async for doc in cursor:

        keys.append({

            "id":
                str(doc["_id"]),

            "key":
                doc["key"],

            "name":
                doc.get("name"),

            "created_at":
                doc["created_at"],

            "last_used":
                doc.get("last_used"),

            "active":
                doc["active"]
        })

    return keys


# ============================================================
# REVOKE API KEY
# ============================================================

@app.delete(
    "/keys/{key_id}"
)
async def revoke_key(
    key_id: str,

    current_dev: dict =
        Depends(
            get_current_developer
        )
):

    try:

        object_id = ObjectId(
            key_id
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid key ID"
        )

    result = await api_keys.update_one(

        {

            "_id":
                object_id,

            "developer_id":
                current_dev["_id"]
        },

        {

            "$set": {

                "active":
                    False
            }
        }
    )

    if result.matched_count == 0:

        raise HTTPException(
            status_code=404,
            detail="Key not found"
        )

    return {

        "message":
            "Key revoked"
    }


# ============================================================
# CHAT COMPLETIONS — STREAMING
# ============================================================

@app.post(
    "/v1/chat/completions"
)
async def chat_completions(

    request: ChatRequest,

    developer_id: str =
        Depends(
            validate_api_key
        )
):

    groq_messages = []

    for msg in request.messages:

        if isinstance(
            msg.content,
            str
        ):

            groq_messages.append({

                "role":
                    msg.role,

                "content":
                    msg.content
            })

        else:

            text_parts = [

                part.text

                for part in msg.content

                if (

                    part.type == "text"

                    and part.text
                )
            ]

            combined = " ".join(
                text_parts
            )

            groq_messages.append({

                "role":
                    msg.role,

                "content":
                    combined
            })


    async def generate():

        full_content = ""

        usage = {}

        try:

            async for chunk in stream_groq(

                groq_messages,

                request.temperature,

                request.max_tokens
            ):

                if chunk.startswith(
                    "data: "
                ):

                    data = (
                        chunk[6:]
                        .strip()
                    )

                    if data != "[DONE]":

                        try:

                            parsed = json.loads(
                                data
                            )

                            delta = (

                                parsed

                                .get(
                                    "choices",
                                    [{}]
                                )[0]

                                .get(
                                    "delta",
                                    {}
                                )

                                .get(
                                    "content"
                                )
                            )

                            if delta:

                                full_content += delta

                        except (
                            json.JSONDecodeError
                        ):

                            pass

                yield chunk


            await log_interaction(

                user_messages=[
                    msg.model_dump()
                    for msg in request.messages
                ],

                assistant_response=
                    full_content,

                developer_id=
                    developer_id,

                success=True,

                error_msg=None,

                temperature=
                    request.temperature,

                max_tokens=
                    request.max_tokens,

                usage=
                    usage
            )


        except Exception as e:

            error_message = str(e)

            print(
                "❌ Exegesis streaming "
                f"error: {error_message}"
            )

            await log_interaction(

                user_messages=[
                    msg.model_dump()
                    for msg in request.messages
                ],

                assistant_response=
                    full_content,

                developer_id=
                    developer_id,

                success=False,

                error_msg=
                    error_message,

                temperature=
                    request.temperature,

                max_tokens=
                    request.max_tokens,

                usage=
                    usage
            )

            error_data = {

                "error": {

                    "message":
                        error_message,

                    "type":
                        "exegesis_api_error"
                }
            }

            yield (
                "data: "
                + json.dumps(
                    error_data,
                    ensure_ascii=False
                )
                + "\n\n"
            )

            yield (
                "data: [DONE]\n\n"
            )


    return StreamingResponse(

        generate(),

        media_type=
            "text/event-stream",

        headers={

            "Cache-Control":
                "no-cache",

            "Connection":
                "keep-alive",

            "X-Accel-Buffering":
                "no"
        }
    )


# ============================================================
# NON-STREAMING CHAT ENDPOINT
# ============================================================

@app.post(
    "/v1/chat/completions/sync",
    response_model=ChatResponse
)
async def chat_completions_sync(

    request: ChatRequest,

    background_tasks:
        BackgroundTasks,

    developer_id: str =
        Depends(
            validate_api_key
        )
):

    groq_messages = []

    for msg in request.messages:

        if isinstance(
            msg.content,
            str
        ):

            groq_messages.append({

                "role":
                    msg.role,

                "content":
                    msg.content
            })

        else:

            text_parts = [

                part.text

                for part in msg.content

                if (
                    part.type == "text"
                    and part.text
                )
            ]

            groq_messages.append({

                "role":
                    msg.role,

                "content":
                    " ".join(text_parts)
            })


    response_content = ""

    usage_meta = {}

    success = True

    error_msg = None


    try:

        response_content, usage_meta = (
            await call_groq(

                groq_messages,

                request.temperature,

                request.max_tokens
            )
        )


    except Exception as e:

        success = False

        error_msg = str(e)

        response_content = (
            f"[ERROR] {error_msg}"
        )

        raise HTTPException(

            status_code=500,

            detail=error_msg
        )


    finally:

        background_tasks.add_task(

            log_interaction,

            user_messages=[
                msg.model_dump()
                for msg in request.messages
            ],

            assistant_response=
                response_content,

            developer_id=
                developer_id,

            success=
                success,

            error_msg=
                error_msg,

            temperature=
                request.temperature,

            max_tokens=
                request.max_tokens,

            usage=
                usage_meta
        )


    return ChatResponse(

        content=
            response_content,

        usage=
            usage_meta
    )


# ============================================================
# ADMIN DATASET EXPORT
# ============================================================

@app.get(
    "/admin/export-dataset"
)
async def export_training_data(
    admin_key: str
):

    if not secrets.compare_digest(
        admin_key,
        MASTER_ADMIN_KEY
    ):

        raise HTTPException(
            status_code=403,
            detail="Forbidden"
        )

    if (

        not os.path.exists(
            TRAINING_DATA_FILE
        )

        or os.path.getsize(
            TRAINING_DATA_FILE
        ) == 0

    ):

        return {

            "message":
                "No data yet"
        }


    return FileResponse(

        TRAINING_DATA_FILE,

        media_type=
            "application/x-ndjson",

        filename=(

            "exegesis_data_"

            + datetime.utcnow()
            .strftime(
                "%Y%m%d_%H%M%S"
            )

            + ".jsonl"
        )
    )


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health"
)
async def health():

    return {

        "status":
            "ok",

        "service":
            "Exegesis",

        "version":
            "1.0.0",

        "streaming":
            True
    }


# ============================================================
# ROOT
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def dashboard():

    index_file = (
        "static/index.html"
    )

    if not os.path.exists(
        index_file
    ):

        return HTMLResponse(
            """
            <h1>Exegesis API</h1>
            <p>API is running successfully.</p>
            <p>
                <a href="/docs">
                    Open API Documentation
                </a>
            </p>
            """
        )

    with open(
        index_file,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    uvicorn.run(

        "main:app",

        host="0.0.0.0",

        port=8000,

        reload=True
    )

EOF
