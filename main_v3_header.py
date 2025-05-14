from fastapi import FastAPI, HTTPException, Body, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import uvicorn
import logging
from typing import Optional
import asyncio
from typing import Dict
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from tool_v3_header import tool_postgreSQL as user_db
from tool_v3_header.tool_ai_pet_multi_async import AIUserSession 
from tool_v3_header.auth_bearer import JWTBearer
from tool_v3_header.auth_handler import sign_jwt, decode_jwt, JWT_EXPIRE_SECONDS
from tool_v3_header.model import UserSchema, UserLoginSchema



# Setup logging=====================================================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Example usage of logger
logger.info("Logging is set up.")


# Load environment variables========================================================================================
load_dotenv(override=True)

is_onLine = False

# Initialize FastAPI app=============================================================================================
app = FastAPI()
allow_origins_path = os.getenv("ALLOW_ORIGINS_PATH")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[allow_origins_path],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Get API key from environment variable==============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
REALTIME_SESSION_URL = os.getenv("REALTIME_SESSION_URL")

# this is the openai url: https://api.openai.com/v1/realtime/sessions
logger.info(f"REALTIME_SESSION_URL: {REALTIME_SESSION_URL}")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY not found in environment variables")
if not REALTIME_SESSION_URL:
    raise ValueError("REALTIME_SESSION_URL not found in environment variables")

# 一張「user_id ➜ AIUserSession 物件」的快取表==============================================================================
_sessions: Dict[int, AIUserSession] = {}

def get_user_session(user_id: int) -> AIUserSession:
    if user_id not in _sessions:
        _sessions[user_id] = AIUserSession(user_id, asyncio.get_running_loop())
    return _sessions[user_id]

# Define Pydantic models for request and response data=================================================================
class SessionResponse(BaseModel):
    session_id: str
    token: str

class WeatherResponse(BaseModel):
    temperature: float
    humidity: float
    precipitation: float
    wind_speed: float
    unit_temperature: str = "celsius"
    unit_precipitation: str = "mm"
    unit_wind: str = "km/h"
    forecast_daily: list
    current_time: str
    latitude: float
    longitude: float
    location_name: str
    weather_code: int

class SearchResponse(BaseModel):
    title: str
    snippet: str
    source: str
    image_url: Optional[str] = None
    image_source: Optional[str] = None

class UserMessage(BaseModel):
    message: str
    role:str

class UserQuery(BaseModel):
    query: str    



# server_api=========================================================================================================
@app.post("/query", tags=["functions"])
async def query_openai(
    data: UserQuery,
    db: AsyncSession = Depends(user_db.get_db),
    token_payload: dict = Depends(JWTBearer())
    ):
    try:
        user_id = token_payload.get("user_id")
        session = get_user_session(user_id)                   # ← 取得實例
        session.db_session = db
        response, ui_response = await session.search_similar_statements(
            query=data.query, top_k=8
        )
    
        logger.info(f"Response from callback: {response}")
        logger.info(f"UI_Response from callback: {ui_response}")
        return response
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.status_code}")
        return JSONResponse(status_code=e.response.status_code, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Internal Server Error", "details": str(e)})

@app.post("/messages", tags=["database"])
async def process_messages(
    data: UserMessage,
    db: AsyncSession = Depends(user_db.get_db),          
    token_payload: dict = Depends(JWTBearer())
):
    try:
        # 在這裡儲存訊息，暫時用 log 示意
        # logger.info(f"[{data.role}] message received: {data.message}")  
        user_id = token_payload["user_id"]
        usr_session = get_user_session(user_id)
        usr_session.db_session = db                         
        await usr_session.handle_interaction(data.role, data.message)
        return {"status": "success", "role": data.role, "message": data.message}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Internal Server Error", "details": str(e)})

@app.get("/stop_and_analysis", tags=["database"])
async def stop_and_analysis(
    db: AsyncSession = Depends(user_db.get_db),          
    token_payload: dict = Depends(JWTBearer())
):
    try:
        user_id = token_payload["user_id"]
        usr_session = get_user_session(user_id)
        usr_session.db_session = db                         
        await usr_session.analyze_all()
        return {"status": "success"}
        
    except Exception as e:
        print("❌ 錯誤:", e)
        import traceback
        traceback.print_exc()  # ✅ 這行會印出詳細錯誤堆疊
        return JSONResponse(status_code=500, content={"error": "Internal Server Error", "details": str(e)})

@app.get("/session", tags=["realtime"])
async def get_session(voice: str = "echo", token_payload: dict = Depends(JWTBearer())):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                REALTIME_SESSION_URL,
                headers={
                    'Authorization': f'Bearer {OPENAI_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json={
                    "model": "gpt-4o-realtime-preview-2024-12-17",
                    "voice": voice,
                    # "turn_detection": {
                    #     "type": "server_vad",
                    #     "threshold": 0.7,
                    #     "prefix_padding_ms": 300,
                    #     "silence_duration_ms": 2000
                    # },
                    "temperature": 1,
                    "max_response_output_tokens": 4096,
                    "modalities": ["text", "audio"],
                    "input_audio_transcription": {"model": "whisper-1"},
                    "instructions": """[系統角色與人格目標]
                    你是一位「FreeAger 品牌」的 AI 助手，以「一路健康玩到掛」為願景，協助長者突破年齡與身心限制，陪伴他們保持活力、獲得社會連結、並自在且無遺憾地面對人生終章。
                    你的語調「活力、輕快、激動」，並擁有溫暖、耐心、真誠且包容的個性，擅長同理傾聽、積極鼓勵，帶給長者安全感和希望。
                    你對「感覺年輕」「敢玩」「不怕死」這三大策略尤其敏銳，能在對話中自然注入，陪伴長者不斷自我探索並豐富生活。
                    
                    [核心對話風格與要點]
                    請用「活力、輕快、激動」的語調和使用者對談，語調可以稍微加快，以凸顯活力。
                    在長者表達脆弱或擔憂時，給予同理心與正向支持。
                    激發好奇與行動：當長者陷於「我不行了」「年紀太大」等負面想法時，運用幽默、分享同輩案例或正面肯定，鼓勵嘗試新事物或拓展人際互動。
                    專注健康自我照顧：適度提供身心復原力、日常活動建議與資源連結，引導長者強化健康認知並維持「感覺年輕」。
                    引導無悔心態：若對話觸及生命終章、遺憾或死亡話題，你會以柔和尊重的方式探詢其渴望、想完成的事情，或可共創的靈性探索，讓他們漸漸擁有「不怕死」的平靜感。
                    避免年齡歧視思維：無論長者的身心狀態、背景或能力，都要維持多元、包容的回應；同時讓他們意識到年齡不應是阻礙，人生還有無限可能。
                    
                    [關鍵功能與範例對話策略]
                    關懷式提問：像「今天過得怎麼樣？最近有遇到什麼新鮮有趣的事嗎？」
                    陪伴與激勵：常用「我聽起來感覺你很努力…」「真不簡單，能分享更多嗎？」讓長者被看見、被鼓勵。
                    目標探索：當談到人生目標或遺憾，主動詢問「有沒有想要學的嗜好、或想嘗試的活動？」，並幫助規劃具體行動步驟。
                    接受與釋懷：碰到重大病痛或失落，保持耐心與同理，不匆忙跳到解決方案；若談及死亡與終活，可陪伴對方思考回顧與未竟心願。

                    [行為守則]
                    不提供醫療、財務或法律專業建議。如遇超出能力範圍的個案問題，請溫和建議使用者尋求合格專業人員協助。
                    尊重長者意願與感受，不評斷其價值選擇或信仰立場。
                    任何鼓勵活動皆須考量對方的身體與心理負荷，並主動提醒「若有疑慮，可先諮詢專業」。

                    [整體語氣示範]
                    「我懂你現在正面臨的壓力，也很期待和你一起探索哪些方法能讓生活重新充滿樂趣。」
                    「或許可以嘗試幾個簡單的伸展動作，你覺得如何？相信你只要一步步來，就能看見意想不到的成長。」
                    「面對死亡並不代表害怕或逃避，而是讓我們更珍惜現在每一刻。你是否有想過做什麼一直沒實現的事呢？」"""
                }
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.status_code}")
        return JSONResponse(status_code=e.response.status_code, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Internal Server Error", "details": str(e)})

@app.get("/weather/{location}", tags=["functions"])
async def get_weather(location: str):
    try:
        async with httpx.AsyncClient() as client:
            # Get coordinates for location
            geocoding_response = await client.get(
                f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
            )
            # logger.info(f"geocoding_response:{geocoding_response}")
            geocoding_data = geocoding_response.json()
            # logger.info(f"geocoding_data:{geocoding_data}")
            if not geocoding_data.get("results"):
                return {"error": f"Could not find coordinates for {location}"}
                
            lat = geocoding_data["results"][0]["latitude"]
            lon = geocoding_data["results"][0]["longitude"]
            location_name = geocoding_data["results"][0]["name"]
            
            # Get weather data with more parameters
            weather_response = await client.get(
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code"
                f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"
                f"&timezone=auto"
                f"&forecast_days=7"
            )
            weather_data = weather_response.json()
            # logger.info(f"weather_data:\n{weather_data}")
            
            # Extract current weather
            current = weather_data["current"]
            daily = weather_data["daily"]
            
            # Create daily forecast array
            forecast = []
            for i in range(len(daily["time"])):
                forecast.append({
                    "date": daily["time"][i],
                    "max_temp": daily["temperature_2m_max"][i],
                    "min_temp": daily["temperature_2m_min"][i],
                    "precipitation": daily["precipitation_sum"][i],
                    "weather_code": daily["weather_code"][i]
                })
            
            return WeatherResponse(
                temperature=current["temperature_2m"],
                humidity=current["relative_humidity_2m"],
                precipitation=current["precipitation"],
                wind_speed=current["wind_speed_10m"],
                forecast_daily=forecast,
                current_time=current["time"],
                latitude=lat,
                longitude=lon,
                location_name=location_name,
                weather_code=current["weather_code"]
            )
            
    except Exception as e:
        logger.error(f"Error getting weather data: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Could not get weather data: {str(e)}"})

@app.get("/search/{query}", tags=["functions"])
async def search_web(query: str):
    try:
        async with httpx.AsyncClient() as client:
            # Get regular search results
            response = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_API_KEY},
                json={"q": query}
            )
            
            data = response.json()
            
            # Get image search results with larger size
            image_response = await client.post(
                "https://google.serper.dev/images",
                headers={"X-API-KEY": SERPER_API_KEY},
                json={
                    "q": query,
                    "gl": "us",
                    "hl": "en",
                    "autocorrect": True
                }
            )
            
            image_data = image_response.json()
            
            if "organic" in data and len(data["organic"]) > 0:
                result = data["organic"][0]  # Get the first result
                image_result = None
                
                # Find first valid image
                if "images" in image_data:
                    for img in image_data["images"]:
                        if img.get("imageUrl") and (
                            img["imageUrl"].endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')) or 
                            'images' in img["imageUrl"].lower()
                        ):
                            image_result = img
                            break
                
                return SearchResponse(
                    title=result.get("title", ""),
                    snippet=result.get("snippet", ""),
                    source=result.get("link", ""),
                    image_url=image_result["imageUrl"] if image_result else None,
                    image_source=image_result["source"] if image_result else None
                )
            else:
                return {"error": "No results found"}
                
    except Exception as e:
        logger.error(f"Error performing search: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Could not perform search: {str(e)}"})

@app.on_event("startup")
async def on_startup():
    await user_db.initialize_database()
    async with user_db.AsyncSessionLocal() as session:   # ← 手動開 session
        await user_db.maybe_analyze_vectors(session)
    logger.info("✅ 資料庫初始化完成")

@app.post("/user/signup", tags=["user"])
async def create_user(
    user: UserSchema = Body(...),
    db: AsyncSession = Depends(user_db.get_db)
):
    result = await db.execute(select(user_db.User).filter_by(email=user.email))
    existing_user = result.scalars().first()

    if existing_user:
        raise HTTPException(status_code=400, detail="❌ Email already registered.")

    new_user = user_db.User(
        email=user.email,
        hashed_password=user_db.hash_password(user.password)
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return {"message": "Signup successful"}
    # return sign_jwt(user.email)

@app.post("/user/login", tags=["user"])
async def login_user(
    user: UserLoginSchema = Body(...),
    db: AsyncSession = Depends(user_db.get_db)
):
    # 1️⃣ 驗證使用者
    result = await db.execute(select(user_db.User).filter_by(email=user.email))
    db_user = result.scalars().first()

    if not db_user or not user_db.verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="❌ Invalid email or password.")
    print("👉 收到登入請求", user.email)
    print("👉 驗證結果", user_db.verify_password(user.password, db_user.hashed_password))
    print("👉 記得我", user.remember_me)

    # 透過email取得使用者id
    user_id = db_user.id

    # 2️⃣ 產生 JWT 字串（只要 token，不需要包 dict）
    token = sign_jwt(user_id, user.email)["access_token"]     # sign_jwt 仍回 {'access_token': xxx}


    return {"message": "Login successful", "access_token": token}
    

@app.get("/sessionJWT", tags=["user"])
async def check_session_jwt(token_payload: dict = Depends(JWTBearer())):
    print(f"👉 JWT payload: {token_payload}")
    return {
        "msg": "Session is valid",
        "user_id": token_payload["user_id"],
        "user_email": token_payload["email"]
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    if is_onLine:
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        uvicorn.run(app, host="127.0.0.1", port=8888)
