import time
from typing import Dict

import jwt
from decouple import config


JWT_SECRET = config("secret")
JWT_ALGORITHM = config("algorithm")
JWT_EXPIRE_SECONDS = 60 * 60 * 24  # 1 天


def token_response(token: str):
    return {
        "access_token": token
    }

def sign_jwt(user_id: int, email: str, remember_me: bool = False) -> Dict[str, str]:
    expire_seconds = JWT_EXPIRE_SECONDS * 30 if remember_me else JWT_EXPIRE_SECONDS  # 30天 或 1天
    payload = {
        "user_id": user_id,
        "email": email,
        "expires": time.time() + expire_seconds
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return token_response(token)



def decode_jwt(token: str) -> dict:
    try:
        decoded_token = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return decoded_token if decoded_token["expires"] >= time.time() else None
    except:
        return {}
