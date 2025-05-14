from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .auth_handler import decode_jwt
import os
from dotenv import load_dotenv
load_dotenv(override=True)




class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)
    
    async def __call__(self, request: Request):
       
        # 👉 從 Authorization header 抓取
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        if not credentials or credentials.scheme.lower() != "bearer":
            raise HTTPException(status_code=403, detail="Invalid or missing Authorization token.")
        token = credentials.credentials
        
        if not self.verify_jwt(token):
            raise HTTPException(status_code=403, detail="Invalid or expired token.")

        payload = decode_jwt(token)
        return payload
    
    def verify_jwt(self, jwtoken: str) -> bool:
        isTokenValid: bool = False

        try:
            payload = decode_jwt(jwtoken)
        except:
            payload = None
        if payload:
            isTokenValid = True

        return isTokenValid
