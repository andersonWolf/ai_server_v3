from pydantic import BaseModel, Field, EmailStr


class PostSchema(BaseModel):
    id: int = Field(default=None)
    title: str = Field(...)
    content: str = Field(...)

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Securing FastAPI applications with JWT.",
                "content": "In this tutorial, you'll learn how to secure your application by enabling authentication using JWT. We'll be using PyJWT to sign, encode and decode JWT tokens...."
            }
        }


class UserSchema(BaseModel):
    email: EmailStr = Field(...)
    password: str = Field(...)
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "abdulazeez@x.com",
                "password": "weakpassword",
            }
        }

class UserLoginSchema(BaseModel):
    email: EmailStr = Field(...)
    password: str = Field(...)
    remember_me: bool = Field(default=False)  # ✅ 新增這一行


    class Config:
        json_schema_extra = {
            "example": {
                "email": "abdulazeez@x.com",
                "password": "weakpassword",
                "remember_me": True  # ✅ 示例也補上
            }
        }
