import os
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()  # Carga variables de .env

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "secret_key_por_defecto")  # Para JWT
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///test.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "jwt_secret_key_por_defecto")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    MAIL_SERVER = os.getenv("MAIL_SERVER", "smtp.gmail.com")
    MAIL_PORT = int(os.getenv("MAIL_PORT", 587))
    MAIL_USERNAME = os.getenv("MAIL_USERNAME", "tu_correo@gmail.com")
    MAIL_PASSWORD = os.getenv("MAIL_PASSWORD", "tu_password")
    MAIL_USE_TLS = True
    MAIL_USE_SSL = False
