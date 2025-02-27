import datetime, random, string
from ..extensions import db
from ..models.user_model import User
from ..utils.email_service import send_verification_email
from ..utils.token_manager import generate_jwt_token

def register_user_service(email, password, name):
    # Verificar si ya existe
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        raise Exception("El email ya está registrado")

    # Crear usuario
    user = User(email=email, name=name)
    user.set_password(password)

    # Generar código de verificación
    code = ''.join(random.choices(string.digits, k=6))  # 6 dígitos
    user.verification_code = code
    user.verification_code_expiration = datetime.datetime.utcnow() + datetime.timedelta(minutes=30)

    db.session.add(user)
    db.session.commit()

    # Enviar correo con el código
    send_verification_email(email, code)

def verify_account_service(email, code):
    user = User.query.filter_by(email=email).first()
    if not user:
        raise Exception("Usuario no encontrado")

    if user.verification_code != code:
        raise Exception("Código de verificación incorrecto")

    if datetime.datetime.utcnow() > user.verification_code_expiration:
        raise Exception("El código ha expirado")

    user.is_verified = True
    user.verification_code = None
    user.verification_code_expiration = None
    db.session.commit()

def login_user_service(email, password):
    user = User.query.filter_by(email=email).first()
    if not user:
        raise Exception("Usuario no encontrado")
    if not user.check_password(password):
        raise Exception("Credenciales inválidas")
    if not user.is_verified:
        raise Exception("La cuenta no está verificada")

    # Generar JWT
    token = generate_jwt_token(user.id)
    return token

def request_password_reset_service(email):
    user = User.query.filter_by(email=email).first()
    if not user:
        raise Exception("Usuario no encontrado")

    # Generar nuevo código de verificación
    code = ''.join(random.choices(string.digits, k=6))
    user.verification_code = code
    user.verification_code_expiration = datetime.datetime.utcnow() + datetime.timedelta(minutes=30)

    db.session.commit()

    # Enviar correo
    send_verification_email(email, code)

def reset_password_service(email, code, new_password):
    user = User.query.filter_by(email=email).first()
    if not user:
        raise Exception("Usuario no encontrado")

    if user.verification_code != code:
        raise Exception("Código de verificación incorrecto")

    if datetime.datetime.utcnow() > user.verification_code_expiration:
        raise Exception("El código ha expirado")

    # Cambiar la contraseña
    user.set_password(new_password)
    # Limpiar código y exp
    user.verification_code = None
    user.verification_code_expiration = None
    db.session.commit()
