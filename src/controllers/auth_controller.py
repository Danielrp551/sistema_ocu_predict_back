from flask import Blueprint, request, jsonify
from ..services.auth_service import (
    register_user_service, 
    verify_account_service, 
    login_user_service,
    request_password_reset_service,
    reset_password_service
)

auth_bp = Blueprint("auth_bp", __name__)

@auth_bp.route("/register", methods=["POST"])
def register_user():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    name = data.get("name")

    try:
        register_user_service(email, password, name)
        return jsonify({"message": "Usuario registrado con éxito. Se envió un código al correo."}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@auth_bp.route("/verify", methods=["POST"])
def verify_account():
    data = request.get_json()
    email = data.get("email")
    code = data.get("code")

    try:
        verify_account_service(email, code)
        return jsonify({"message": "Cuenta verificada con éxito"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    try:
        token = login_user_service(email, password)
        return jsonify({"access_token": token}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 401

@auth_bp.route("/request-password-reset", methods=["POST"])
def request_password_reset():
    data = request.get_json()
    email = data.get("email")

    try:
        request_password_reset_service(email)
        return jsonify({"message": "Se envió un código de verificación para restablecer la contraseña"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@auth_bp.route("/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json()
    email = data.get("email")
    code = data.get("code")
    new_password = data.get("new_password")

    try:
        reset_password_service(email, code, new_password)
        return jsonify({"message": "Contraseña actualizada con éxito"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
