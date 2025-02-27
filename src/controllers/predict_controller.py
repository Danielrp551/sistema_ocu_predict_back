from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from ..services.predict_service import predict_disease_service

predict_bp = Blueprint("predict_bp", __name__)

@predict_bp.route("/", methods=["POST"])
@jwt_required()
def predict():
    # Verificar si se envió 'file'
    if 'file' not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    user_id = get_jwt_identity()  # ID del usuario que hace la petición

    try:
        result = predict_disease_service(file, user_id)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
