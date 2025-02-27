from flask import Blueprint, jsonify, send_file, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from ..services.history_service import get_user_history_service, update_doctor_feedback_service, get_history_by_id_with_image_service

history_bp = Blueprint("history_bp", __name__)

@history_bp.route("/", methods=["GET"])
@jwt_required()
def get_history():
    user_id = get_jwt_identity()
    try:
        history_records = get_user_history_service(user_id, request.args)
        return jsonify(history_records), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@history_bp.route("/download", methods=["GET"])
@jwt_required()
def download_image():
    # Ejemplo de cómo podrías manejar la descarga
    file_path = request.args.get("file_path")
    # Aquí asume que 'file_path' es la ruta real en disco
    # Asegúrate de validar el path para seguridad
    try:
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@history_bp.route("/feedback/<int:history_id>", methods=["PUT"])
@jwt_required()
def update_feedback(history_id):
    data = request.get_json()
    feedback = data.get("feedback")  # "correcto" o "incorrecto"   
    try:
        update_doctor_feedback_service(history_id, feedback)
        return jsonify({"message": "Feedback recibido"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@history_bp.route("/<int:history_id>", methods=["GET"])
@jwt_required()
def get_history_by_id(history_id):
    """
    Retorna la información detallada de una solicitud en particular.
    """
    user_id = get_jwt_identity()
    try:
        record_dict = get_history_by_id_with_image_service(user_id, history_id)
        return jsonify(record_dict), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400