from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from ..services.home_service import get_home_stats_service

home_bp = Blueprint("home_bp", __name__)

@home_bp.route("/stats", methods=["GET"])
@jwt_required()
def get_home_stats():
    """
    Retorna estadísticas de la pantalla de inicio,
    filtradas por un rango de fechas (hoy, semana, mes, etc.)
    """
    user_id = get_jwt_identity()
    date_range = request.args.get("range", "hoy")  # por defecto "hoy"

    try:
        stats = get_home_stats_service(user_id, date_range)
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
