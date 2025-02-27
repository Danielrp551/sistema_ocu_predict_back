from flask import Flask
from flask_cors import CORS
from config import Config
from .extensions import db, jwt
from .controllers.auth_controller import auth_bp
from .controllers.predict_controller import predict_bp
from .controllers.history_controller import history_bp
from .controllers.home_controller import home_bp
from .services.aws_service import download_model_from_s3
from .services.predict_service import initialize_model
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(app, resources={r"/*": {"origins": "*"}})

    with app.app_context():
        local_model_path = "src/modelosDeep/modelo_resnet18_patience_5_drpt_02.pth"
        s3_model_key = os.environ.get("MODEL_S3_KEY")
        
        print(f"🌍 S3 Model Key: {s3_model_key}")  # Verificar si se obtiene correctamente

        download_model_from_s3(local_model_path, s3_model_key)

        initialize_model()

    # Inicializar extensiones
    db.init_app(app)
    jwt.init_app(app)

    # Registrar blueprints
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(predict_bp, url_prefix="/predict")
    app.register_blueprint(history_bp, url_prefix="/history")
    app.register_blueprint(home_bp, url_prefix="/home")

    # Crear tablas si no existen
    with app.app_context():
        db.create_all()

    return app
