from ..extensions import db
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=True)
    is_verified = db.Column(db.Boolean, default=False)

    # Podrías almacenar un verification_code temporalmente
    verification_code = db.Column(db.String(50), nullable=True)
    verification_code_expiration = db.Column(db.DateTime, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def set_password(self, raw_password):
        self.password = generate_password_hash(raw_password)

    def check_password(self, raw_password):
        return check_password_hash(self.password, raw_password)
