from app.extensions import db
from werkzeug.security import generate_password_hash, check_password_hash
from app.models.roles import RoleEnum

class User(db.Model):
    __tablename__ = "users"

    user_id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    username = db.Column(db.String(80), nullable=False)
    birth_date = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(1), nullable=False)  
    role = db.Column(db.Enum(RoleEnum), nullable=False)

    patient_profile = db.relationship("Patient", back_populates="user", uselist=False, lazy="select")
    doctor_profile = db.relationship("Doctor", back_populates="user", uselist=False, lazy="select")
    notifications = db.relationship("Notification", back_populates="user", lazy="dynamic")

    def set_password(self, raw_password):
        self.password = generate_password_hash(raw_password)

    def check_password(self, raw_password):
        return check_password_hash(self.password, raw_password)
