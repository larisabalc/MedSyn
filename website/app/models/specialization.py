from app.extensions import db

class Specialization(db.Model):
    __tablename__ = "specialization"

    specialization_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    specialization_name = db.Column(db.String(150), nullable=False, unique=True)

    doctors = db.relationship("Doctor", back_populates="specialization", lazy="dynamic")
    services = db.relationship("Service", back_populates="specialization")

    def __repr__(self):
        return f"<Specialization {self.specialization_name}>"
