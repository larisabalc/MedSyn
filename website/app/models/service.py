from app.extensions import db

class Service(db.Model):
    __tablename__ = "service"

    service_id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    specialization_id = db.Column(db.Integer, db.ForeignKey("specialization.specialization_id"), nullable=False)

    service_name = db.Column(db.String(200), nullable=False)

    specialization = db.relationship("Specialization", back_populates="services")
    appointments = db.relationship("Appointment", back_populates="service")

    def __repr__(self):
        return f"<Service {self.service_name}>"
