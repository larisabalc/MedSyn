from app.extensions import db

class Availability(db.Model):
    __tablename__ = "availability"

    availability_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("doctor.doctor_id"), nullable=False)
    availability_date = db.Column(db.Date, nullable=False)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    availability_status = db.Column(db.Boolean, default=True)

    doctor = db.relationship("Doctor", back_populates="availabilities")
    appointments = db.relationship("Appointment", back_populates="availability")

    def __repr__(self):
        return (
            f"<Availability Doctor={self.doctor_id} "
            f"{self.availability_date} {self.start_time}-{self.end_time}>"
        )
