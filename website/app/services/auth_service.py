import re
from app.models.user import User
from app.extensions import db
from app.models.roles import RoleEnum
from datetime import datetime

EMAIL_REGEX = r"^[A-Za-z0-9._%+-]+@(patient|doctor)(\.com)?$"

def extract_role_from_email(email):
    if email.endswith("@patient.com"):
        return RoleEnum.PATIENT
    if email.endswith("@doctor.com"):
        return RoleEnum.DOCTOR
    return None

def is_valid_email(email):
    return re.match(EMAIL_REGEX, email) is not None

def is_valid_gender(gender):
    return gender in ("F", "M")

def register_user(email, username, password, gender, birthdate):
    if not is_valid_email(email):
        return None, "Invalid email format. Must be '@patient.com' or '@doctor.com'."

    if not is_valid_gender(gender):
        return None, "Gender must be 'M' or 'F'."

    role = extract_role_from_email(email)
    if not role:
        return None, "Email must belong to a patient or doctor domain."

    if User.query.filter_by(email=email).first():
        return None, "Email already exists."

    birthdate_obj = None
    if birthdate:
        try:
            birthdate_obj = datetime.strptime(birthdate, "%Y-%m-%d").date()
        except ValueError:
            return None, "Birthdate must be in YYYY-MM-DD format."

    user = User(
        email=email,
        username=username,
        password=password,
        role=role,
        gender=gender,
        birth_date=birthdate_obj
    )
    user.set_password(password)

    db.session.add(user)
    db.session.commit() 

    try:
        if role == RoleEnum.PATIENT:
            from app.models.patient import Patient
            from app.models.notification import Notification

            patient = Patient(
                patient_id=user.user_id,
                insurance_no=None,  
                emergency_contact=None  
            )
            db.session.add(patient)
            db.session.commit()

            notification = Notification(
                user_id=user.user_id,
                title="Complete your profile",
                message="Your patient profile is incomplete. Please set insurance number and emergency contact."
            )
            db.session.add(notification)
            db.session.commit()

        elif role == RoleEnum.DOCTOR:
            from app.models.doctor import Doctor
            from app.models.specialization import Specialization
            from app.models.notification import Notification

            default_spec = Specialization.query.filter_by(specialization_name="Unassigned").first()
            if not default_spec:
                default_spec = Specialization(specialization_name="Unassigned")
                db.session.add(default_spec)
                db.session.commit()

            doctor = Doctor(
                doctor_id=user.user_id,
                licence_no=None,  
                specialization_id=default_spec.specialization_id 
            )
            db.session.add(doctor)
            db.session.commit()

            notification = Notification(
                user_id=user.user_id,
                title="Complete your profile",
                message="Please update your licence number to complete your profile."
            )
            db.session.add(notification)
            db.session.commit()

    except Exception as e:
        db.session.rollback()
        return None, f"Error creating role-specific info: {str(e)}"

    return user, None


def authenticate_user(email, password):
    user = User.query.filter_by(email=email).first()
    if not user:
        return None, "Email not found."

    if not user.check_password(password):
        return None, "Incorrect password."

    return user, None
