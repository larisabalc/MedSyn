from datetime import datetime

from flask import Blueprint, render_template, redirect, send_from_directory, url_for, session, request, flash

from sqlalchemy import or_
from sqlalchemy.orm import joinedload

from app.extensions import db

from app.models.roles import RoleEnum
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor
from app.models.specialization import Specialization
from app.models.appointment import Appointment
from app.models.medical_record import MedicalRecord
from app.models.notification import Notification
from app.models.availability import Availability

doctor_bp = Blueprint("doctor", __name__, url_prefix="/doctor")

@doctor_bp.route("/dashboard")
def dashboard():
    if session.get("role") != RoleEnum.DOCTOR:
        return redirect(url_for("auth.login"))

    user_id = session.get("user_id")
    user = User.query.get(user_id)

    return render_template("doctor/dashboard.html", user=user)

@doctor_bp.route('/video_call/<int:appointment_id>')
def video_call(appointment_id):
    user_id = session.get("user_id")
    user = User.query.get(user_id)

    patient_id = Appointment.query.get(appointment_id).patient_id
    patient = Patient.query.get(patient_id).user.username

    print(user.username)
    print(patient)
    
    return redirect(f"http://localhost:8080/video_call.html?room={appointment_id}&doctor={user.username}&patient={patient}&role=doctor")

# region Availability
@doctor_bp.route("/availability/delete/<int:availability_id>", methods=["POST"])
def delete_availability(availability_id):
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("auth.login"))

    availability = Availability.query.get_or_404(availability_id)

    if availability.doctor_id != user_id:
        flash("Unauthorized action.", "danger")
        return redirect(url_for("doctor.availability"))

    db.session.delete(availability)
    db.session.commit()

    flash("Availability slot deleted.", "success")
    return redirect(url_for("doctor.availability"))


@doctor_bp.route("/availability", methods=["GET", "POST"])
def availability():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("auth.login"))

    user = User.query.get(user_id)
    doctor = Doctor.query.get(user.user_id)

    if not doctor:
        flash("Doctor profile not found.", "danger")
        return redirect(url_for("auth.logout"))

    if request.method == "POST":
        date_str = request.form.get("availability_date")
        start_time_str = request.form.get("start_time")
        end_time_str = request.form.get("end_time")

        if not date_str or not start_time_str or not end_time_str:
            flash("All fields are required.", "danger")
            return redirect(url_for("doctor.availability"))

        start_time = datetime.strptime(start_time_str, "%H:%M").time()
        end_time = datetime.strptime(end_time_str, "%H:%M").time()

        if start_time >= end_time:
            flash("Start time must be before end time.", "danger")
            return redirect(url_for("doctor.availability"))

        availability = Availability(
            doctor_id=doctor.doctor_id,
            availability_date=datetime.strptime(date_str, "%Y-%m-%d").date(),
            start_time=start_time,
            end_time=end_time,
            availability_status=True
        )

        db.session.add(availability)
        db.session.commit()

        flash("Availability slot added successfully.", "success")
        return redirect(url_for("doctor.availability"))

    availabilities = Availability.query.filter_by(doctor_id=doctor.doctor_id).order_by(Availability.availability_date, Availability.start_time).all()

    return render_template(
        "doctor/availability.html",
        user=user,
        doctor=doctor,
        availabilities=availabilities
    )
# endregion

# region MedicalRecord
@doctor_bp.route("/medical_record/<int:record_id>/delete", methods=["POST"])
def delete_medical_record(record_id):
    user_id = session.get("user_id")

    if not user_id:
        return "Not logged in", 401

    record = MedicalRecord.query.filter_by(record_id=record_id).first()

    if not record:
        flash("Medical record not found.", "danger")
        return redirect(request.referrer or url_for("doctor.patients"))

    if record.doctor_id is not None and record.doctor_id != user_id:
        flash("You cannot delete a medical record created by another doctor.", "danger")
        return redirect(request.referrer or url_for("doctor.patients"))

    db.session.delete(record)
    db.session.commit()

    flash("Medical record deleted successfully.", "success")
    return redirect(request.referrer or url_for("doctor.patients"))


@doctor_bp.route("/appointments/<int:appointment_id>/medical_record/create", methods=["GET", "POST"])
def create_medical_record(appointment_id):
    if session.get("role") != RoleEnum.DOCTOR:
        return redirect(url_for("auth.login"))

    appt = Appointment.query.get_or_404(appointment_id)

    if appt.doctor_id != session.get("user_id") or appt.status != "Completed":
        flash("Cannot create a medical record for this appointment.", "danger")
        return redirect(url_for("doctor.appointments"))

    if request.method == "POST":
        symptoms = request.form.get("symptoms")
        diagnosis = request.form.get("diagnosis")
        treatment = request.form.get("treatment")
        cholesterol_lvl = request.form.get("cholesterol_lvl")
        blood_pressure_lvl = request.form.get("blood_pressure_lvl")

        new_record = MedicalRecord(
            doctor_id=session.get("user_id"),
            patient_id=appt.patient_id,
            symptoms=symptoms,
            diagnosis=diagnosis,
            treatment=treatment,
            cholesterol_lvl=int(cholesterol_lvl) if cholesterol_lvl else None,
            blood_pressure_lvl=blood_pressure_lvl,
            is_generated=False
        )

        db.session.add(new_record)
        db.session.commit()

        notif_message = (
            f"Dr. {appt.doctor.user.username} has created a medical record for your "
            f"appointment on {appt.availability.availability_date.strftime('%Y-%m-%d')}."
        )
        patient_notification = Notification(
            user_id=appt.patient.user.user_id,
            appointment_id=appt.appointment_id,
            title="New Medical Record",
            message=notif_message,
            send_time=datetime.utcnow()
        )

        db.session.add(patient_notification)
        db.session.commit()

        flash("Medical record created successfully.", "success")
        return redirect(url_for("doctor.appointments"))

    return render_template("doctor/create_medical_record.html", appointment=appt)
# endregion

# region Patients
@doctor_bp.route("/patients/<int:patient_id>")
def patient_profile(patient_id):
    user_id = session.get("user_id")
    if not user_id:
        return "Not logged in", 401

    patient = Patient.query.options(joinedload(Patient.user)).filter_by(patient_id=patient_id).first()
    if not patient:
        return "Patient not found", 404

    medical_records = MedicalRecord.query.filter(
        MedicalRecord.patient_id == patient_id,
        or_(
            MedicalRecord.doctor_id == user_id,
            MedicalRecord.is_generated == True
        )
    ).all()

    return render_template("doctor/patient_profile.html",
                            patient=patient,
                            medical_records=medical_records)

@doctor_bp.route("/patients")
def patients():
    user_id = session.get("user_id")
    if not user_id:
        return "Not logged in", 401

    appointments = (
        Appointment.query
        .filter_by(doctor_id=user_id)
        .options(joinedload(Appointment.patient))
        .all()
    )

    patients = {appt.patient.patient_id: appt.patient for appt in appointments}

    return render_template("doctor/patients.html", patients=list(patients.values()))
# endregion

# region Appointments
@doctor_bp.route("/appointments/complete/<int:appointment_id>", methods=["POST"])
def complete_appointment(appointment_id):
    if session.get("role") != RoleEnum.DOCTOR:
        return redirect(url_for("auth.login"))

    appointment = Appointment.query.get_or_404(appointment_id)

    if appointment.doctor_id != session.get("user_id"):
        return "Unauthorized", 403

    if appointment.status != "Booked":
        flash("Only booked appointments can be completed.", "warning")
        return redirect(url_for("doctor.appointments"))

    appointment.status = "Completed"
    db.session.commit()

    flash("Appointment marked as completed.", "success")
    return redirect(url_for("doctor.appointments"))


@doctor_bp.route("/appointments/cancel/<int:appointment_id>", methods=["POST"])
def cancel_appointment(appointment_id):
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("auth.login"))

    user = User.query.get(user_id)
    if user.role != "DOCTOR":
        return "Unauthorized", 403

    appointment = Appointment.query.get(appointment_id)
    if not appointment or appointment.doctor.user.user_id != user_id:
        return "Appointment not found", 404

    appointment.status = "Canceled"
    appointment.availability.availability_status = True

    Notification.query.filter_by(appointment_id=appointment.appointment_id).delete(synchronize_session=False)
    db.session.commit()

    patient_user = appointment.patient.user
    notif_message = (
        f"Your appointment with Dr. {appointment.doctor.user.username} "
        f"on {appointment.availability.availability_date.strftime('%Y-%m-%d')} "
        f"from {appointment.availability.start_time.strftime('%H:%M')} "
        f"to {appointment.availability.end_time.strftime('%H:%M')} has been canceled by the doctor."
    )
    patient_notification = Notification(
        user_id=patient_user.user_id,
        appointment_id=appointment.appointment_id,
        title="Appointment Canceled",
        message=notif_message,
        send_time=datetime.utcnow()
    )

    db.session.add(patient_notification)
    db.session.commit()

    return redirect(url_for("doctor.appointments"))

@doctor_bp.route("/appointments")
def appointments():
    if session.get("role") != RoleEnum.DOCTOR:
        return redirect(url_for("auth.login"))

    doctor_id = session.get("user_id")
    appointments = Appointment.query.filter_by(doctor_id=doctor_id).all()
    current_time = datetime.utcnow()

    return render_template("doctor/appointments.html", appointments=appointments, current_time=current_time)
# endregion

# region Profile
@doctor_bp.route("/profile")
def profile():
    if session.get("role") != RoleEnum.DOCTOR:
        return redirect(url_for("auth.login"))
    
    user_id = session.get("user_id")

    if not user_id:
        return redirect(url_for("auth.login"))

    user = User.query.get(user_id)
    doctor = Doctor.query.filter_by(doctor_id=user_id).first()

    if not doctor:
        flash("Doctor profile not found.", "danger")
        return redirect(url_for("auth.login"))

    return render_template(
        "doctor/profile.html",
        user=user,
        doctor=doctor
    )

@doctor_bp.route("/profile/edit", methods=["GET", "POST"])
def edit_profile():
    user_id = session.get("user_id")

    if not user_id:
        return redirect(url_for("auth.login"))

    user = User.query.get(user_id)
    doctor = Doctor.query.filter_by(doctor_id=user_id).first()
    specializations = Specialization.query.all()

    if not doctor:
        flash("Doctor profile not found.", "danger")
        return redirect(url_for("doctor.profile"))

    if request.method == "POST":
        user.username = request.form.get("username")
        user.gender = request.form.get("gender")
        user.birth_date = request.form.get("birth_date")

        doctor.licence_no = request.form.get("licence_no")
        doctor.specialization_id = request.form.get("specialization_id")

        db.session.commit()

        flash("Profile updated successfully.", "success")
        return redirect(url_for("doctor.profile"))

    return render_template(
        "doctor/edit_profile.html",
        user=user,
        doctor=doctor,
        specializations=specializations
    )
# endregion

# region Notifications
@doctor_bp.route("/notifications", methods=["GET"])
def notifications():
    if session.get("role") != RoleEnum.DOCTOR:
        return redirect(url_for("auth.login"))

    doctor_id = session.get("user_id")
    now = datetime.utcnow()

    notifications = Notification.query.filter(
        Notification.user_id == doctor_id,
        (Notification.send_time == None) | (Notification.send_time <= now)
    ).order_by(Notification.created_at.desc()).all()

    unread_count = Notification.query.filter(
        Notification.user_id == doctor_id,
        Notification.is_read == False,
        (Notification.send_time == None) | (Notification.send_time <= now)
    ).count()

    session["unseen_notifications"] = unread_count

    return render_template(
        "doctor/notifications.html",
        notifications=notifications
    )

@doctor_bp.route("/notifications/mark_seen/<int:notif_id>", methods=["POST"])
def mark_notification_seen(notif_id):
    if session.get("role") != RoleEnum.DOCTOR:
        return redirect(url_for("auth.login"))

    notif = Notification.query.get_or_404(notif_id)

    if notif.user_id != session.get("user_id"):
        return redirect(url_for("doctor.notifications"))

    notif.is_read = True
    notif.read_at = datetime.utcnow()
    db.session.commit()

    return redirect(url_for("doctor.notifications"))
# endregion