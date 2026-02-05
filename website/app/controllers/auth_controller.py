from flask import Blueprint, render_template, request, redirect, url_for, session

from app.services.auth_service import register_user, authenticate_user

from app.models.roles import RoleEnum

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/patient/dashboard")
def patient_dashboard():
    user_id = session.get("user_id")

    if not user_id:
        return redirect(url_for("auth.login"))
    
    from app.models.user import User
    user = User.query.get(user_id)

    return render_template(
        "patient/dashboard.html",
        user=user
    )

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        user, error = register_user(
            email=request.form["email"],
            username=request.form["username"],
            password=request.form["password"],
            gender=request.form["gender"],
            birthdate=request.form["birthdate"]
        )

        if error:
            return render_template("auth/register.html", error=error, form=request.form)

        return redirect(url_for("auth.login"))

    return render_template("auth/register.html", hide_navbar=True, error=None, form={})


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        user, error = authenticate_user(
            email=request.form["email"],
            password=request.form["password"]
        )

        if user:
            session["user_id"] = user.user_id
            session["role"] = user.role

            if user.role == RoleEnum.PATIENT:
                return redirect(url_for("auth.patient_dashboard"))
            elif user.role == RoleEnum.DOCTOR:
                return redirect(url_for("doctor.dashboard"))

    return render_template("auth/login.html", hide_navbar=True,error=error, form=request.form)

@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))
