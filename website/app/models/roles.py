from enum import Enum

class RoleEnum(str, Enum):
    PATIENT = "PATIENT"
    DOCTOR = "DOCTOR"