from enum import Enum

class AppointmentStatus(Enum):
    BOOKED = "Booked"
    CANCELED = "Canceled"
    COMPLETED = "Completed"