import sys
import logging

# ---------------- LOGGING SETUP ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- ERROR DETAIL FUNCTION ----------------
def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return f"Error: {str(error)} (No traceback available)"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error occurred in script: [{file_name}] "
        f"at line number: [{line_number}] "
        f"error message: [{str(error)}]"
    )

# ---------------- CUSTOM EXCEPTION ----------------
class CustomException(Exception):
    def __init__(self, error, error_detail=sys):
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message

# ---------------- MAIN TEST ----------------
                  # ✅ clean output (no crash)