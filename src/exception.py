import os


def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"

    error_message = (
        f"Error occurred python script name [{file_name}] line number [{line_number}] "
        f"error message [{str(error)}]"
    )

    return error_message


class AgenticException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message