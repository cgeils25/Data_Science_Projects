from datetime import datetime

class date_n_time():
    """
    Class to get current date and time in the format: dd_mm_yyyy_hh_mm_ss
    """
    def __init__(self):
        self.time_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    def __str__(self):
        return self.time_now