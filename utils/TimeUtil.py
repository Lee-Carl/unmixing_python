from datetime import datetime


class TimeUtil:
    @staticmethod
    def getCurrentTime(time_format: str = "%Y-%m-%d %H:%M:%S"):
        start_time = datetime.now()
        return start_time.strftime(time_format)

