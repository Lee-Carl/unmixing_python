from typing import Dict


class EGU:
    def __init__(self, dtrue):
        self.dtrue = dtrue

    def __call__(self) -> Dict:
        print("init: EGU")
        self.dtrue['src'] = self.dtrue['name']
        return self.dtrue
