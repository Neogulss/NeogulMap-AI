# 에러 확인용으로 추가했습니다.
class PredictionError(Exception):
    def __init__(self, public_message: str):
        self.public_message = public_message
        super().__init__(public_message)