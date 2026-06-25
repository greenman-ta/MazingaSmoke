class MediaError(Exception):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)

class GeneralInputError(Exception):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)