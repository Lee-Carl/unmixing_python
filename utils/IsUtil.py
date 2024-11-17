class IsUtil:
    @staticmethod
    def isAttr(instanceObj: object, prop: str) -> bool:
        return hasattr(instanceObj, prop)
