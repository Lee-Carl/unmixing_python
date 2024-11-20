from typing_extensions import get_args

from custom_types.type_alias import HsiData
from typing import Any


class TypeUtil:
    @staticmethod
    def is_HsiData(data: Any) -> bool:
        return isinstance(data, get_args(HsiData))
