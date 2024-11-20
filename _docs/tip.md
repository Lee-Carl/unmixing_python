已知的python版本对代码的影响
1. 在python3.9+中，`typing_extensions`包的内容会被包含在`typing`包中
2. 在python3.9+中，`Union[str, None]`可以写成`str | None`（早这样多好）