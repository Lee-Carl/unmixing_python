import os

libs = []
with open("a.txt", "r", encoding="utf-8") as file:
    for line in file:
        libs.append(line.split(".")[1].strip(" "))

print(len(libs))

def has_duplicates(lst):
    seen = set()
    for x in lst:
        # 如果在集合中已经见过这个元素，则有重复
        if x in seen:
            return True
        seen.add(x)
    return False


# 示例

result = has_duplicates(libs)
print("是否有重复元素：", result)
