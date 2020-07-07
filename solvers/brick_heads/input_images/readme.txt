"""
    衣服 list[0]: 颜色   list[1]: 轮廓
    
    -1：该属性没有检测结果，
    性别 value 0，1 分别表示 女性 男性
    眼镜 value -1，1，2 分别表示 无眼镜 小框眼镜 大框眼镜
    余下属性，value 1 2 3，分别表示 小 中 大（短 中 长, 矮 中 高）
"""

description = {
    'gender': -1,
    'brow': -1, # 美貌
    'eye': -1, #眼睛
    'ear': -1, #耳朵
    'nose': -1, # 鼻子
    'mouth': -1, #嘴巴
    'hair': -1,
    'skin': -1,
    'glasses': -1,
    'clothes': [[], []]
}