"""
MuJoCo 工具函数 — 解决中文路径问题

MuJoCo 的 from_xml_path() 无法处理中文路径，使用 from_xml_string() 替代。
"""
import mujoco


def load_model_safe(xml_path):
    """
    安全加载 MuJoCo 模型（支持中文路径）

    Args:
        xml_path: XML 文件路径（可以包含中文）

    Returns:
        mujoco.MjModel
    """
    try:
        # 先尝试直接加载（如果路径是纯 ASCII）
        return mujoco.MjModel.from_xml_path(xml_path)
    except (ValueError, UnicodeDecodeError):
        # 如果失败（中文路径），读取文件内容后用 from_xml_string
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_string = f.read()
        return mujoco.MjModel.from_xml_string(xml_string)
