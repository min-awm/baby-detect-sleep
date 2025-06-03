import os

def get_abs_path(relative_path: str, base_file: str = __file__) -> str:
    base_dir = os.path.dirname(os.path.abspath(base_file))
    return os.path.abspath(os.path.join(base_dir, relative_path))