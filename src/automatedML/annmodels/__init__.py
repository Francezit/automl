
from ..ann import ANNArchitecture
from ..models import HyperparametersManager


def get_supported_models():
    import os
    names = []
    for f in os.listdir(os.path.dirname(__file__)):
        if not f.startswith("_"):
            names.append(f.replace(".py", ""))
    return names


def get_architecture(name: str, hp_manager: HyperparametersManager = None, **kargs) -> ANNArchitecture:
    import os
    filename = os.path.join(os.path.dirname(__file__), f"{name.lower()}.py")
    assert os.path.exists(filename), f"{name} does not exists"

    inner_code = f'''
from .{name.lower()} import get_architecture
base_class=get_architecture()
'''
    loc = {}
    exec(inner_code, globals(), loc)
    base_class = loc["base_class"]
    model = base_class(hp_manager=hp_manager, **kargs)
    return model


__all__ = ["get_supported_models", "get_annmodel"]


# https://deci.ai/blog/sota-dnns-overview/
