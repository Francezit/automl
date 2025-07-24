import os
import random
import shutil


class TempContext():

    @property
    def foldername(self) -> str:
        return self.__folder

    def __init__(self, folder: str) -> None:
        self.__folder = folder

    def __enter__(self):
        if self.__folder:
            os.makedirs(self.__folder, exist_ok=True)
            return self
        else:
            raise Exception("The context is disposed")

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if self.__folder:
            shutil.rmtree(self.__folder)
            self.__folder = None
        else:
            raise Exception("The context is disposed")


def create_temp_context(basefolder: str) -> TempContext:
    folder = None
    while folder is None or os.path.exists(folder):
        folder = os.path.join(basefolder, f"temp_{random.randint(1000,9999)}")
    return TempContext(folder)


def generate_temp_folder(basefolder: str):
    folder = None
    while folder is None or os.path.exists(folder):
        folder = os.path.join(basefolder, f"temp_{random.randint(1000,9999)}")
    os.makedirs(folder, exist_ok=True)
    return folder


def remove_temp_folder(temp_folder: str | TempContext):
    if isinstance(temp_folder, str):
        shutil.rmtree(temp_folder)
    elif isinstance(temp_folder, TempContext):
        shutil.rmtree(temp_folder)
    else:
        raise Exception("Invalid Operation")


__init__ = ["generate_temp_folder", "create_temp_context",
            "remove_temp_folder", "TempContext"]
