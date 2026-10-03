import json
import os
import threading
from typing import Optional
from util.logger import Logger

config_logger = Logger("Configs")

CONFIG_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
)
_SAVE_DEBOUNCE_S = 0.4

loaded_config = {}
_save_lock = threading.Lock()
_save_timer: Optional[threading.Timer] = None

if not os.path.exists(CONFIG_FILE_PATH):
    with open(CONFIG_FILE_PATH, 'w') as config_file:
        json.dump({}, config_file)
    config_logger.Warn("Couldn't find existing config file")
with open(CONFIG_FILE_PATH, 'r') as config_file:
    loaded_config = json.load(config_file)
    config_logger.Log(f"Loaded config: {loaded_config}")


def save_config() -> None:
    global loaded_config
    with _save_lock:
        with open(CONFIG_FILE_PATH, 'w') as config_file:
            json.dump(loaded_config, config_file, indent=4)
            config_file.write("\n")


def _flush_save() -> None:
    global _save_timer
    with _save_lock:
        _save_timer = None
    try:
        save_config()
    except Exception as exc:
        config_logger.Error(f"Failed to save config: {exc}")


def schedule_save() -> None:
    global _save_timer
    with _save_lock:
        if _save_timer is not None:
            _save_timer.cancel()
        _save_timer = threading.Timer(_SAVE_DEBOUNCE_S, _flush_save)
        _save_timer.daemon = True
        _save_timer.start()


def flush_save_now() -> None:
    global _save_timer
    with _save_lock:
        if _save_timer is not None:
            _save_timer.cancel()
            _save_timer = None
    save_config()


class Config:
    # Type 0: String, 1: Int, 2: Float
    def __init__(self, category: str, key: str, typev: int, value: str):
        global config_logger
        
        self.category = category
        self.key = key
        self.typev = typev

        representation = loaded_config.get(category, {}).get(key, {})
        stored_type = representation.get("type", -1)
        if stored_type == -1:
            config_logger.Warn("Couldn't find preference {category}.{key}")
            self.value = value
            self.save()
        elif stored_type != typev:
            config_logger.Warn(f"Type mismatch for {category}.{key}: {typev} != {representation.get('type', -1)}")
            self.value = value
            self.save()
        else:
            self.value = representation.get("value", value)

    def save(self) -> None:
        loaded_config[self.category] = loaded_config.get(self.category, {})
        loaded_config[self.category][self.key] = {"value": self.value, "type": self.typev}
        schedule_save()

    def valueString(self) -> str:
        return self.value
    def valueInt(self) -> int:
        global config_logger
        try:
            return int(self.value)
        except:
            config_logger.Error(f"Couldn't convert {self.category}.{self.key} to int")
            return 0
    def valueFloat(self) -> float:
        global config_logger
        try:
            return float(self.value)
        except:
            config_logger.Error(f"Couldn't convert {self.category}.{self.key} to float")
            return 0.0
        
    def setFloat(self, value: float) -> None:
        global config_logger
        if self.typev != 2:
            config_logger.Warn(f"Type mismatch for {self.category}.{self.key}: {self.typev} != 2")
            return
        new_value = str(value)
        if self.value == new_value:
            return
        self.value = new_value
        self.save()

    def setInt(self, value: int) -> None:
        global config_logger
        if self.typev != 1:
            config_logger.Warn(f"Type mismatch for {self.category}.{self.key}: {self.typev} != 1")
            return
        new_value = str(value)
        if self.value == new_value:
            return
        self.value = new_value
        self.save()

    def __str__(self):
        return f"{self.category}.{self.key}: {self.value}"
    def __repr__(self):
        return str(self)

class ConfigCategory:
    def __init__(self, category: str):
        self.category = category

    def getStringConfig(self, key: str, value: str) -> Config:
        return Config(self.category, key, 0, value)
    def getIntConfig(self, key: str, value: int) -> Config:
        return Config(self.category, key, 1, str(value))
    def getFloatConfig(self, key: str, value: float) -> Config:
        return Config(self.category, key, 2, str(value))
