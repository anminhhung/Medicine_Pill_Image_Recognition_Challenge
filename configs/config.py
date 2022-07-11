from configparser import ConfigParser
import os 

def init_config(config_path=None) -> ConfigParser:
    config = ConfigParser()

    config_path = 'configs/config.ini'

    config.read(config_path)
    return 