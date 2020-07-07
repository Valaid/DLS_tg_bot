from enum import Enum
from aiogram.dispatcher.filters.state import State, StatesGroup

TOKEN = '1239115079:AAHB2lI3iB2K23fYVC9iZTtRp1XFCp0IWUE'
# db_file = "database.vdb"
# class States(Enum):
#     """
#     Мы используем БД Vedis, в которой хранимые значения всегда строки,
#     поэтому и тут будем использовать тоже строки (str)
#     """
#     S_START = "0"  # Начало нового диалога
#     S_STYLE = "1"
#     S_CONTENT = "2"

class States(StatesGroup):
    S_START = State() # Начало нового диалога
    S_STYLE = State()
    S_CONTENT = State()