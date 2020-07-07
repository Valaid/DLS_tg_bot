# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 22:21:15 2020

@author: Dns1
"""

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
import torch
import torchvision.models as models
from aiogram.utils import executor
from aiogram.utils.helper import Helper, HelperMode, ListItem
from PIL import Image
from io import BytesIO
from aiogram.types import ReplyKeyboardRemove, ReplyKeyboardMarkup, KeyboardButton
# from helped_classes import set_state, get_current_state
from Style_transfer_model import StyleTransferModel
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from config import States

# db_file = "database.vdb"


TOKEN = '1239115079:AAHB2lI3iB2K23fYVC9iZTtRp1XFCp0IWUE'
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())
image1 = False
image2 = False
bio = BytesIO()
bio.name = 'image.jpeg'
button_hi = KeyboardButton('Давай!')
greet_kb = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).add(button_hi)


@dp.message_handler(commands=['start','reset'], state='*')
async def process_start_command(message: types.Message):
  # set_state(message.from_user.id, str(States.S_START.value))
  await States.S_START.set()
  await message.reply("Привет!\nЯ умею переносить стиль на фотографии, давай поиграем!", 
                      reply_markup=greet_kb)
  
@dp.message_handler(text = "Давай!", state = States.S_START)
async def process_start_style_button_command(message: types.Message, state: FSMContext):
  # set_state(message.from_user.id, str(States.S_START.value))
  await States.next()
  await message.reply("Отправь мне картинку стиля")  
  
@dp.message_handler(commands=['help'],state = '*')
async def process_help_command(message: types.Message):
    await message.reply("/reset - начать сначала\n/start_style_transfer - запустить основную функцию бота (перенос стиля) \n/help - вызвать справку")
    

# @dp.message_handler(lambda message: get_current_state(message.chat.id) == States.S_START.value, commands=['start_style_transfer'])
@dp.message_handler(state = States.S_START, commands=['start_style_transfer'])
async def process_start_style_command(message: types.Message, state: FSMContext):
    # set_state(message.from_user.id, States.S_STYLE.value)
    await States.next()
    await message.reply("Отправь мне картинку стиля")

# @dp.message_handler(lambda message: get_current_state(message.chat.id) == States.S_STYLE.value, content_types=types.ContentType.PHOTO)    
@dp.message_handler(state = States.S_STYLE, content_types=types.ContentType.PHOTO)
async def get_style(message: types.Message, state: FSMContext):
    # set_state(message.from_user.id, States.S_CONTENT.value)
    global image1
    image1 = await bot.download_file_by_id(message.photo[-1].file_id)
    image1 = Image.open(image1)
    # await message.photo[-1].download('data/%s/style.jpg' %
    #                               (message.from_user.id))
    await States.next()
    await message.reply('Отлично, со стилем определились.\nТеперь отправь мне изображение, стиль которого ты хочешь изменить')

# @dp.message_handler(lambda message: get_current_state(message.chat.id) == States.S_CONTENT.value, content_types=types.ContentType.PHOTO)
@dp.message_handler(state = States.S_CONTENT, content_types=types.ContentType.PHOTO)  
async def get_context(message: types.Message, state: FSMContext):
    image2 = await bot.download_file_by_id(message.photo[-1].file_id)
    image2 = Image.open(image2)
    image3 = image2
    await message.reply('Окейсики, теперь подожди немного')
    model = StyleTransferModel(device,cnn,cnn_normalization_mean, cnn_normalization_std, image1, image2)
    img = image3
    unloader = transforms.ToPILImage()
    img = model(img)
    img2 = unloader(img.squeeze(0))
    bio = BytesIO()
    bio.name = 'image.jpeg'
    img2.save(bio, 'JPEG')
    bio.seek(0)
    await bot.send_photo(message.from_user.id,photo = bio)
    await message.reply('Вот такие пироги, если хочешь попробовать с другими изображениями нажми кнопку "Давай!"', reply_markup=greet_kb)
    await States.S_START.set()

    
@dp.message_handler()
async def echo_message(msg: types.Message):
    await bot.send_message(msg.from_user.id, msg.text)
    
if __name__ == '__main__':
    executor.start_polling(dp)