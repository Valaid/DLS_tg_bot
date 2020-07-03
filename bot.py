# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 22:21:15 2020

@author: Dns1
"""

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher


from aiogram.utils.helper import Helper, HelperMode, ListItem
from PIL import Image
from vedis import Vedis
from config import States, TOKEN

db_file = "database.vdb"


TOKEN = '1239115079:AAHB2lI3iB2K23fYVC9iZTtRp1XFCp0IWUE'
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
image1 = False
image2 = False

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
  set_state(message.from_user.id, str(States.S_START.value))
  await message.reply("Привет!\nЯ умею переносить стиль на фотографии, давай поиграем!")
    
@dp.message_handler(state = TestStates.all(), commands=['help'])
async def process_help_command(message: types.Message):
    await message.reply("Напиши мне что-нибудь, и я отправлю этот текст тебе в ответ!")

@dp.message_handler(func=lambda message: get_current_state(message.chat.id) == States.S_START.value, commands=['start_style_transfer'])    
async def process_start_style_command(message: types.Message):
    set_state(message.from_user.id, States.S_STYLE.value)
    await message.reply("Отправь мне картинку стиля")

@dp.message_handler(func=lambda message: get_current_state(message.chat.id) == States.S_STYLE.value, content_types=types.ContentType.PHOTO)    
async def get_style(message: types.Message):
    set_state(message.from_user.id, States.S_CONTENT.value)
    global image1
    image1 = await bot.download_file_by_id(message.photo[-1].file_id)
    image1 = Image.open(image1)
    # await message.photo[-1].download('data/%s/style.jpg' %
    #                               (message.from_user.id))
    await message.reply('Отлично, со стилем определились.\n Теперь отправь мне изображение, стиль которого ты хочешь изменить')

@dp.message_handler(func=lambda message: get_current_state(message.chat.id) == States.S_CONTENT.value, content_types=types.ContentType.PHOTO)    
async def get_style(message: types.Message):
    image2 = await bot.download_file_by_id(message.photo[-1].file_id)
    image2 = Image.open(image2)
    image3 = image2
    await message.reply('Окейсики, теперь подожди немного')
    model = StyleTransferModel(device,cnn,cnn_normalization_mean, cnn_normalization_std, image1, image2)
    img = image3
    img = model(img)
    await bot.send_photo(message.from_user_id,img)
    set_state(message.from_user.id, States.S_START.value)

async def set_dir(message, type_path):
    path = 'data/%s/%s' % (message.from_user.id, type_path)
    if not os.path.exists(path):
        os.makedirs(path)
    
@dp.message_handler()
async def echo_message(msg: types.Message):
    await bot.send_message(msg.from_user.id, msg.text)
    
if __name__ == '__main__':
    executor.start_polling(dp)