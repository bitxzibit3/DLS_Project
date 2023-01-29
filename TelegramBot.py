from modelsettings import *


@bot.message_handler(commands=['start'])
def handle_start(message):
    styles = [elem[:-4] for elem in os.listdir(path['style'])]
    user_id = message.from_user.id
    sended_by_users[user_id] = {}
    sended_by_users[user_id]['style'] = False
    sended_by_users[user_id]['content'] = False
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    btn1 = types.KeyboardButton('Choose existed style')
    btn2 = types.KeyboardButton('Send own style picture')
    markup.add(btn1, btn2)
    bot.reply_to(message,
                 'You have to send two photos: with content and style\n \
Send a style photo firstly\n \
You can choose existed styles or send me yours: what do you want?',
                 reply_markup=markup)


@bot.message_handler(content_types=['text'])
def text_handler(message):
    if message.text == 'Choose existed style':
        show_existed_styles(message)
    elif message.text in styles:
        style_name = message.text
        src = path['style'] + style_name + '.jpg'
        sended_by_users[message.from_user.id]['style'] = src
        bot.send_photo(message.from_user.id, open(src, 'rb'), caption=f'Your style: {style_name}')
        get_content(message)
    elif message.text == 'Send own style picture':
        bot.send_message(message.chat.id, 'Send your style photo and add caption \'style\'!')
        get_content(message)
    else:
        pass


def get_content(message):
    bot.send_message(message.chat.id, 'Send your content photo and add caption \'content\'')


def show_existed_styles(message):
    lst = os.listdir(path['style'])
    styles_to_send = []
    styles_message = ''
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    for i, f in enumerate(lst):
        style_path = path['style'] + '/' + f
        if i == 0:
            styles_to_send.append(telebot.types.InputMediaPhoto(open(style_path, 'rb'),
                                                                caption=styles_message))
            styles_message = f'{i + 1}. {f[:-4]}'
        else:
            styles_to_send.append(telebot.types.InputMediaPhoto(open(style_path, 'rb')))
            styles_message = styles_message + "\n" + f'{i + 1}. {f[:-4]}'
        markup.add(types.KeyboardButton(f[:-4]))
    styles_to_send[0].caption = styles_message
    bot.send_media_group(message.from_user.id, styles_to_send)
    bot.send_message(message.from_user.id, 'Choose one of the styles:', reply_markup=markup)


@bot.message_handler(content_types=['photo'])
def take_photo(message):
    try:
        caption = message.caption.lower()
        if caption not in ('content', 'style'):
            bot.send_message('Wrong photo, try again!')
        user_id = message.from_user.id
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        src = path[caption] + 'User' + caption.capitalize()
        src += ('_' + str(user_id) + '.jpg')
        file = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(token, file_info.file_path))
        with open(src, 'wb') as f:
            f.write(file.content)
        sended_by_users[message.from_user.id][caption] = src
    except:
        bot.send_message(message.chat.id, 'Something wents wrong!')
    src = {'style': sended_by_users[message.from_user.id]['style'],
           'content': sended_by_users[message.from_user.id]['content']}

    if src['style'] and src['content']:
        bot.send_message(user_id, 'In process...')
        res = run_transfer(bot, user_id, src['style'], src['content'])
        res_path = path['result'] + str(user_id)
        save_pic_and_gif(res, res_path)
        gif = open(res_path + '.gif', 'rb')
        photo = open(res_path + '.jpg', 'rb')
        bot.send_video(user_id, gif, caption='This gif shows process of transformation!')
        gif.close()
        bot.send_photo(user_id, photo, caption='This photo is the result of transformation!')
        photo.close()
        # Так как фото контента 100% присылается пользователям, будем его удалять
        os.remove(src['content'])
        # так как фото стиля может быть выбранным из "библиотеки", а может быть отправлено пользователем
        # будем удалять стиль только в том случае, если он загружен пользователем
        if src['style'].startswith('UserStyle'):
            os.remove(src['style'])

@bot.message_handler(commands=['help'])
def handle_help(message):
    bot.reply_to(message, helping_message)


bot.infinity_polling()
