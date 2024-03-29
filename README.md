# Telegram - бот, реализующий перенос стилей с одного изображения на другое

## Contents:

  * [Intro](#intro)

  * [Telegram-Bot](#tg-bot)
  
  * [Neural Net](#nn)
    * [Layers description](#layers-description)
      * [Normalization](#normalization)
      * [Content Loss Layer](#content-loss-layer)
      * [Style Loss Layer](#style-loss-layer)
    * [Architecture](#architecture)
  
  * [Examples](#examples)
  
## Intro
В данном проекте реализовался Telegram - бот, который позволял бы своим пользователям на основе двух изображений: изображения стиля и изображения контента получать картинку, на котором стиль с первой картинки перенесен на вторую, и GIF - анимацию того, как исходная картинка преобразовалась.

При получения результата пользователю необходимо отправить боту две фотографии:
  * изображение со стилем может быть отправлено пользователем, а может быть выбрано из уже имеющихся в библиотеке бота стилей
  * изображение с контентом пользователь отправляет боту сам

Одни из изображений стилей, уже присутствующих в боте:



![Malevich](https://user-images.githubusercontent.com/71255336/215344498-8de4b457-3c46-4fbd-9526-c75415bfca7d.jpg)


![Van Gog](https://user-images.githubusercontent.com/71255336/215344502-7741d9d8-e931-48d7-a92f-f68fe65d77ba.jpg)


При получении двух фотографий бот обрабатывает их с помощью нейронной сети, и отправляет результат её работы обратно пользователю.
  
  
## Telegram-Bot

При написании телеграм - бота использовалась библиотека для языка Python [PyTelegramBotAPI](https://github.com/eternnoir/pyTelegramBotAPI/), которая очень хорошо адаптирована под TelegramAPI, и, по сути, является удобным инструментом для работы с Telegram. Установку данного модуля можно осуществить с помощью команды через pip:
```
$ pip install pyTelegramBotAPI
```
Сам экземпляр бота создается при помощи команды (где token - ваш TelegramAPI-токен):
```
bot = telebot.TeleBot(token, parse_mode=None)
```
После установки модуля были прописаны триггеры на такие команды, как ```/start``` и ```/help```, которые были реализованы при помощи указания соответствующих аргументов в декораторе функций:
```
@bot.message_handler(commands=['start'])
@bot.message_handler(commands=['help'])
```
Команда ```/start``` используется для начала работы с ботом. В реализации этой функции была также использована конструкция из TeleBot ```ReplyKeyboardMarkup```, которая реализует интеративные кнопки, показываемые в чате с ботом:

![image](https://user-images.githubusercontent.com/71255336/215343389-16a615a1-133c-46cc-8fc8-3a047d9661a6.png)

Далее, были использованы хэндлеры, которые также являются декораторами из TeleBot и используется для реагирования на поступающие от пользователя сообщения различных типов, будь то текст или изображения. Внутри хэндлеров реализована условная конструкция, позволяющая один и тот же хэндлер в зависимости от полученного сообщения выполнять различные действия.

После получения двух фотографий от пользователя бот скачивает и сохраняет их в локальной директории, чтобы затем передать их в нейронную сеть для дальнейшей обработке.

## Neural Net

Для удобной работы с нейронными сетями были использованы следующие модули: [pytorch](https://pytorch.org/docs/stable/index.html), [torchvision](https://pytorch.org/vision/stable/), [PIL](https://pypi.org/project/Pillow/), [matplotlib](https://matplotlib.org/)
В качестве основы для нейронной сети была взята предобученная на датасете ImageNet сверточная сеть VGG19, взятая из модуля torchvision.models.
При попадании изображений в сеть происходит предобработка: они обрезаются и приводятся к тензорам.
Далее, можно ознакомиться со слоями, которые использовались при написании архитектуры сети:

### Layer description

### $\cdot$ Normalization

Даннный слой был использован лишь для одной цели: нормировка входных данных в соответствии с нормировачными данными сети VGG19

#### $\cdot$ ContentLoss

Данный слой использовался для подсчета так называемого content-loss, который вычисляется по методу Mean Square Error (MSE). Данный тип лосса показыает, насколько результат работы сети отличается от исходного изображения контента.

#### $\cdot$ StyleLoss

Данный слой по аналогии с предыдущим подсчитывает style-loss, который вычисляется по методу Mean Square Error (MSE), но применненого к матрице Грамма получаемых feature maps. Данный тип лосса показыает, насколько стиль результата работы сети рознится от необходимого нам стиля.

### Architecture

Как уже было сказано выше, за основу сети была взята предобученная на ImageNet модель VGG19.

Для каждого ContentLoss - слоя и StyleLoss - слоя было обозначено место для вставки, и в процессе формирования итоговой сети между слоями VGG19 вставлялись соответствующие loss - слои.

Таким образом, была получена следующая архитектура:

![image](https://user-images.githubusercontent.com/71255336/215352563-c0987b57-69f9-48fe-b2b5-d94652d32f2b.png)

Модель с данной архитектурой, а также все вспомогательные элементы были обернуты в класс ```NSTnet```. Также в нем был реализован метод ```transfer_style```, который и реализовал работу модели с конкретными наборами данных.

В процессе выполнения этого метода исходное изображение(а точнее, его копия) изменялась так, чтоб значение итогового лосса, который включал себя Style-loss и Content-Loss с соответствующими весами оказалось минимальной. При этом все остальное в сети было постоянно, то есть никакие веса не обновлялись и другие изображения не изменялись. На каждой десятой итерации (а всего их по умолчанию 300) сохранялось текущее состояние, чтоб в дальнейшем можно было получить GIF - анимацию процесса. Таким образом, на выходе сети мы получали список изображений, который превращался в GIF - анимацию.

## Examples
В качестве примеров хотел бы показать вам пару результатов обработки сетью:


![to1](https://user-images.githubusercontent.com/71255336/215353661-faf2b9fe-121c-4b36-95fe-02c4ca1fe331.gif)
![to2](https://user-images.githubusercontent.com/71255336/215353663-23b40b08-c74a-4296-8468-5c2760d8ebfd.gif)
![to3](https://user-images.githubusercontent.com/71255336/215353742-d9e91389-fbe1-4c29-9718-a55ad57789dd.gif)

