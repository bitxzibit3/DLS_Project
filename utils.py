import telebot
from telebot import types
import requests
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

path = {'style': './resources/styles/',
        'content': './resources/content/',
        'result': './resources/results/'}

imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # scale imported image
    transforms.ToTensor()])

unloader = transforms.ToPILImage()  # reconvert into PIL image


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def tensorToImage(tensor):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    return image


def showTensor(tensor):
    image = tensorToImage(tensor)
    plt.imshow(image)


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
cnn = torch.load('./vgg19_result.pth').to(device).eval()


content_layers = ['conv_2', 'conv_4']
style_layers = ['conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6']


def save_pic_and_gif(pics, path):
    # result pic saving
    pic = pics[-1]
    pic.save(path + '.jpg')
    pics[0].save(
        path + '.gif',
        save_all=True,
        append_images=pics[1:],  # Срез который игнорирует первый кадр.
        optimize=True,
        duration=150,
        loop=0)




token = '5928095979:AAHPlNvpCZRDXloDtJwXQxh7wI_WYb4ke2s'
styles = [elem[:-4] for elem in os.listdir(path['style'])]
sended_by_users = {}

bot = telebot.TeleBot(token, parse_mode=None)

helping_message = '''There is a bot to transfer styles from one your image to another!
To start, type \'start\' to me, and you will give instructions how do it!'''
