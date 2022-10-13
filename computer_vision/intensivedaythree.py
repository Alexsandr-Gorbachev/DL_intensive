from PIL import Image
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Embedding, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from IPython import display as ipd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from collections import Counter
from tensorflow.keras import utils

from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns 

import dataset, модель, segmentation,  check_for_errors, traide, home
import gdown
import zipfile
import os
import random
import time 

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class AccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        self.times = []


    def plot_graph(self):        
        plt.figure(figsize=(20, 14))
        plt.subplot(2, 2, 1)
        plt.title('Точность', fontweight='bold')
        plt.plot(self.train_acc, label='Точность на обучащей выборке')
        plt.plot(self.val_acc, label='Точность на проверочной выборке')
        plt.xlabel('Эпоха обучения')
        plt.ylabel('Доля верных ответов')
        plt.legend()        
        plt.show()
       

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.train_acc.append(logs['accuracy'])
        self.val_acc.append(logs['val_accuracy'])
        self.train_loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])
        t = round(time.time() - self.start_time, 1)
        self.times.append(t)
        if logs['val_accuracy'] > self.accuracymax:
            self.accuracymax = logs['val_accuracy']
            self.idxmax = epoch
        print(f'Эпоха {epoch+1}'.ljust(10)+ f'Время обучения: {t}c'.ljust(25) + f'Точность на обучающей выборке: {bcolors.OKBLUE}{round(logs["accuracy"]*100,2)}%{bcolors.ENDC}'.ljust(50) +f'Точность на проверочной выборке: {bcolors.OKBLUE}{round(logs["val_accuracy"]*100,2)}%{bcolors.ENDC}')
        self.cntepochs += 1

    def on_train_begin(self, logs):
        self.idxmax = 0
        self.accuracymax = 0
        self.cntepochs = 0

    def on_train_end(self, logs):
        ipd.clear_output(wait=True)
        for i in range(self.cntepochs):
            if i == self.idxmax:
                print('\33[102m' + f'Эпоха {i+1}'.ljust(10)+ f'Время обучения: {self.times[i]}c'.ljust(25) + f'Точность на обучающей выборке: {round(self.train_acc[i]*100,2)}%'.ljust(41) +f'Точность на проверочной выборке: {round(self.val_acc[i]*100,2)}%'+ '\033[0m')
            else:
                print(f'Эпоха {i+1}'.ljust(10)+ f'Время обучения: {self.times[i]}c'.ljust(25) + f'Точность на обучающей выборке: {bcolors.OKBLUE}{round(self.train_acc[i]*100,2)}%{bcolors.ENDC}'.ljust(50) +f'Точность на проверочной выборке: {bcolors.OKBLUE}{round(self.val_acc[i]*100,2)}%{bcolors.ENDC}' )
        self.plot_graph()

class TerraDataset:
    train_test_ratio = 0.1
    bases = {
        'Трейдинг' : {
            'url': 'https://storage.yandexcloud.net/aiueducation/Intensive/shares.zip',
            'info': 'Вы скачали базу с данными по акциям трех российских компаний: Яндекс, Газпром и полиметаллы',
            'dir_name': 'трейдинг',
            'task_type': 'traiding',            
        },
        'Самолеты' : {
            'url': 'https://storage.yandexcloud.net/aiueducation/Intensive/airplane.zip',
            'info': 'Вы скачали базу сегментированных изображений самолетов. База содержит 981 изображение',
            'dir_name': 'самолеты',
            'task_type': 'air_segmentation',
        },
        'Губы' : {
            'url': 'https://storage.yandexcloud.net/aiueducation/Intensive/lips.zip',
            'info': 'Вы скачали базу сегментированных изображений губ. База содержит 1000 изображение',
            'dir_name': 'губы',
            'task_type': 'air_segmentation',
        },
        'Самолеты_макс' : {
            'url': 'https://storage.yandexcloud.net/aiueducation/Intensive/airplane.zip',
            'info': 'Вы скачали базу сегментированных изображений самолетов. База содержит 981 изображение',
            'dir_name': 'самолеты',
            'task_type': 'air_max_segmentation',
        },
        'Люди' : {
            'url': 'https://storage.yandexcloud.net/aiueducation/Intensive/segment_people.zip',
            'info': 'Вы скачали базу сегментированных изображений людей. База содержит 1500 изображение',
            'dir_name': 'люди',
            'task_type': 'man_segmentation',            
        },
        'Умный_дом' : {
            'url': 'https://storage.yandexcloud.net/aiueducation/Intensive/cHome.zip',
            'info': 'Вы скачали базу аудиоданных с командами для умного дома. База содержит 4 класса: «Кондиционер», «Свет», «Телевизор», «Фоновая речь»',
            'dir_name': 'умный_дом',
            'task_type': 'speech_recognition',  
            'classes': ['Кондиционер', 'Свет', 'Телевизор', 'Фон']
        },
    }
    def __init__(self, name):
        '''
        parameters:
            name - название датасета
        '''        
        self.base = self.bases[name]
        self.sets = None
        self.classes = None

    def load(self):
        '''
        функция загрузки датасета
        '''
        
        print(f'{bcolors.BOLD}Загрузка датасета{bcolors.ENDC}',end=' ')
        
        # Загурзка датасета из облака
        fname = gdown.download(self.base['url'], None, quiet=True)

        if Path(fname).suffix == '.zip':
            # Распаковка архива
            with zipfile.ZipFile(fname, 'r') as zip_ref:
                zip_ref.extractall(self.base['dir_name'])

            # Удаление архива
            os.remove(fname)

        # Вывод информационного блока
        print(f'{bcolors.OKGREEN}Ok{bcolors.ENDC}')
        print(f'{bcolors.OKBLUE}Ифно:{bcolors.ENDC}')
        print(f'    {self.base["info"]}')
        return self.base['task_type']

    def samples(self, **kwargs):
        '''
        Функция визуализации примеров
        '''
        # Визуализация датасета изображений для задачи классификации
        if self.base['task_type'] == 'traiding':              
              if 'begin' in kwargs and 'end' in kwargs:
                  dataset.показать_примеры(путь="трейдинг", начало=kwargs['begin'], конец=kwargs['end'])
              else:
                  dataset.показать_примеры(путь="трейдинг")

        if self.base['task_type'] == 'air_segmentation' or self.base['task_type'] == 'air_max_segmentation' or self.base['task_type'] == 'man_segmentation': 
              dataset.показать_примеры_сегментации(
                  оригиналы = f'{self.base["dir_name"]}/original/',
                  сегментированные_изображения = f'{self.base["dir_name"]}/segment/'
                  )
              
        if self.base['task_type'] == 'speech_recognition':
          if 'category' in kwargs:
              dataset.показать_примеры(путь="умный_дом", файл = kwargs['category'])
          else:
              dataset.показать_примеры(путь="умный_дом")
    
    def create_sets(self, *params):
        if self.base['task_type'] == 'traiding':
            print(f'{bcolors.BOLD}Формирование выборок:{bcolors.ENDC}', end=' ')
            kwargs =  {'путь':'трейдинг', 
                       'акции':params[0],
                       'количество_анализируемых_дней':params[1],
                       'период_предсказания':params[2],
                       }
            self.sets = dataset.создать_выборки(**kwargs)
            print(f'{bcolors.OKGREEN}Ok{bcolors.ENDC}')
            
        if self.base['task_type'] == 'air_segmentation':          
            self.sets = dataset.создать_выборки(путь=self.base["dir_name"])
            print()
            print(f'Размер созданных выборок:')
            print(f'  Обучающая выборка: {self.sets[0][0].shape}')
            print(f'  Метки обучающей выборки: {self.sets[0][1].shape}')
            print(f'  Проверочная выборка: {self.sets[1][0].shape}')
            print(f'  Метки проверочной выборки: {self.sets[1][1].shape}')
            print()

        if self.base['task_type'] == 'man_segmentation':          
            self.sets = dataset.создать_выборки(путь='люди')
            print()
            print(f'Размер созданных выборок:')
            print(f'  Обучающая выборка: {self.sets[0][0].shape}')
            print(f'  Метки обучающей выборки: {self.sets[0][1].shape}')
            print(f'  Проверочная выборка: {self.sets[1][0].shape}')
            print(f'  Метки проверочной выборки: {self.sets[1][1].shape}')
            print()

        if self.base['task_type'] == 'air_max_segmentation':          
            self.sets = dataset.создать_выборки(путь='самолеты', размер=[160, 320])
            print()
            print(f'Размер созданных выборок:')
            print(f'  Обучающая выборка: {self.sets[0][0].shape}')
            print(f'  Метки обучающей выборки: {self.sets[0][1].shape}')
            print(f'  Проверочная выборка: {self.sets[1][0].shape}')
            print(f'  Метки проверочной выборки: {self.sets[1][1].shape}')
            print()

        if self.base['task_type'] == 'speech_recognition':          
            self.sets = dataset.создать_выборки(путь='умный_дом', длина=params[0], шаг=params[1])
            print()
            print(f'Размер созданных выборок:')
            print(f'  Обучающая выборка: {self.sets[0][0].shape}')
            print(f'  Метки обучающей выборки: {self.sets[0][1].shape}')
            print(f'  Проверочная выборка: {self.sets[1][0].shape}')
            print(f'  Метки проверочной выборки: {self.sets[1][1].shape}')
            print(f'Распределение по классам:')
            f, ax =plt.subplots(1,2, figsize=(16, 5))            
            ax[0].bar(np.array(self.base['classes'])[list(Counter(list(np.argmax(self.sets[0][1], axis=1))).keys())], Counter(list(np.argmax(self.sets[0][1], axis=1))).values())
            ax[0].set_title('Обучающая выборка')
            ax[1].bar(np.array(self.base['classes'])[list(Counter(list(np.argmax(self.sets[1][1], axis=1))).keys())], Counter(list(np.argmax(self.sets[1][1], axis=1))).values(), color='g')
            ax[1].set_title('Проверочная выборка')
            plt.show()
            print()
    
class TerraModel:    
    def __init__(self, task_type, trds):
        self.model = None
        self.task_type = task_type
        self.trds = trds

    def create_model(self, layers, **kwargs):
      if self.trds.base['task_type'] == 'traiding':      
          self.model = модель.создать_сеть(layers, self.trds.sets[0][0].shape[1:], параметры_модели=None, задача = 'временной ряд')
      if self.trds.base['task_type'] == 'speech_recognition':      
          self.model = модель.создать_сеть(layers, self.trds.sets[0][0].shape[1:], параметры_модели=None, задача = 'аудио')
      if self.trds.base['task_type'] == 'air_segmentation' or self.trds.base['task_type'] == 'air_max_segmentation' or self.trds.base['task_type'] == 'man_segmentation':
          if 'type_model' in kwargs:
             if kwargs['type_model'] == 'PSP':
               self.model = модель.создать_PSP(
                   входной_размер = self.trds.sets[0][0].shape[1:],
                   количество_блоков = kwargs['count_block'],
                   стартовый_блок = kwargs['start_block'],
                   блок_PSP = kwargs['PSP_block'],
                   финальный_блок = kwargs['finally_block'],
               )
             if kwargs['type_model'] == 'U-net':
               self.model = модель.создать_UNET(
                   входной_размер = self.trds.sets[0][0].shape[1:],
                   блоки_вниз = kwargs['down_blcoks'],
                   блок_внизу = kwargs['bottom_block'],
                   блоки_вверх = kwargs['up_blocks'],
               )
          else:
            self.model = модель.создать_сеть(layers, self.trds.sets[0][0].shape[1:], параметры_модели=None, задача='сегментация изображений')

      

    def train_model(self, epochs, use_callback = True, **kwargs):
      if self.trds.base['task_type'] == 'traiding':
        модель.обучение_модели(self.model, self.trds.sets[0][0], self.trds.sets[0][1], self.trds.sets[1][0], self.trds.sets[1][1], 64, epochs, 0.2, **kwargs)
      if self.trds.base['task_type'] == 'speech_recognition':
        модель.обучение_модели(self.model, self.trds.sets[0][0], self.trds.sets[0][1], self.trds.sets[1][0], self.trds.sets[1][1], 128, epochs, 0.2, **kwargs)
      if self.trds.base['task_type'] == 'air_segmentation' or self.trds.base['task_type'] == 'air_max_segmentation' or self.trds.base['task_type'] == 'man_segmentation':
        модель.обучение_модели(self.model, self.trds.sets[0][0], self.trds.sets[0][1], self.trds.sets[1][0], self.trds.sets[1][1], 32, epochs, 0.2)
      if self.trds.base['task_type'] == 'hr_regression':            
          self.model.compile(loss='sparse_categorical_crossentropy', optimizer = Adam(0.0001), metrics=['accuracy'])
          accuracy_callback = AccuracyCallback()
          callbacks = []
          if use_callback:
              callbacks = [accuracy_callback]
          y_train = np.argmax(self.trds.sets[0][1], axis=1)
          y_test = np.argmax(self.trds.sets[1][1], axis=1)
          history = self.model.fit(self.trds.sets[0][0], y_train,
                        batch_size = self.trds.sets[0][0].shape[0]//25,
                        validation_data=(self.trds.sets[1][0], y_test),
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose = 0)
          return history
          
    def test_model(self, *params):
      if self.trds.base['task_type'] == 'traiding':
        traide.model_test(
            self.model,
            self.trds.sets[1][0], self.trds.sets[1][1],
            params[0],
            params[1])
        traide.example_traid(self.model, params[0], params[1])        
      if self.trds.base['task_type'] == 'air_segmentation' or self.trds.base['task_type'] == 'air_max_segmentation' or self.trds.base['task_type'] == 'man_segmentation':
        segmentation.тест_модели(self.model, self.trds.sets[1][0])
      if self.trds.base['task_type'] == 'speech_recognition':
        home.тест_модели(self.model, params[0], params[1], params[2])
      if self.trds.base['task_type'] == 'hr_regression':
        print(f'{bcolors.BOLD}Тестирование модели на случайном примере тестовой выборки: {bcolors.ENDC}')
        print()
        модель.тест_модели_вакансии(self.model, self.trds.sets[1][0], self.trds.sets[1][1])


class TerraIntensive:
    def __init__(self):
       self.trds = None
       self.trmodel = None
       self.task_type = None

    def load_dataset(self, ds_name):
        self.trds = TerraDataset(ds_name)
        self.task_type = self.trds.load()

    def samples(self, **kwargs):
        self.trds.samples(**kwargs)

    def dataset_info(self):
        self.trds.dataset_info()

    def create_sets(self, *params):
        self.trds.create_sets(*params)

    def create_model(self, layers=None, **kwargs):
        print(f'{bcolors.BOLD}Создание модели нейронной сети{bcolors.ENDC}', end=' ')
        self.trmodel = TerraModel(self.task_type, self.trds)
        self.trmodel.create_model(layers, **kwargs)
        print(f'{bcolors.OKGREEN}Ok{bcolors.ENDC}')

    def create_model_combine(self, *branches):
        print(f'{bcolors.BOLD}Создание комбинированной модели нейронной сети{bcolors.ENDC}', end=' ')
        self.trmodel = TerraModel(self.task_type, self.trds, 'Combined')
        self.trmodel.create_model_combine(*[b+'-linear' for b in branches])
        print(f'{bcolors.OKGREEN}Ok{bcolors.ENDC}')

    def train_model(self, epochs):
        self.trmodel.train_model(epochs)

    def test_model(self, *data):
        self.trmodel.test_model(*data)        

    def traiding(self):
      if self.trmodel.trds.base['task_type'] == 'traiding':
          print(f'{bcolors.BOLD}Производится симуляция торговой сессии{bcolors.ENDC}')
          traide.traiding(self.trmodel.model, self.trmodel.trds.sets[1][0], 'результат')