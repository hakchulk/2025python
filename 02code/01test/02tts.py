from deep_translator import GoogleTranslator
from gtts import gTTS
import pygame
import time

# 번역
translated = GoogleTranslator(source='auto', target='en').translate('좋은 하루네요')
# print(result) 

# TTS
tts = gTTS(text=translated, lang='en')
tts.save('voice.mp3')


# 재생
import os
# pygame 임포트 전에 이 코드가 반드시 먼저 실행되어야 합니다.
os.environ['SDL_AUDIODRIVER'] = 'dummy'

pygame.mixer.init()
pygame.mixer.music.load('voice.mp3')
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    time.sleep(0.1)

pygame.mixer.music.unload()