
from bot.main_menu import main_menu
from bot.bot_instance import bot

def info(message):
    bot.reply_to(message, "FinanceBot создается как студенческий проект. Наша цель это разработка Telegram бота на Python и работа с нейросетью LSTM.\
Принцип работы: введите тикер компании например ROSB или TCSG, далее по тикеру выгружаются архивные данные по цене акций за последние 5 лет со свечой 24 часа, после чего \
при помощи нейросети, мы предсказываем стоимость акций за последний год. ") 
    main_menu(message)