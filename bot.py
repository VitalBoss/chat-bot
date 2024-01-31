from text_clf import clf, bran, big_model
import re

# clf - принимает сообщение, выдает 0 в случае, если текст не по теме, 1 если по теме
# bran - принимает сообщение,выдает 1 в случае, если есть токсичность, 0 в противном
# big_model - принимает сообщение, выдает сообщение

if __name__ == "__main__":
    '''
    Примерная логика работы бота: если сообщение токсичное, то посылаем нахуй
    Если сообщение токсичное, шлем туда же
    Если же оно не токсичное и по теме, то получаем ответ от модели 
    '''
    #s = "Ты чмо"
    #s = "Какие конкретные проекты по строительству и реконструкции дорог предусмотрены в ближайших планах для нашего района?"
    s = "Каким образом решаются вопросы утилизации отходов и поддержания чистоты на улицах?"
    #s = 'По чем сейчас бананы в африке'
    if bran(s):
        print("Ваше сообщение токсичное.")
    else:
        if clf(s):
            answer = big_model(s)
            if "!" in answer:
                print(answer[:1 + answer.find(".")])
            else:
                k = answer.find(".", answer.find(".") + 1)
                print(answer[:k + 1])
        else:
            print("Ваше сообщение не по теме.")