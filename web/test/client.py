import requests

if __name__ == '__main__':
    data_test = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3.0, 2.0, 1682.0, 1682.0, 184.0, 5.2, 1.834, 0.0, 1.0, 1.0]
    # Отправляем запрос на сервер с набором данных
    r = requests.post('http://localhost:5000/predict', json=data_test)
    
    print('Статус сервера:', r.status_code)
    
    # Получаем предсказание стоимости недвижимости
    if r.status_code == 200:
        print('Ответ сервера - предсказание:', r.json()['prediction'])
    else:
        print('Ответ сервера:', r.text)