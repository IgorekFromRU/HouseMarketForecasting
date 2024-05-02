import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy.special import inv_boxcox

from scipy.stats import shapiro

#============================================
# Случайные цвета RGB
def rand_rgb():
    return (np.random.random(), np.random.random(), np.random.random())


#=============================================
# Вычисление интеркватильного размаха методом Тьюки    
# Функция IQR возвращает булеву маску, 
# где значение True указывает на то, что наблюдение не является выбросом, 
# а False указывает на то, что наблюдение может быть выбросом.
def IQR(data: pd.Series, quantile: float = 0.25) -> pd.Series:
    Q1 = data.quantile(quantile)
    Q3 = data.quantile(1 - quantile)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print('Число записей перед фильтрацией: ', len(data))
    mask_for_filtering = data.between(lower_bound,upper_bound)
    print(f'После применения фильтрации IQR останется {mask_for_filtering.sum()} записей')

    return mask_for_filtering


#============================================
# Поиск выбросов методом z-score (поиск по отклонению). 
# Возвращает маску очищенных от выбросов.
# Функция определения выбросов по методу z-отклонений
def outliers_z_score(data: pd.Series,
                     log_scale: bool = False,
                     left: int = 3,
                     right: int = 3) :

    # Если требуется переход к логарифмическому масштабу
    if log_scale:
        if data.min() > 0: # если значения признака больше 0
            x = np.log(data)
        else:
            x = np.log(data+1) # иначе добавляем единицу
    else:
        x = data

    # Вычисляем мат.ожидание и стандарт. отклонение
    mu = x.mean()
    sigma = x.std()
    lower_bound = mu - left * sigma
    upper_bound = mu + right * sigma

    mask_for_filtering = x.between(lower_bound,upper_bound)

    print('Число записей перед фильтрацией: ', len(x))
    print(f'После применения z_score останется {mask_for_filtering.sum()} записей')

    return mask_for_filtering

#=================================================
# Процент пропущенных значений для каждого столбца
def visual_missing_data(data: pd.DataFrame=None):
    if not isinstance(data, pd.DataFrame):
        print('Data must be a DataFrame')
        return
    if data.empty:
        print('Data is empty')
        return
    missing_percent_sorted = (
        ((data.isnull().sum() / len(data)) * 100)
        .sort_values(ascending=True)
        .loc[lambda x: x > 0]
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_percent_sorted, y=missing_percent_sorted.index, color='skyblue')
    plt.xlabel('Процент пропущенных значений')
    plt.ylabel('Признаки')
    plt.title('Процент пропущенных значений по столбцам')

    # Добавление подписей к барам
    for index, value in enumerate(missing_percent_sorted):
        plt.text(value, index, f'{value:.2f}', ha='left', va='center')
    plt.show()

#==============================================
# Построение гистограммы и 'ящика с усами'
def visual_hist_box(data: pd.Series=None, c_name: str='', x_label: str='', y_label: str=''):
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Гистограмма
    sns.histplot(data, ax=axes[0], kde=True)
    axes[0].set_title(f'Гистограмма -{c_name}')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)

    # Boxplot
    sns.boxplot(x=data, ax=axes[1], color='skyblue')
                                # data.name
    axes[1].set_title(f'Boxplot -{c_name}')
    axes[1].set_xlabel(x_label)

    plt.tight_layout()
    plt.show()

#==============================================
# Построение гистограммы и 'ящика с усами'
def visual_hist_box_2(data: pd.Series=None):
    # два подграфика ax_box и ax_hist
    # кроме того, укажем, что нам нужны:
    _, (ax_box, ax_hist) = plt.subplots(2, # две строки в сетке подграфиков,
                                    sharex = True, # единая шкала по оси x и
                                    gridspec_kw = {'height_ratios': (.15, .85)}) # пропорция 15/85 по высоте
 
    sns.boxplot(x = data, ax = ax_box, orient='h', medianprops={"color": "red", "linestyle": '--'})
    sns.histplot(x = data, ax = ax_hist, bins = 20, kde = True)
    ax_hist.axvline(data.mean(), color='red', linestyle='--', linewidth=0.8)
 
    ax_box.set(xlabel = '') # пустые кавычки удаляют подпись (!)
    ax_hist.set(xlabel = data.name)
    ax_hist.set(ylabel = 'count')
 
    plt.show()


#=========================================
# Сравнение двух гистограмм
def visual_compare_hist(data1: pd.Series=None, data2: pd.Series=None, rmk1='', rmk2=''):
    # сравним изначальное распределение и распределение после преобразования Бокса-Кокса
    _, (ax_hist1, ax_hist2) = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4))
 
    sns.histplot(x = data1, bins = 15, ax = ax_hist1)
    ax_hist1.set_title(rmk1)
 
    sns.histplot(x = data2, bins = 15, color = 'green', ax = ax_hist2)
    ax_hist2.set_title(rmk2)
 
    plt.tight_layout()
    plt.show()

    
#=============================================
# Короткая информация по таблице
def short_info(data: pd.DataFrame):
    print(f'Размер таблицы :{data.shape}')
    print(f'Число пустых значений: \n{data.isnull().sum()}')
    display(data.describe(include='all').T)


#============================================
# функционал - замена значений в серии, основываясь 
# на наличии определенного текста по словарю ключ- список значений для поиска.
def find_replace_by_contains_word(data: pd.Series, replace_dict):
    for key, values in tqdm(replace_dict.items(), desc="Replacing values"):
        for value in values:
            data.mask(data.str.contains(value, na=False), key,inplace=True)

#==========================================
# Извлечение числовых данных
def extract_numbers(x):
    if pd.isna(x):
        return None
    pattern = r'\b\d+(\.\d+)?\b'  # 111.11
    numbers = re.search(pattern, x)  # Находим группу чисел, не разделенных символами    
    return numbers.group(0) if numbers  else None


#======================================
# Статистические тесты на номальное распределение
# Шапиро-Уилка from scipy.stats import shapiro
# К-квадрат А'Агостино на основе ассиметрии и экцесса from scipy.stats import normaltest
# Андерсона-Дарлинга from scipy.stats import anderson
# Хи-квадрат scipy.stats import chisquare
# Лиллиэфорс statsmodels.stats.diagnostic import lilliefors
# Жарке-Бера ( для выборок > 2000) scipy.stats import jarque_bera
# Критерий согласия Колмогоров-Смирнова scipy.stats import kstest(data, 'norm')
#

alpha = 0.05 # уровень значимости    

def test_shapiro(data):
    stat, p_value = shapiro(data)
    print(f"Статистика теста Шапиро-Уилка: {stat}, p-value: {p_value}")
    if p_value > alpha:
        print('Принимаем гипотезу о нормальности(Gaussian)')
    else:
        print('Отклоняем гипотезу о нормальности(Gaussian)')
        
# функция для принятия решения об отклонении нулевой гипотезы
def decision_hypothesis(p):
    print(f'p-value = {p:.3f}')
    if p <= alpha:
        print(f'p-значение меньше, чем заданный уровень значимости {alpha:.2f}.',
              'Отвергаем нулевую гипотезу в пользу альтернативной.')
    else:
        print(f'p-значение больше, чем заданный уровень значимости {alpha:.2f}.',
              'У нас нет оснований отвергнуть нулевую гипотезу.')


#====================================================
# Преобразовании серии обратно по методу Бокса-Кокса
def inverse_boxcox_transform(y, y_pred, lambda_):
    y_inv = inv_boxcox(y, lambda_).round(0).astype(int)
    y_pred_inv = inv_boxcox(y_pred, lambda_).round(0).astype(int)
    nan_indices = np.isnan(y_pred_inv)
    if nan_indices.any():
        print(f'Количество NaN значений: {nan_indices.sum()}')
        return y_inv[~nan_indices], y_pred_inv[~nan_indices]
    return y_inv, y_pred_inv


#====================================================
# Метрика MAE, вход - Данные, трансформированные по методу Бокса-Кокса
def metric_mae_boxcox(y_train, y_train_pred, y_test, y_test_pred, lambda_):
    y_train, y_train_pred = inverse_boxcox_transform(y_train, y_train_pred, lambda_)
    y_test, y_test_pred = inverse_boxcox_transform(y_test, y_test_pred, lambda_)

    metric_function = mean_absolute_error 
    score_train = metric_function(y_train, y_train_pred).round(3)
    score_test = metric_function(y_test, y_test_pred).round(3)

    metric = 'MAE'
    width = 20
    print(f'{metric} -  train: {score_train:>{width}.3f}')
    print(f'{metric}  - test: {score_test:>{width}.3f}')
    return score_train, score_test

        
#===============================================
# Вывод набора метрик, вход - Данные, трансформированные по методу Бокса-Кокса
def show_metrics_boxcox(y_train, y_train_pred, y_test, y_test_pred, lambda_):
    
    y_train, y_train_pred = inverse_boxcox_transform(y_train, y_train_pred, lambda_)
    y_test, y_test_pred = inverse_boxcox_transform(y_test, y_test_pred, lambda_)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred) * 100
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    r2_train = r2_score(y_train, y_train_pred)

    mae_test = mean_absolute_error(y_test, y_test_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred) * 100
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_test = r2_score(y_test, y_test_pred)
    
    width = 20
    print(f'MAE -  train: {mae_train:>{width}.3f}') # ---> 0 - best
    print(f'MAPE - train: {mape_train:>{width}.3f}') # ---> 0 - best
    print(f'RMSE - train: {rmse_train:>{width}.3f}') # ---> 0 - best
    print(f'R_2 -  train: {r2_train:>{width}.3f}\n') # ---> 1 - best

    print(f'MAE -  test: {mae_test:>{width}.3f}')
    print(f'MAPE - test: {mape_test:>{width}.3f}')
    print(f'RMSE - test: {rmse_test:>{width}.3f}')
    print(f'R_2  - test: {r2_test:>{width}.3f}')


#======================================================================
# Функция для расчета метрик и вывода на экран
def show_metrics(y_train,y_train_pred,y_test, y_test_pred):

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)*100
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    r2_train = r2_score(y_train, y_train_pred)

    mae_test = mean_absolute_error(y_test, y_test_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred)*100
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_test = r2_score(y_test, y_test_pred)

    width = 20
    print(f'MAE -  train: {mae_train:>{width}.3f}') # ---> 0 - best
    print(f'MAPE - train: {mape_train:>{width}.3f}') # ---> 0 - best
    print(f'RMSE - train: {rmse_train:>{width}.3f}') # ---> 0 - best
    print(f'R_2 -  train: {r2_train:>{width}.3f}\n') # ---> 1 - best

    print(f'MAE -  test: {mae_test:>{width}.3f}')
    print(f'MAPE - test: {mape_test:>{width}.3f}')
    print(f'RMSE - test: {rmse_test:>{width}.3f}')
    print(f'R_2  - test: {r2_test:>{width}.3f}')