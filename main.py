import argparse  # модуль (библиотека) для обработки параметров коммандной строки
import numpy as np  # модуль для работы с массивами и векторных вычислений
import skimage.io  # модуль для обработки изображений, подмодуль для чтения и записи
# в некоторых модулях некоторые подмодули надо импортировать вручную, а не просто "import module" и потом в коде писать "module.submodule.something..."


def mse(img1, img2):  # можно задавать значения параметров по умолчанию
    """Вычисление среднеквадратической ошибки между двумя изображениями."""
    height, width = img1.shape[:2]
    error_sum = 0.0

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    for i in range(height):
        for j in range(width):
            error_sum += (img1[i][j] - img2[i][j]) ** 2 / (height * width)
    mse_value = error_sum
    return mse_value


def psnr(img1, img2):  # можно задавать значения параметров по умолчанию
    """Вычисление пикового отношения сигнал/шум между двумя изображениями."""
    mse_value = mse(img1, img2)
    if mse_value == 0:
        return float('inf')  # Бесконечность, если изображения идентичны
    max_pixel_value = 1.0  # Нормализуем до диапазона [0, 1]
    psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse_value))
    print(psnr_value)


def ssim(img1, img2):  # можно задавать значения параметров по умолчанию
    """Вычисление индекса структурного сходства (SSIM) между двумя изображениями."""
    height, width = img1.shape[:2]
    # Средние значения
    mean1, mean2 = 0.0, 0.0
    for i in range(height):
        for j in range(width):
            mean1 += img1[i][j]
            mean2 += img2[i][j]
    mean1 /= (height * width)
    mean2 /= (height * width)
    
    # Дисперсии и ковариация
    var1, var2, covariance = 0.0, 0.0, 0.0
    for i in range(height):
        for j in range(width):
            diff1 = img1[i][j] - mean1
            diff2 = img2[i][j] - mean2
            var1 += diff1 ** 2
            var2 += diff2 ** 2
            covariance += diff1 * diff2
    var1 /= (height * width - 1)
    var2 /= (height * width - 1)
    covariance /= (height * width - 1)
    
    # Параметры стабилизации
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    
    # Вычисление SSIM
    ssim_value = ((2 * mean1 * mean2 + C1) * (2 * covariance + C2)) / ((mean1 ** 2 + mean2 ** 2 + C1) * (var1 + var2 + C2))
    print(ssim_value)


def gauss(img, sigma_d):
    """Применение гауссового фильтра к изображению."""
    height, width = img.shape
    # Определение размера ядра
    kernel_radius = int(3 * sigma_d)
    kernel_size = 2 * kernel_radius + 1
    
    # Создание ядра Гаусса
    gauss_kernel = np.zeros((kernel_size, kernel_size))
    for x in range(-kernel_radius, kernel_radius + 1):
        for y in range(-kernel_radius, kernel_radius + 1):
            gauss_kernel[x + kernel_radius][y + kernel_radius] = np.exp(-(x**2 + y**2) / (2 * sigma_d**2))
    
    # Нормализация ядра
    gauss_kernel /= np.sum(gauss_kernel)
    
    # Применение фильтра
    result = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            weighted_sum = 0.0
            for x in range(-kernel_radius, kernel_radius + 1):
                for y in range(-kernel_radius, kernel_radius + 1):
                    xi = min(max(i + x, 0), height - 1)
                    yj = min(max(j + y, 0), width - 1)
                    weighted_sum += img[xi][yj] * gauss_kernel[x + kernel_radius][y + kernel_radius]
            result[i][j] = weighted_sum
    return result


def bilateral(img, sigma_d, sigma_r):
    """Применение билатерального фильтра к изображению."""
    height, width = img.shape
    result = np.zeros_like(img)
    
    # Определение радиуса ядра
    kernel_radius = int(3 * sigma_d)
    
    for i in range(height):
        for j in range(width):
            weighted_sum = 0.0
            normalization = 0.0
            
            # Применение фильтра
            for x in range(-kernel_radius, kernel_radius + 1):
                for y in range(-kernel_radius, kernel_radius + 1):
                    xi = min(max(i + x, 0), height - 1)
                    yj = min(max(j + y, 0), width - 1)
                    
                    # Пространственное расстояние
                    spatial_weight = np.exp(-(x**2 + y**2) / (2 * sigma_d**2))
                    
                    # Интенсивность
                    intensity_weight = np.exp(-((img[xi][yj] - img[i][j]) ** 2) / (2 * sigma_r**2))
                    
                    weight = spatial_weight * intensity_weight
                    weighted_sum += img[xi][yj] * weight
                    normalization += weight
            
            result[i][j] = weighted_sum / normalization
    return result


def median(img, rad):
    """Применение медианной фильтрации с окном (2 * rad + 1) x (2 * rad + 1)."""
    height, width = img.shape
    window_size = 2 * rad + 1  # Размер окна фильтрации
    result = np.zeros_like(img)
    
    # Применение медианной фильтрации
    for i in range(height):
        for j in range(width):
            # Определяем границы окна
            i_min = max(i - rad, 0)
            i_max = min(i + rad + 1, height)
            j_min = max(j - rad, 0)
            j_max = min(j + rad + 1, width)
            
            # Извлекаем подматрицу и находим медиану
            window = img[i_min:i_max, j_min:j_max]
            median_value = np.median(window)
            result[i, j] = median_value
    
    return result


def compare(img1, img2):
    """Сравнение изображений на основе преобразования Фурье."""
    # Преобразование Фурье
    fft1 = np.fft.fft2(img1)
    fft2 = np.fft.fft2(img2)
    
    # Вычисление амплитуд
    amplitude1 = np.abs(fft1)
    amplitude2 = np.abs(fft2)
    
    # Вычисление разницы амплитуд
    diff = np.abs(amplitude1 - amplitude2)
    threshold = np.max(amplitude1) * 0.1  # пример порога сравнения
    match = np.mean(diff) < threshold
    
    return match


def img_prepare(img):
    imgr = img / 255
    imgr = img[:, :, 0]
    return imgr

if __name__ == '__main__':  # если файл выполняется как отдельный скрипт (python script.py), то здесь будет True. Если импортируется как модуль, то False. Без этой строки весь код ниже будет выполняться и при импорте файла в виде модуля (например, если захотим использовать эти функции в другой программе), а это не всегда надо.
    # получить значения параметров командной строки
    parser = argparse.ArgumentParser(  # не все параметры этого класса могут быть нужны; читайте мануалы на docs.python.org, если интересно
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help',  # в конце списка параметров и при создании list, tuple, dict и set можно оставлять запятую, чтобы можно было удобно комментить или добавлять новые строчки без добавления и удаления новых запятых
    )
    parser.add_argument('command', help='Command description')  # add_argument() поддерживает параметры вида "-p 0.1", может сохранять их как числа, строки, включать/выключать переменные True/False ("--activate-someting"), поддерживает задание значений по умолчанию; полезные параметры: action, default, dest - изучайте, если интересно
    parser.add_argument('parameters', nargs='*')  # все параметры сохранятся в список: [par1, par2,...] (или в пустой список [], если их нет)
    parser.add_argument('input_file1')
    parser.add_argument('input_file2')
    args = parser.parse_args()

    res = None

    # Можете посмотреть, как распознаются разные параметры. Но в самом решении лишнего вывода быть не должно.
    # print('Распознанные параметры:')
    # print('Команда:', args.command)  # между 2 выводами появится пробел
    # print('Её параметры:', args.parameters)
    # print('Входной файл:', args.input_file)
    # print('Выходной файл:', args.output_file)

    img1 = skimage.io.imread(args.input_file1)  # прочитать изображение

    img1 = img_prepare(img1)

    # получить результат обработки для разных комманд
    if args.command == 'mse':
        img2 = skimage.io.imread(args.input_file2)
        img2 = img_prepare(img2)
        ans = mse(img1, img2)
        print(ans)

    elif args.command == 'psnr':
        img2 = skimage.io.imread(args.input_file2)
        img2 = img_prepare(img2)
        psnr(img1, img2)

    elif args.command == 'ssim':
        img2 = skimage.io.imread(args.input_file2)
        img2 = img_prepare(img2)
        ssim(img1, img2)

    elif args.command == 'median':
        rad = args.parameters[0]
        res = median(img1, rad)

    elif args.command == 'gauss':
        sigma_d = args.parameters[0]
        res = gauss(img1, sigma_d)

    elif args.command == 'bilateral':
        sigma_d = args.parameters[0]
        sigma_r = args.parameters[1]
        res = bilateral(img1, sigma_d, sigma_r)

    elif args.command == 'compare':
        img2 = skimage.io.imread(args.input_file2)
        img2 = img_prepare(img2)
        compare(img1, img2)

    if res != None:
        # сохранить результат
        res = np.clip(res, 0, 1)  # обрезать всё, что выходит за диапазон [0, 1]
        res = np.round(res * 255).astype(np.uint8)  # конвертация в байты
        skimage.io.imsave(args.input_file2, res)


    # Ещё некоторые полезные штуки в Питоне:
    
    # l = [1, 2, 3]  # list
    # l = l + [4, 5]  # сцепить списки
    # l = l[1:-2]  # получить кусок списка (slice)
    
    # Эти тоже можно сцеплять и т.п. - читайте мануалы
    # t = (1, 2, 3)  # tuple, элементы менять нельзя, но можно сцеплять и т.д.
    # s = {1, 'a', None}  # set
    
    # d = {1: 'a', 2: 'b'}  # dictionary
    # d = dict((1, 'a'), (2, 'b'))  # ещё вариант создания
    # d[3] = 'c'  # добавить или заменить элемент словаря
    # value = d.get(3, None)  # получить (get) и удалить (pop) элемент словаря, а если его нет, то вернуть значение по умолчанию (в данном случае - None)
    # for k, v in d.items()    for k in d.keys() (или просто "in d")    for v in d.values() - варианты прохода по словарю
    
    # if 6 in l:  # проверка на вхождение в list, tuple, set, dict
    #     pass
    # else:
    #     pass

    # print(f'Какое-то число: {1.23}. \nОкруглить до сотых: {1.2345:.2f}. \nВывести переменную: {args.input_file}. \nВывести список: {[1, 2, "a", "b"]}')  # f-string позволяет создавать строки со значениями переменных
    # print('Вывести текст с чем-нибудь другим в конце вместо перевода строки.', end='1+2=3')
    # print()  # 2 раза перевести строку
    # print()
    # print('  Обрезать пробелы по краям строки и перевести всё в нижний РеГиСтР.   \n\n\n'.strip().lower())

    # import copy
    # tmp = copy.deepcopy(d)  # глубокая, полная копия объекта
    
    # Можно передавать в функцию сколько угодно параметров, если её объявить так:
    # def func(*args, **kwargs):
    # Тогда args - это list, а kwargs - это dict
    # При вызове func(1, 'b', c, par1=2, par2='d') будет: args = [1, 'b', c], а kwargs = {'par1': 2, 'par2': 'd'}.
    # Можно "раскрывать" списки и словари и подавать их в функции как последовательность параметров: some_func(*[l, i, s, t], **{'d': i, 'c': t})
    
    # p = pathlib.Path('/home/user/Documents') - создать объект Path
    # p2 = p / 'dir/file.txt' - добавить к нему ещё уровени
    # p.glob('*.png') и p.rglob('*.png') - найти все файлы нужного вида в папке, только в этой папке и рекурсивно; возвращает не list, а generator (выдаёт только по одному элементу за раз), поэтому если хотите получить сразу весь список файлов, то надо обернуть результат в "list(...)".
