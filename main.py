import argparse  # модуль (библиотека) для обработки параметров коммандной строки
import numpy as np  # модуль для работы с массивами и векторных вычислений
import skimage.io  # модуль для обработки изображений, подмодуль для чтения и записи
# в некоторых модулях некоторые подмодули надо импортировать вручную, а не просто "import module" и потом в коде писать "module.submodule.something..."


def mse(img1, img2):  # можно задавать значения параметров по умолчанию
    '''Отразить изображение'''  # комментарий docsting - выводится в подсказке к функции
    height, width = img1.shape[:2]  # 3-я ось (каналы цветов) нам здесь не нужна
    

    #raise NotImplementedError('Напишите код функции!')  # Вызвать исключение нужного типа (надо удалить, чтобы функция работала)
    
    return res


def psnr(img1, img2):  # можно задавать значения параметров по умолчанию
    '''Отразить изображение'''  # комментарий docsting - выводится в подсказке к функции
    height, width = img1.shape[:2]  # 3-я ось (каналы цветов) нам здесь не нужна
    

    #raise NotImplementedError('Напишите код функции!')  # Вызвать исключение нужного типа (надо удалить, чтобы функция работала)
    
    return res


def ssim(img1, img2):  # можно задавать значения параметров по умолчанию
    '''Отразить изображение'''  # комментарий docsting - выводится в подсказке к функции
    height, width = img1.shape[:2]  # 3-я ось (каналы цветов) нам здесь не нужна
    

    #raise NotImplementedError('Напишите код функции!')  # Вызвать исключение нужного типа (надо удалить, чтобы функция работала)
    
    return res


def extract(img, left_x, top_y, width, height: int):  # можно задавать типы параметров: будут выводится в подсказке к функции, но проверки типов нет
    res = np.empty((height, width), dtype=float)  # просто массив, без заполнения нулями или единицами после создания
    
    # тут ваш код
    #raise NotImplementedError('Напишите код функции!')

    h, w = img.shape[:2]

    for i in range(height):
        for j in range(width):
            if top_y + i < 0 or left_x + j < 0 or top_y + i >= h or left_x + j >= w:
                res[i][j] = 0
            else:
                res[i][j] = img[top_y + i][left_x + j]

                
    return res


def rotate(img, direction, angle):
    # тут ваш код
    # raise NotImplementedError('Напишите код функции!')

    h, w = img.shape[:2]

    angle %= 360

    if angle == 90 and direction == 'ccw' or angle == 270 and direction == 'cw':
        res = np.zeros((w, h), dtype=float)
        for i in range(h):
            for j in range(w):
                res[j][i] = img[i][w - j - 1]
    if angle == 180:
        res = np.zeros((h, w), dtype=float)
        for i in range(h):
            for j in range(w):
                res[i][j] = img[h - i - 1][w - j - 1]
    if angle == 90 and direction == 'cw' or angle == 270 and direction == 'ccw':
        res = np.zeros((w, h), dtype=float)
        for i in range(h):
            for j in range(w):
                res[j][i] = img[h - i - 1][j]
    if angle == 0:
        res = np.zeros((h, w), dtype=float)
        for i in range(h):
            for j in range(w):
                res[i][j] = img[i][j]
    
    return res


def autocontrast(img):
    res = np.zeros_like(img)  # массив из нулей такой же формы и типа
    
    # тут ваш код
    # raise NotImplementedError('Напишите код функции!')

    h, w = img.shape[:2]

    mx = 0
    mn = 256

    for i in range(h):
        for j in range(w):
            if img[i][j] > mx:
                mx = img[i][j]
            if img[i][j] < mn:
                mn = img[i][j]

    div = mx - mn
    
    for i in range(h):
        for j in range(w):
            res[i][j] = (img[i][j] - mn) / div

                
    return res


def fixinterlace(img):
           
    return res


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
    parser.add_argument('output_file')
    args = parser.parse_args()

    # Можете посмотреть, как распознаются разные параметры. Но в самом решении лишнего вывода быть не должно.
    # print('Распознанные параметры:')
    # print('Команда:', args.command)  # между 2 выводами появится пробел
    # print('Её параметры:', args.parameters)
    # print('Входной файл:', args.input_file)
    # print('Выходной файл:', args.output_file)

    img1 = skimage.io.imread(args.input_file1)  # прочитать изображение
    img2 = skimage.io.imread(args.input_file1)  # прочитать изображение

    img1 = img1 / 255  # перевести во float и диапазон [0, 1]
    img2 = img2 / 255
    if len(img1.shape) == 3:  # оставим только 1 канал (пусть будет 0-й) для удобства: всё равно это ч/б изображение
        img1 = img1[:, :, 0]

    if len(img2.shape) == 3:  # оставим только 1 канал (пусть будет 0-й) для удобства: всё равно это ч/б изображение
        img2 = img2[:, :, 0]

    # получить результат обработки для разных комманд
    if args.command == 'mse':
        res = mse(img1, img2)

    elif args.command == 'psnr':
        ans = psnr(img1, img2)

    elif args.command == 'rssim':
        direction = args.parameters[0]
        angle = int(args.parameters[1])
        res = rotate(img, direction, angle)

    elif args.command == 'amedian':
        res = autocontrast(img)

    elif args.command == 'gauss':
        res = fixinterlace(img)

    elif args.command == 'bilateral':
        res = fixinterlace(img)

    elif args.command == 'compare':
        res = fixinterlace(img)

    if res != None:
        # сохранить результат
        res = np.clip(res, 0, 1)  # обрезать всё, что выходит за диапазон [0, 1]
        res = np.round(res * 255).astype(np.uint8)  # конвертация в байты
        skimage.io.imsave(args.output_file, res)


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
