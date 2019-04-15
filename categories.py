# -*- coding: utf-8 -*-

import os
import pandas as pd

import constants as const


def fill_number_category(out_path):
    numbers = [
            "ноль",
            "один",
            "два",
            "три",
            "четыре",
            "пять",
            "шесть",
            "семь",
            "восемь",
            "девять"
            ]
    f = open(out_path, "w")
    for i in numbers:
        f.write(i + "\r\n")
    f.close()


def fill_direction_category(out_path):
    directions = [
            "вперед",
            "назад",
            "влево",
            "вправо",
            "север",
            "юг",
            "восток",
            "запад"
            ]
    with open(out_path, "w") as f:
        for d in directions:
            f.write(d + "\r\n")


def fill_name_category(out_path):
    names_frame = pd.read_csv(os.path.join(const.DATA_PATH, 'russian_names.csv'), header=0, sep=';')
    with open(out_path, 'w') as f:
        frame = names_frame.sort_values('PeoplesCount', ascending=False)
        for row in frame.head(n=300).itertuples():
            f.write(row.Name.lower().replace('ё', 'е') + '\n')


def fill_categories():
    if not os.path.exists(const.CATEGORIES_PATH):
        os.makedirs(const.CATEGORIES_PATH)
        
    categories = ["number", "direction", "name", "surname"]
    for category in categories:
        filename = category + ".txt"
        out_path = os.path.join(const.CATEGORIES_DIR, filename)

        function_name = "fill_" + category + "_category"
        globals()[function_name](out_path)


if __name__=="__main__":
    fill_categories()
