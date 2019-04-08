# -*- coding: utf-8 -*-

import os
import pandas as pd

import constants as const

def fill_number_category(out_path):
    f = open(out_path, "w")
    for i in range(10):
        f.write("%d\r\n" % i)
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
    f = open(out_path, "w")
    for d in directions:
        f.write(d + "\r\n")
    f.close()

def fill_categories():
    if not os.path.exists(const.CATEGORIES_DIR):
        os.makedirs(const.CATEGORIES_DIR)
        
    categories = ["number", "direction"]
    for category in categories:
        filename = category + ".txt"
        out_path = os.path.join(const.CATEGORIES_DIR, filename)

        function_name = "fill_" + category + "_category"
        globals()[function_name](out_path)

if __name__=="__main__":
    fill_categories()
