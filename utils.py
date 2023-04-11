# -*- coding: utf-8 -*-
import os


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
