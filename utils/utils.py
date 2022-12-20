# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:53:11 2021

@author: Nutzer
"""

import time
import os
import glob
from skimage import io


def print_timestamp(msg, args=None):
    
    print('[{0:02.0f}:{1:02.0f}:{2:02.0f}] '.format(time.localtime().tm_hour,time.localtime().tm_min,time.localtime().tm_sec) +\
          msg.format(*args if not args is None else ''))



    
