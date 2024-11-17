"""
Project: White Mold 
Description: Manage log messages.
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 30/11/2023
Version: 2.0
"""

# Importing libraries
import os
import logging
from datetime import datetime
from common.utils import Utils

def logging_create_log(log_folder, log_filename):
       
    # setting logging file 
    now = datetime.now()
    log_filename_full = log_filename + now.strftime('-%Y-%m-%d-%Hh%Mm%Ss.log')
    filename = os.path.join(log_folder, log_filename_full)
    log_format = '%(levelname)s %(asctime)s: %(message)s'
    logging.basicConfig(filename=filename, encoding='utf-8', 
                        level=logging.DEBUG, format=log_format)

def logging_info(message):
    logging.info(message)

def logging_warning(message):
    logging.warning(message)

def logging_debug(message):
    logging.debug(message)

def logging_error(message):
    logging.error(message)

def logging_critical(message):
    logging.critical(message)

def get_datetime():
    now = datetime.now()
    date_time_text = now.strftime('%Y/%m/%d %H:%M:%S')
    return date_time_text

def logging_sheet(sheet_list):
    logging_info(f'')
    for item in sheet_list:
        logging_info(f'{item}')    
        continue
    logging_info(f'')
     