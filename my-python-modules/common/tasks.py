"""
Project: White Mold 
Description: This class manage the time spent in tasks during the processing in Python programs
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 20/02/2024
Version: 1.0
"""
# Importing Python libraries 
from datetime import datetime

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

class Tasks:

    def __init__(self):
        self.processing_start_time = datetime.now()
        self.processing_finish_time = None
        self.processiing_duration = None
        self.tasks = []

    def start_task(self, task_name):
        # getting time
        # now = datetime.now()

        # setting task attributes 
        task = {
            'task_name': task_name,
            'task_start': datetime.now(),
            'task_finish': None,
            'task_duration': None,
        }

        # adding task in tasks list
        self.tasks.append(task)

        
    def finish_task(self, task_name):
        for task in self.tasks:
            if task['task_name'] == task_name:            
                task_start = task['task_start']
                now = datetime.now()
                task_duration = now - task_start 
                task['task_finish'] = now
                task['task_duration'] = task_duration

    def finish_processing(self):
        self.processing_finish_time = datetime.now()
        self.processiing_duration = self.processing_finish_time - self.processing_start_time
        
    def to_string(self):
        text = LINE_FEED + LINE_FEED + \
               'TASKS SUMMARY' + LINE_FEED + \
               '-------------' + LINE_FEED + \
               LINE_FEED
        count = 0
        for task in self.tasks:
            count += 1
            text += f'{count:02}. ' + \
                    task['task_name'].ljust(60) + \
                    task['task_start'].strftime("%H:%M:%S") + ' > ' + \
                    task['task_finish'].strftime("%H:%M:%S") + ' = ' + \
                    str(task['task_duration']) + \
                    LINE_FEED
                    # task['task_duration'].total_seconds() + \

        text += LINE_FEED + \
                'Processing total time'.ljust(64) + \
                self.processing_start_time.strftime("%H:%M:%S") + ' > ' + \
                self.processing_finish_time.strftime("%H:%M:%S") + ' = ' + \
                str(self.processiing_duration) + \
                LINE_FEED

        # returning all tasks in one strign text 
        return text             

    # def get_task

    #     print("Current date:",datetime.utcnow())
    #     date= datetime.utcnow() - datetime(1970, 1, 1)
    #     print("Number of days since epoch:",date)
    #     seconds =(date.total_seconds())
    #     milliseconds = round(seconds*1000)
    #     print("Milliseconds since epoch:",milliseconds)