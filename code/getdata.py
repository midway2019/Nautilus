import sys
import os
import numpy as np


def getprofile(benchmark):
    QoS = 0
    throughput = 0
    script = 'DeathStarBench/' + benchmark + '/sss.csv'
    file = open(script, 'r')
    text = ' 1'
    while text:
        text = file.readline()
        if('Requests/sec:' in text):
            throughput = float(text.split()[1])
        if('Thread Stats ' in text):
            text = file.readline()
            QoS_text = text.split()[3]
            if('ms' in QoS_text):
                QoS = float(QoS_text[0:-2])/1000
            elif('us' in QoS_text):
                QoS = float(QoS_text[0:-2])/1000000
            elif('s' in QoS_text):
                QoS = float(QoS_text[0:-1])/1
    return QoS, throughput
