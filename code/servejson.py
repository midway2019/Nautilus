import json
import time, datetime
import csv


def getDate(t):
    datatime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(float(str(t)[0:10])))
    datatime = datatime+'.'+str(t)[10:]
    t = datatime
    cday = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
    return cday
def getQoS(data):
    QoS = []
    for request in data:
        spans = request[u'spans']
        startTime = []
        Duration = []
        endTime = []
        for span in spans:
            startTime.append(span[u'startTime'])
            Duration.append(span[u'duration'])
            endTime.append(span[u'startTime'] + span[u'duration'])
        startTime.sort()
        endTime.sort()
        QoS.append(endTime[-1] - startTime[0])
    QoS.sort()
    return QoS[int(round(0.99*len(QoS)-1))]
def servejson():
    jsonfile = open('test.json','r')
    load_dict = json.load(jsonfile)
    data = load_dict[u'data']
    Time = {}
    Start = {}
    file = open('result.csv','w')
    csv_writer = csv.writer(file)
    QoS = getQoS(data)
    for request in data:
        serviceName = request[u'processes']
        spans = request[u'spans']
        ss = {}
        for span in spans:
            ss[span[u'spanID']] = span
        for span in spans:
            processName = serviceName[span[u'processID']][u'serviceName']
            Duration = span[u'duration']
            startTime = span[u'startTime']
            reference = span[u'references']
            if len(reference) == 0:
                processFather = u''
            else:
                process = ss[(reference[0])[u'spanID']]
                processFather = serviceName[process[u'processID']][u'serviceName']
            if(processName == processFather):
                continue
            Name = processName + ' ' + processFather
            if(Time.has_key(Name)):
                tt = Time[Name]
                tt.append(Duration)
                Time[Name] = tt
            else:
                Time[Name] = [Duration]
            if(Start.has_key(Name)):
                tt = Start[Name]
                tt.append(startTime)
                Start[Name] = tt
            else:
                Start[Name] = [startTime]
    info = []
    for key in Time.keys():
        kTime = Time[key]
        kTime.sort()
        kStart = Start[key]
        FirstStart = min(kStart)
        FirstStart = getDate(FirstStart)
        throughput = []
        for i,item in enumerate(kStart):
            kStart[i] = getDate(kStart[i])
            kStart[i] = (kStart[i] - FirstStart).total_seconds()
            if kStart[i] <= 1:
                throughput.append(kStart[i])
        result = [key,QoS,kTime[int(round(0.99*len(kTime))-1)],len(throughput)]
        info.append(result)
    return info
        #csv_writer.writerow(result)
    #file.close()

    

