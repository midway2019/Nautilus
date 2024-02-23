import os
import numpy as np
def generateyaml(deployment, resources, benchmark):
    name = {0:'nginx',1:'unique-id',2:'url-shorten',3:'text',\
        4:'media',5:'user-mention',6:'compose-post',7:'user-timeline',\
            8:'post-storage',9:'write-home-timeline',10:'social-graph',11:'user'}
    size = np.size(resources)
    resources = np.reshape(resources, (int(size/3), 3))
    for i,line in enumerate(resources):
        CPU = line[0]
        Mem = line[1]
        IO = line[2]
        yaml(name[i], CPU, Mem, IO, benchmark)
def yaml(service, CPU, Mem, IO, benchmark):
    CPU = str(int(CPU*100)) + 'm'
    Mem = str(int(Mem*100)) + 'Mi'
    IO = str(int(IO*100)) + 'Ki'
    isTem = False
    position = ''
    #if service == 0:
        #position = '      nodeName: k8s-master\n'
    #else:
        #position = '      nodeName: ubuntu\n'
    dir = 'DeathStarBench/' + benchmark + '/'
    for file in os.listdir(dir + 'k8s-yaml'):
        isTem = False
        if service in file:
            file1 = open(dir + 'k8s-yaml1/' + file,'w')
            for line in open(dir + 'k8s-yaml/' + file,'r'):
                file1.writelines([line])
                if 'template' in line: 
                    isTem = True
                if isTem & ('metadata' in line):
                    file1.writelines(['      annotations:\n'])
                    file1.writelines(['        kubernetes.io/ingress-bandwidth: ', IO,'\n'])
                    file1.writelines(['        kubernetes.io/egress-bandwidth: ', IO,'\n'])
                if isTem & ('spec' in line):
                    file1.writelines([position])
                if 'image' in line:
                    file1.writelines(['        resources:\n'])
                    file1.writelines(['          limits:\n'])
                    file1.writelines(['            cpu: ',CPU,'\n'])
                    file1.writelines(['            memory: ',Mem,'\n'])
                    file1.writelines(['          requests:\n'])
                    file1.writelines(['            cpu: ',CPU,'\n'])
                    file1.writelines(['            memory: ',Mem,'\n'])
            file1.close()
        '''
        else:
            file1 = open('k8s-yaml1/' + file,'w')
            for line in open('k8s-yaml/' + file,'r'):
                file1.writelines([line])
                if 'template' in line:
                    isTem = True
                if isTem & ('spec' in line):
                    file1.writelines(['      nodeName: k8s-master\n'])
        '''


def changeResources(service,label,resources):
    name = {0:'nginx-thrift',1:'unique-id-service',2:'url-shorten-service',3:'text-service',\
        4:'media-service',5:'user-mention-service',6:'compose-post-service',7:'user-timeline-service',\
        8:'post-storage-service',9:'write-home-timeline-service',10:'social-graph-service',11:'user-service',12:'jaeger'}
    CPU = 0
    IO = 0
    Mem = 0
    if(label == 0):
        CPU = str(int(resources*100)) + 'm'
        string = 'cpu=' + CPU
        ss = 'kubectl set resources deployment -n social-network ' + name[service] + ' --limits=' + string + ' --requests=' + string
        os.system(ss)
    elif(label == 1):
        Mem = str(int(resources*100)) + 'Mi'
        string = 'memory=' + Mem
        ss = 'kubectl set resources deployment -n social-network ' + name[service] + ' --limits=' + string + ' --requests=' + string
        os.system(ss)
    elif(label == 2):
        IO = str(int(resources*100)) + 'Ki'
        string = IO