import numpy as np
def migrate_microservice(source,target,mnum):
    source = np.array(source)
    target = np.array(target)
    micronum = source[:,0]
    line = source[np.where(micronum != mnum)[0][0],:]
    line1 = source[np.where(micronum == mnum)[0][0],:]
    tmp = source[line1, :]
    source = source[line,:]
    np.concatenate((target,tmp),axis=0)
    target = np.sort(target)
    return source, target

def migration(resources,deployments,node_cpu,node_mem,graph):
    k = 3
    cpu = []
    mem = []
    comm_delta = []
    for enum,deployment in enumerate(deployments):
        resource = resources[deployment]
        cpu_usage, mem_usage = get_sum(resource, deployment)
        cpu.append(cpu_usage)
        mem.append(mem_usage)
        comm_delta_node = []
        for item in deployment:
            for j in range(0, k):
                if(j != enum):
                    k_comm_delta = sum(graph[item][deployment[enum]])
                    j_comm_delta = sum(graph[item][deployments[j]])
                    comm_delta_node.append([item,j,k_comm_delta - j_comm_delta])
        comm_delta_node = sorted(comm_delta_node)
        comm_delta.append(comm_delta_node)

    res_cpu = node_cpu - cpu
    res_mem = node_mem - mem
    deployments = deployments
    while(res_cpu[0] < 0):
        deployment = deployments[0]
        for item in comm_delta[0]:
            servnum = item[0]
            tarnode = item[1]
            cpu_par = resources[servnum][0]
            mem_par = resources[servnum][1]
            if(cpu_par < res_cpu[tarnode] and mem_par < res_mem[tarnode]):
                comm_delta[0], comm_delta[item[0]] = migrate_microservice(comm_delta[0],comm_delta[item[0]],servnum)
                deployments[tarnode].append(servnum)
                deployments[0].delete(servnum)
                res_cpu[tarnode] = res_cpu[tarnode] - cpu_par
                res_mem[tarnode] = res_mem[tarnode] - mem_par
                res_cpu[0] = res_cpu[0] + cpu_par
                res_mem[0] = res_mem[0] + mem_par
                break
        if(res_cpu[0] > 0):
            break
    return deployments, resources

def get_sum(resources,deployments):
    cpu = 0
    mem = 0
    for item in deployments:
        cpu = cpu + resources[item][0]
        mem = mem + resources[item][1]
    return cpu, mem

