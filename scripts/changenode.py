import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--micro", type=str)
parser.add_argument("--node", type=str)
args = parser.parse_args()

file_name = "mongodb-" + args.micro + ".yaml"
file_read_name = "k8s-yaml-base/"
file_write_name = "k8s-yaml/"
file_read = open(file_read_name + file_name, 'r')
file_write = open(file_write_name + file_name, 'w')

is_write = False
write_ser = "      nodeName: " + args.node + "\n"
for line in file_read:
    file_write.write(line)
    if "template" in line:
        is_write = True
    if "spec" in line and is_write == True:
        file_write.write(write_ser)

file_read.close()
file_write.close()
