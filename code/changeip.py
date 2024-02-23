import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str)
args = parser.parse_args()
wfile = open('wrk2/scripts/social-network/compose-post.lua','w')
for line in open('wrk2/scripts/social-network/.compose-post.lua','r'):
    if 'local path' in line:
        string = '  local path = \"http://' + args.ip + ':8080/wrk2-api/post/compose\"\n'
        wfile.writelines(string)
    else:
        wfile.writelines(line)
wfile.close()