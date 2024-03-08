IOBAND=$1
IMAGE=$2

docker run -d -v /home/test_file:/test_file --device-read-bps /dev/dm-0:$IOBAND --device-write-bps /dev/dm-0:$IOBAND $IMAGE
