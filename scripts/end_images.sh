IMAGE=$1
CONTAINERID=`docker ps|grep ${IMAGE}|awk '{print $1}'`

echo $CONTAINERID
if [ -n $CONTAINERID ]; then
	docker stop $CONTAINERID
fi
echo stop docker
