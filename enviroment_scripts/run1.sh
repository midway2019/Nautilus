echo change path into Microservivce
pp=`pwd`
cd DeathStarBench/socialNetwork
echo `pwd`
kubectl delete -f k8s-yaml1/
kubectl apply -f k8s-yaml1/
sleep 2
kubectl apply -f k8s-yaml1/
dd=`kubectl get pod -n social-network |grep /|awk '{print $2}' |grep [^1/1]|wc -l`
echo waiting the deployments
while [ $dd -gt 1 ]
do
        echo $dd
        sleep 2
        dd=`kubectl get pod -n social-network |grep /|awk '{print $2}' |grep [^1/1]|wc -l`
done
echo deploy have finished
sleep 5

ip=`kubectl -n social-network get svc nginx-thrift |awk '{print $3}'|sed -n '2p'`
python changeip.py --ip $ip
python scripts/init_social_graph.py --ip $ip
echo generate http requests have finished
./wrk2/wrk -D exp -t1 -c1 -d20s -L -s ./wrk2/scripts/social-network/compose-post.lua http://$ip:8080/wrk2-api/post/compose -R$1 > sss.csv
#./wrk2/wrk -D exp -t1 -c1 -d20s -L -s ./wrk2/scripts/social-network/compose-post.lua http://$ip:8080/wrk2-api/post/compose -R$1 
date +%s
time=`date +%s`
port=`kubectl -n social-network get svc jaeger-out|awk '{print $5}'|sed -n '2p'|cut -d : -f 2|cut -d / -f 1`
echo go back to scheduler
cd $pp
sleep 10
python wrk2/getdata.py
#python getdata.py --port $port --save test1.json --start $time
#python servejson.py  --file result1/result.csv  
wait

