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
wait

