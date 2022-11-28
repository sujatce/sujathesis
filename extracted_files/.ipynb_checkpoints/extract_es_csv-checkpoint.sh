cd /home/ubuntu/thesis/DeathStarBench/socialNetwork
sudo docker-compose up -d

cd /home/ubuntu/sujathesis/extracted_files/
/home/ubuntu/.local/bin/es2csv -q '*' -f traceID spanID process.serviceName operationName startTimeMillis startTime tags references -S startTime -u http://18.212.223.207:9200 -i jaeger-span-2022-11-25 -D span -o /home/ubuntu/sujathesis/extracted_files/traffictrace.csv
git add *
git commit
git push

status:[400 TO 499]
startTime:>1669420762829319
startTimeMillis: >now-2d

/home/ubuntu/.local/bin/es2csv -q 'startTime:[1669420762829319 TO 1669420762829339]' -f traceID spanID process.serviceName operationName startTimeMillis startTime tags references -S startTime -u http://18.212.223.207:9200 -i jaeger-span-2022-11-25 -D span -o /home/ubuntu/sujathesis/extracted_files/traffictrace.csv


/home/ubuntu/.local/bin/es2csv -q 'startTimeMillis:>now-2d' -f traceID spanID process.serviceName operationName startTimeMillis startTime tags references -S startTime -u http://54.197.37.208:9200 -i jaeger-span-2022-11-25 -D span -o /home/ubuntu/sujathesis/extracted_files/traffictrace.csv


./wrk -D exp -t 5 -c 10 -d 125 -L -s ./scripts/social-network/compose-post.lua http://localhost:8080/wrk2-api/post/compose -R 10 & ./wrk -D exp -t 5 -c 10 -d 125 -L -s ./scripts/social-network/read-home-timeline.lua http://localhost:8080/wrk2-api/home-timeline/read -R 10 & ./wrk -D exp -t 5 -c 10 -d 125 -L -s ./scripts/social-network/read-user-timeline.lua http://localhost:8080/wrk2-api/user-timeline/read -R 10 & python3 /home/ubuntu/thesis/sujathesis/traffic_scripts/testscenario_1.py

#### For viewing tensorboard grapsh
pip install tensorboard
tensorboard --logdir sujathesis/suja_stgcn/output/tensorboard/train