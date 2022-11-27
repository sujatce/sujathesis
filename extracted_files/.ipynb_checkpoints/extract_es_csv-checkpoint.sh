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