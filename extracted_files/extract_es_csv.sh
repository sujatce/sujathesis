cd /home/ubuntu/sujathesis/extracted_files/
/home/ubuntu/.local/bin/es2csv -q '*' -f traceID spanID process.serviceName operationName startTimeMillis startTime tags references -S startTime -u http://18.212.223.207:9200 -i jaeger-span-2022-11-25 -D span -o /home/ubuntu/sujathesis/extracted_files/traffictrace.csv
git add *
git commit
git push