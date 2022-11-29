echo 'First generate a regular traffic using compose, read-home-timeline and read-user-timeline - 10 Threads with 10 requests per second - This simulates normal scenario traffic which will be used to train the model'
./wrk -D exp -t 5 -c 10 -d 120 -L -s ./scripts/social-network/compose-post.lua http://localhost:8080/wrk2-api/post/compose -R 10 & ./wrk -D exp -t 5 -c 10 -d 120 -L -s ./scripts/social-network/read-home-timeline.lua http://localhost:8080/wrk2-api/home-timeline/read -R 10 & ./wrk -D exp -t 5 -c 10 -d 120 -L -s ./scripts/social-network/read-user-timeline.lua http://localhost:8080/wrk2-api/user-timeline/read -R 10

echo ' This step will generate flooded calls (like Cyber attack) by producing 500 requests per 5 seconds on compose. Imperva - Cyber security leader indicates a typical DDos Attack has over 100 requests per second in a typical application which can create disruption in application. Hence we are simply using this scenario to mimic and produce a DDOS attack scenario which is then used to test the model. This is as per professors inputs' 

./wrk -D exp -t 40 -c 80 -d 5 -L -s ./scripts/social-network/compose-post.lua http://localhost:8080/wrk2-api/post/compose -R 100

