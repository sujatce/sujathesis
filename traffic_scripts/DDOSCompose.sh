sleep 1020 #Wait for 17 minutes - This allows only normal traffic in initial period, this is important, as initial period is used to train the model
sleep random%120 #randomly sleep up to 2 mins
./wrk -D exp -t 70 -c 150 -d 5 -L -s ./scripts/social-network/compose-post.lua http://localhost:8080/wrk2-api/post/compose -R 200

