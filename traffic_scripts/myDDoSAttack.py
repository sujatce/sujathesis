import aiohttp
import asyncio
import time


async def read_user_timeline(session, url, user):
    async with session.get(url+"?user_id="+str(user)+"&start=0&stop=5000") as response:
        return await response.text()


async def normal_user_timeline_read(url, users):

    tasks = []
    print("Normal reading user account traffic")
    conn = aiohttp.TCPConnector(limit=200)
    async with aiohttp.ClientSession(connector=conn) as session:

        # read a user account timeline every five seconds
        for i in users:

            task = asyncio.ensure_future(read_user_timeline(session, url, str(i)))
            tasks.append(task)
            resp = await asyncio.gather(*tasks)
            print("\n\nReading User_"+str(i) + " account activity\n")

            time.sleep(5)

        resps = await asyncio.gather(*tasks)


async def dos_attack_timeline_read(url, anomaly_users):

    tasks = []
    print("\n\nNormal anomalous user account traffic")

    conn = aiohttp.TCPConnector(limit=200)
    async with aiohttp.ClientSession(connector=conn) as session:

        # generate a account read API call as part of DDoS
        for a in anomaly_users:

            anomaly = asyncio.ensure_future(read_user_timeline(session, url, str(a)))
            tasks.append(anomaly)
            resp = await asyncio.gather(*tasks)
            print("\n\nAnomalous - Reading User_" + str(a) + " account activity\n")

if __name__ == '__main__':
    print("Denial Distribution of Services")

    # address
    address = "http://localhost:8080/wrk2-api/user-timeline/read"

    starting_time = time.time()

    # wait for initial posting of content to cache user activity
    # time.sleep(5)

    with open("../data/experiment/user_IDs.txt", 'r') as file:
        user_IDs = file.read().strip().split(',')

    normal_user_IDs = user_IDs[:-1]

    normal_users = normal_user_IDs[:24]

    # read normal user timeline throughout data traffic
    print("Denial Distribution of services - Normal Data")
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(normal_user_timeline_read(address, normal_users))
    loop.run_until_complete(future)

    normal_time = time.time()
    print("DDoS - Normal Data Time: " + str(round(normal_time - starting_time)) + " seconds")

    # time.sleep(5)

    # read user timeline GET requests
    with open("../data/experiment/user_IDs.txt", 'r') as file:
        attack_user_IDs = file.read().strip().split(',')

    anomaly_user_IDs = attack_user_IDs[:-1]
    anomalous_users = anomaly_user_IDs[24:124]

    print(len(anomaly_user_IDs))
    print(str(anomaly_user_IDs))

    # seed a DDoS attack with reading user timeline at end of testing data set
    print("Denial Distribution of services - Anomalous Data")
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(dos_attack_timeline_read(address, anomalous_users))
    loop.run_until_complete(future)

    end_time = time.time()
    print("DDoS - Total Time: " + str(round(end_time - starting_time)) + " seconds")
