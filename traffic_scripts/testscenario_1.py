import requests
import time
import random
import aiohttp
import asyncio

startTime = time.time()
idx = 1

corReq = 0
incReq = 0

from random import randint


async def register_user(session, addr, user):

    payload = {
                'first_name': 'first_name_' + user,
                'last_name': 'last_name_' + user,
                'username': 'username_' + user,
                'password': 'password_' + user,
                'user_id': user
            }

    async with session.post(addr, data=payload) as response:
        return await response.text()

async def login_user(session, addr, user):

    payload = {
                'username': 'username_1' + user,
                'password': 'vol' + user+'97Chasm'
            }
        
    async with session.post(addr, data=payload) as response:
        return await response.text()
    
async def normal_registration(url):

    tasks = []
    print("Normal registration traffic")
    conn = aiohttp.TCPConnector(limit=200)
    async with aiohttp.ClientSession(connector=conn) as session:

        # wait five seconds
        idx = 970

        # register one or two new accounts every 5 seconds
        for i in range(24):

            value = randint(1, 2)

            for j in range(value):

                task = asyncio.ensure_future(register_user(session, url, str(idx)))
                tasks.append(task)
                resp = await asyncio.gather(*tasks)
                print("User_"+str(idx) + " registered\n")

                idx += 1
            time.sleep(5)
            resps = await asyncio.gather(*tasks)

        resps = await asyncio.gather(*tasks)


async def batch_registration(url):

    tasks = []
    print("\n\nBatch registration Attack")
    conn = aiohttp.TCPConnector(limit=200)
    async with aiohttp.ClientSession(connector=conn) as session:

        # prepare a batch registration of 100 bot accounts
        for i in range(1100, 1201):

            task = asyncio.ensure_future(register_user(session, url, str(i)))
            tasks.append(task)
            resp = await asyncio.gather(*tasks)
            print("User_"+str(i) + " registered\n")
        resps = await asyncio.gather(*tasks)
        print(str(len(resps)))

async def normal_login():
    with open("loginData.txt", "w") as file:
        file.write("The login Data\n\n")

    # logging in traffic for 2 minutes
    print("Normal Traffic")
    for i in range(0, 118):

        # incorrect traffic is 10% of overall traffic
        if idx % 11 == 0:

            print("\nIncorrect Login Traffic")
            print("User_" + str(idx) + " attempting to log in")
            response = requests.get(
                'http://localhost:8080/api/user/login?username=username_' + str(idx-1) +
                '&password=vol' + str(idx-1) + '97Chasm')
            print('Status code:' + str(response.status_code))
            incReq += 1
            with open("loginData.txt", "a") as file:
                file.write("\nUser_" + str(idx-1) + " failed to login")

        print("\n\nCorrect Login Traffic")


        # for k in range(val):
        print("User_" + str(idx) + " logging in")
        response = requests.get(
            'http://localhost:8080/api/user/login?username=username_' + str(idx) + '&password=password_' + str(idx))
        print('Status code:' + str(response.status_code))
        idx += 1
        corReq += 1

        with open("loginData.txt", "a") as file:
            file.write("\nUser_" + str(idx) + " logging in")

        time.sleep(1)

    
async def batch_login():
    tasks = []
    print("\n\nBatch Login Attack")
    conn = aiohttp.TCPConnector(limit=200)
    async with aiohttp.ClientSession(connector=conn) as session:

        # prepare a batch login of 100 times
        for i in range(1, 100):
            task = asyncio.ensure_future(login_user(session, url, str(i)))
            tasks.append(task)
            resp = await asyncio.gather(*tasks)
            print("User_"+str(i) + " login")
        resps = await asyncio.gather(*tasks)
        print(str(len(resps)))

if __name__ == '__main__':

    # address
    addr = "http://localhost:8080/wrk2-api/user/register"
    login_addr = "http://localhost:8080/api/user/login"

    start_time = time.time()

    # register normal users throughout data traffic
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(normal_registration(addr))
    loop.run_until_complete(future)
    
    # register normal users throughout data traffic
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(normal_login(login_addr))
    loop.run_until_complete(future)

    # seed a batch registration of bot accounts
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(batch_login(addr))
    loop.run_until_complete(future)

    final_time = time.time()
    print("Total Time: " + str(round(final_time - start_time)) + " seconds")
