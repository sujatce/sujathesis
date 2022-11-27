import aiohttp
import asyncio
import time

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


if __name__ == '__main__':

    # address
    addr = "http://localhost:8080/wrk2-api/user/register"

    start_time = time.time()

    # register normal users throughout data traffic
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(normal_registration(addr))
    loop.run_until_complete(future)

    # seed a batch registration of bot accounts
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(batch_registration(addr))
    loop.run_until_complete(future)

    final_time = time.time()
    print("Total Time: " + str(round(final_time - start_time)) + " seconds")
