import aiohttp
import asyncio

async def fetch(session, url, user):
    
    payload = {
        'first_name': 'first_name_' + user, 
        'last_name': 'last_name_' + user, 
        'username': 'username_' + user, 
        'password': 'password_' + user, 
        'user_id': user
    }
    
    async with session.post(url, data=payload) as response:
        return await response.text()
    

async def createSession(url):
    
    tasks = []
    conn = aiohttp.TCPConnector(limit=200)
    
    print("Creating Client Session")
    async with aiohttp.ClientSession(connector=conn) as session:
        
        idx = 0
        for i in range(101, 150):
            task = asyncio.ensure_future(fetch(session, url, str(i)))
            tasks.append(task)
            resp = await asyncio.gather(*tasks)
            print(str(idx) + " - " + str(resp))
            
            idx += 1
        resp = await asyncio.gather(*tasks)
        print("Users registered")
    
if __name__ == '__main__':
    url = "http://localhost:8080/wrk2-api/user/register"
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(createSession(url))
    loop.run_until_complete(future)