import aiohttp
import asyncio

async def fetch(session, addr, data):
    print("Posting request")
    async with session.post(addr, data=data) as response:

        return await response.text()
    
    
async def main():
    
    print("Main")
    addr = "http://localhost:8080/wrk2-api/user/register"
    data = {
        'first_name': 'first_name_965', 
        'last_name': 'last_name_965', 
        'username': 'username_965', 
        'password': 'password_965', 
        'user_id': '965'
    }
    
    tasks = []
    async with aiohttp.ClientSession() as session:
        tasks.append(fetch(session, addr, data))
        
        resp = await asyncio.gather(*tasks)
        print(resp)
        print("Request posted")
        
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    
    
