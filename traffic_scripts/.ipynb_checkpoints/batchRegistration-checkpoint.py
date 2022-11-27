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

if __name__ == '__main__':
    url = "http://localhost:8080/wrk2-api/user/register"
    loop = asyncio.get_event_loop()
#     future = asyncio.ensure_future(createSession(url))
#     loop.run_until_complete(future)