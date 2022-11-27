import requests

response = requests.get('http://localhost:8080/wrk2-api/user-timeline/read?user_id=488&start=0&stop=100')
print('Status code:' + str(response.status_code))
