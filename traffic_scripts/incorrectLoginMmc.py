import requests

print("Cached Users not in Memcached not logging in")

for i in range(7, 14):
    print("Cached user_" + str(i) + " attempting to log in")
    response = requests.get('http://localhost:8080/api/user/login?username=username_' + str(i) + '&password=vol397Chasm')
    print('Status code:' + str(response.status_code))
    