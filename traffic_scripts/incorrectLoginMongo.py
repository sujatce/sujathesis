import requests
import time

print("Not finding registered Users in Mongo")

# wait 7 seconds
time.sleep(7)
# idx = 1
for i in range(1, 50):
    print("User_" + str(i) + " attempting to log in")
    response = requests.get('http://localhost:8080/api/user/login?username=username_' + str(i) + '&password=vol397Chasm')
    print('Status code:' + str(response.status_code))

# response = requests.get('http://localhost:8080/api/user/login?username=username_' + str(i) + '&password=vol397Chasm')
#     for i in range(42, 73):
#     print("User " + str(i) + " logging in")
#     response = requests.get('http://192.168.1.15:8080/api/user/login?username=username_' + str(i) + '&password=vol397Chasm')
#     print('Status code:' + str(response.status_code))
