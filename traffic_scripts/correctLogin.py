import requests

# print("Finding 129 Registered Users in Mongo and caching in Memcached")
# idx = 1
# for i in range(1, 130):
#     print("User_" + str(i) + " logging in")
#     response = requests.get('http://localhost:8080/api/user/login?username=username_' + str(i) + '&password=password_' + str(i))
#     print('Status code:' + str(response.status_code))
#     idx += 1

# requests.exceptions.ConnectionError: HTTPConnectionPool(host='192.168.1.15', port=8080): Max retries exceeded with url: /api/user/login?username=username_1&password=password_1 (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f7eb4e28320>: Failed to establish a new connection: [Errno 111] Connection refused',))


#     if idx % 3 == 0:
#         print("User_" + str(i) + " logging in a second time")
#         respns = requests.get('http://localhost:8080/api/user/login?username=username_' + str(i) + '&password=password_' + str(i))
#         print('Status code:' + str(response.status_code))

## Training Data

print("Finding 7 Registered Users in Mongo and caching in Memcached")
idx = 1
for i in range(701, 707):
    print("User_" + str(i) + " logging in")
    response = requests.get('http://localhost:8080/api/user/login?username=username_' + str(i) + '&password=password_' + str(i))
    print('Status code:' + str(response.status_code))
    idx += 1
    
    if idx % 5 == 0:
        print("User_" + str(i) + " logging in a second time")
        respns = requests.get('http://localhost:8080/api/user/login?username=username_' + str(i) + '&password=password_' + str(i))
        print('Status code:' + str(response.status_code))