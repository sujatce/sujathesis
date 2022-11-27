import requests
import time
import random

startTime = time.time()
idx = 1

corReq = 0
incReq = 0

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

# time.sleep(10)
# brute force attack
time.sleep(1)
print("\n\nBrute Force Attack")
for j in range(1, 92):
    print("User_" + str(1) + " attempting to log in")
    response = requests.get(
        'http://localhost:8080/api/user/login?username=username_1&password=vol' + str(j) + '97Chasm')
    print('Status code:' + str(response.status_code))

print("Correct Requests: " + str(corReq))
print("Incorrect Requests: " + str(incReq))

with open("loginData.txt", "a") as file:
    file.write("\n\nCorrect Logins: " + str(corReq))
    file.write("\nIncorrect Login: " + str(incReq))

endTime = time.time()
print("Login Traffic - Difference in seconds: " + str(round(endTime - startTime)))


# password guess attack in final second
# six incorrect logging

# three incorrect logins = one every 8 seconds

# 10% of logins are incorrect

# password guess attack in final two second
# six incorrect logging
