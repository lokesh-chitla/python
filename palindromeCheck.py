# Palindrome or not checking #

import sys
a = 989
tmp = []
count = 0

while(True):
    tmp.append(a%10)
    a = int(a/10)
    if a == 0:
        break;
    count += 1

a =0;

while(a <= count):
    if tmp[a] != tmp[count-a]:
        print("Oh, Not a Pallindromic!")
        sys.exit(1)
    a += 1

print("Yes, Its Pallindromic")
