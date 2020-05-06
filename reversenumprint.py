#prints given number in revers order, ex a = 123, prints 321 #

a = 42699
while(True):
    print(a%10, end="")
    a = int(a/10)
    if a == 0:
        break;
