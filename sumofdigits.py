# Sum of digits in the number, ex: 231 = 2+3+1 = 6 #

a = 42699
sum = 0
while(True):
    sum += (a%10)
    a = int(a/10)
    if a==0:
        break;

print(sum)
