from random import randint


liste = [randint(1, 100) for i in range(20)]
pair = []
impair = []
for element in liste:
    if element % 2 == 0:
        pair.append(element)
    else:
        impair.append(element)
pair.sort(reverse=True)
impair.sort(reverse=True)

sumpair = 0
sumimpair = 0
for i in range(4):
    sumpair += pair[i]
    sumimpair += impair[i]

print(f"somme des 5 plus grand pair {sumpair} | somme des 5 plus grand impair {sumimpair}")
print("pair:")
print(pair)
print("impair:")
print(impair)