def nor (lis,min,max):
    norm = []
    for i in lis:
        vas = max - min
        value = i - min
        norm.append(value / vas)
    return norm
money = [70000,60000,52000]
year = [45,44,40]
print(nor(money,min(money),max(money)))
print(nor(year,min(year),max(year)))

