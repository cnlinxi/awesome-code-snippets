# when you want to traverse two list and add index at the same time
# you can use 'enumerate' & 'zip'
# Attention: element in list must use brackets
for index, (a, b) in enumerate(zip([1, 2, 3], [4, 5, 6])):
    print('index:{0}, a:{1}, b:{2}'.format(index, a, b))
