"""
1.使用locals() 内置函数来查看局部命名空间中的内容
2.使用 globals()内置函数来查看全局命名空间中的内容
注意: 如果locals() 没有在函数体中，而是在py文件中，打印出来 的内容和
 globals() 内容相同。
"""
d = 30
 
 
def func(c):
    a = 10
    b = 20
    print(locals())
    # {'b': 20, 'a': 10, 'c': 30}
 
 
func(30)
 
# 下面这两个打出的一样，同时注意红笔，globals()看全局时，局部变量并没显示，它只显示func和d=30                   
print(globals())  
print(locals())
