import numpy as np 


vv = [[1, 3, 5], [0], [7,9], [0],[11]]
vv2 = [[1, 3, 5], [0], [7,9], [0],[11]]
def test(vv):

    tmp=[]
    for i in range(len(vv)-1,-1,-1): 
        #print(i) 
        if vv[i][0] !=0:
            tmp1 =vv[i] 
            continue 
        elif vv[i][0] ==0: 
            # if i==0: 
            #     for j in range(len(vv)-1-i,-1,-1) :
            #         if vv[j][0]!=0: 
            #             tmp =vv[j]
            #             break 
            #     vv[i] = tmp
            
            # if i == (len(vv)-1):
            #     for j in range(len(vv)) :
            #         if vv[j][0]!=0: 
            #             tmp =vv[j]
            #             break 
            #     vv[i] = tmp
            # else:            
            for j in range(i-1,-1,-1) :
                if vv[j][0]!=0: 
                    tmp =vv[j]
                    break 
                elif j>0:
                    continue
                elif j ==0:
                    for k in range(j,i+1):
                        vv[k]=tmp1
                    return vv
            vv[i] = tmp
                    
    return vv
v1 = [[1, 3, 5], [0], [7,9], [0],[11]]
v2 = [[0], [0], [7,9], [0],[11]]
v3 = [[0], [0], [0], [0],[11]]
v4 = [[1, 3, 5], [0], [7,9], [0],[0]]
v5 = [[0], [0], [7,9], [0],[0]]


print(test(v1),'\n')
print(  test(v2),'\n')
print(      test(v3),'\n')
print(test(v4),'\n')
print(test(v5),'\n',)

    [[1, 3, 5], [1, 3, 5], [7, 9], [7, 9], [11]] 

[[7, 9], [7, 9], [7, 9], [7, 9], [11]] 

[[11], [11], [11], [11], [11]] 

[[1, 3, 5], [1, 3, 5], [7, 9], [7, 9], [7, 9]] 

[[7, 9], [7, 9], [7, 9], [7, 9], [7, 9]]