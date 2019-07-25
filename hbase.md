In [2]: import happybase                                                                                                                                                        

In [3]: conn = happybase.Connection(host="10.9.75.202",autoconnect=False)                                                                                                       

In [4]: conn                                                                                                                                                                    
Out[4]: <happybase.connection.Connection at 0x7f170bdcf4a8>
In [7]: table = happybase.Table(name='midas_offline',connection=conn)                                                                                                           

In [8]: table                                                                                                                                                                   
Out[8]: <happybase.table.Table name='midas_offline'>

In [9]: filter_str = """RowFilter (=, 'substring:{}')"""                                                                                                                        

In [10]: request = ['2019-06-23']                                                               In [23]: filter_str.format(request)                                                                                                                                             
Out[23]: "RowFilter (=, 'substring:['2019-06-23']')"                                                              

In [11]: ooo = conn.open()

In [29]: for key,value in table.scan(filter=filter_str.format(request2),batch_size=1,): 
    ...:     for k,v in value.items(): 
    ...:         k2= k.decode("utf8").split(":")[1] 
    ...:         v2 = v.decode("utf8") 
    ...:         print(k2,v2) 
    ...:         break 