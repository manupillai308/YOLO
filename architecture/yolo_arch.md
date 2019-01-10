Name | Filters | Outputs
--- | --- | --- 
Conv 1 | 7 x 7 x 64, stride=2 | 224 x 224 x 64
Max Pool 1 | 2 x 2, stride=2        | 112 x 112 x 64    
Conv 2     | 3 x 3 x 192            | 112 x 112 x 192   
Max Pool 2 | 2 x 2, stride=2        | 56 x 56 x 192     
Conv 3     | 1 x 1 x 128            | 56 x 56 x 128     
Conv 4     | 3 x 3 x 256            | 56 x 56 x 256     
Conv 5     | 1 x 1 x 256            | 56 x 56 x 256     
Conv 6     | 1 x 1 x 512            | 56 x 56 x 512     
Max Pool 3 | 2 x 2, stride=2        | 28 x 28 x 512     
Conv 7     | 1 x 1 x 256            | 28 x 28 x 256     
Conv 8     | 3 x 3 x 512            | 28 x 28 x 512     
Conv 9     | 1 x 1 x 256            | 28 x 28 x 256     
Conv 10    | 3 x 3 x 512            | 28 x 28 x 512     
Conv 11    | 1 x 1 x 256            | 28 x 28 x 256     
Conv 12    | 3 x 3 x 512            | 28 x 28 x 512     
Conv 13    | 1 x 1 x 256            | 28 x 28 x 256     
Conv 14    | 3 x 3 x 512            | 28 x 28 x 512     
Conv 15    | 1 x 1 x 512            | 28 x 28 x 512     
Conv 16    | 3 x 3 x 1024           | 28 x 28 x 1024    
Max Pool 4 | 2 x 2, stride=2        | 14 x 14 x 1024    
Conv 17    | 1 x 1 x 512            | 14 x 14 x 512    
Conv 18    | 3 x 3 x 1024           | 14 x 14 x 1024    
Conv 19    | 1 x 1 x 512            | 14 x 14 x 512     
Conv 20    | 3 x 3 x 1024           | 14 x 14 x 1024    
Conv 21    | 3 x 3 x 1024           | 14 x 14 x 1024    
Conv 22    | 3 x 3 x 1024, stride=2 | 7 x 7 x 1024      
Conv 23    | 3 x 3 x 1024           | 7 x 7 x 1024     
Conv 24    | 3 x 3 x 1024           | 7 x 7 x 1024      
FC 1       | -                      | 4096              
FC 2       | -                      | 7 x 7 x 30 (1470) 
