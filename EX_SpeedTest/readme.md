used to test whether the using of deep point conv is fast than Conv
## Note:
1.the code is not for train or test, it is just uesd to show inference time, thus, please ignore the top1 accuracy.


When test the inference time of original mobilenet:
```
python main_moblie.py
```
When test the inference time of traditional CNN with the same structral as moblienet:
```
python main_conv.py
```

## result

the result of code can be found at the 'result' file