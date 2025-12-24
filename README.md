# hikCamera

测试海康工业相机

运行

```
python camera_stream.py
```

## Long time test

### for x86 
```
nohup python long_timg_test.py --duration <time/seconds> --no-display > long_test_24hours.log 2>&1 &
```

--duartion <time/seconds> : during time

--no-display : Don't display real time video stream



### for OrangePi 
```
nohup python long_time_test_for_orangePi.py --duration <time/seconds> --no-display > long_test_24hours.log 2>&1 &
```