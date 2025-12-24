# hikCamera

测试海康工业相机

运行

```
python camera_stream.py
```

长时间测试

```
nohup python long_timg_test.py --duration <time/seconds> --no-display > long_test_24hours.log 2>&1 &
```

--duartion <time/seconds> : during time

--no-display : Don't display real time video stream


```
nohup python long_time_test_for_orangePi.py --duration <time/seconds> --no-display > long_test_24hours.log 2>&1 &
```