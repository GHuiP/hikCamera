# hikCamera

测试海康工业相机

运行

```
python camera_stream.py
```

长时间测试

```
nohup python long_timg_test.py --duration 86400 --no-display > long_test_24hours.log 2>&1 &
```

--duartion <time/seconds> : during time

--no-display : Don't display real time video stream
