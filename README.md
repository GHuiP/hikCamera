# hikCamera

## Test HkCamera

RUN
### for x86
```
python camera_stream.py
```
### for OrangePi
```
python camera_stream_for_orangePi.py
```

## Long time test

### for x86 
```
nohup python long_timg_test.py --duration <time/seconds> --no-display > <log_file_name.log> 2>&1 &
```

### for OrangePi 
```
nohup python long_time_test_for_orangePi.py --duration <time/seconds> --no-display > <log_file_name.log> 2>&1 &
```

--duartion <time/seconds> : during time

--no-display : Don't display real time video stream