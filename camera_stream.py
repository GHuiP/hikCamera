# -- coding: utf-8 --

import sys
import os
import time
import cv2
import numpy as np
from ctypes import *

# 设置本地MVS SDK路径
current_dir = os.path.dirname(os.path.abspath(__file__))
mv_import_path = os.path.join(current_dir, 'Python', 'MvImport')
if mv_import_path not in sys.path:
    sys.path.append(mv_import_path)

# 导入MVS模块
from MvCameraControl_class import *
# 导入像素格式定义
from PixelType_header import *

# 全局变量
g_bExit = False
g_frame_queue = []  # 用于存储图像帧的队列

# 图像回调函数
def image_callback(pstFrame, pUser, bAutoFree):
    global g_frame_queue
    stFrame = cast(pstFrame, POINTER(MV_FRAME_OUT)).contents
    
    if stFrame.pBufAddr:
        try:
            # 获取图像基本信息
            width = stFrame.stFrameInfo.nWidth
            height = stFrame.stFrameInfo.nHeight
            pixel_type = stFrame.stFrameInfo.enPixelType
            frame_len = stFrame.stFrameInfo.nFrameLen
            
            # 调试信息
            print(f"收到图像 - 宽: {width}, 高: {height}, 像素格式: {pixel_type}, 帧长度: {frame_len}")
            
            # 根据像素格式计算通道数
            channels = 1  # 默认单通道
            if pixel_type in [PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed]:
                channels = 3
            elif pixel_type in [PixelType_Gvsp_RGBA8_Packed, PixelType_Gvsp_BGRA8_Packed]:
                channels = 4
            # 添加其他像素格式的处理...
            
            # 计算每像素的字节数
            pixel_bytes = 1  # 默认1字节/像素
            if pixel_type in [PixelType_Gvsp_Mono10, PixelType_Gvsp_Mono10_Packed,
                             PixelType_Gvsp_BayerGR10, PixelType_Gvsp_BayerGR10_Packed,
                             PixelType_Gvsp_RGB10_Packed]:
                pixel_bytes = 2
            elif pixel_type in [PixelType_Gvsp_Mono12, PixelType_Gvsp_Mono12_Packed,
                               PixelType_Gvsp_BayerGR12, PixelType_Gvsp_BayerGR12_Packed,
                               PixelType_Gvsp_RGB12_Packed]:
                pixel_bytes = 2
            elif pixel_type in [PixelType_Gvsp_Mono14, PixelType_Gvsp_Mono16,
                               PixelType_Gvsp_BayerGR16, PixelType_Gvsp_RGB16_Packed]:
                pixel_bytes = 2
            
            # 转换图像数据到numpy数组
            # 先创建一维数组，再根据通道数调整形状
            frame_data = np.ctypeslib.as_array(stFrame.pBufAddr, (frame_len,))
            
            # 根据像素格式和通道数重新调整数据形状
            if pixel_type in [PixelType_Gvsp_Mono8]:
                # 单通道8位灰度图像
                frame_data = frame_data.reshape((height, width))
            elif pixel_type in [PixelType_Gvsp_Mono10_Packed, PixelType_Gvsp_Mono12_Packed]:
                # 单通道10/12位灰度图像（压缩格式）
                # 需要特殊处理，这里简化为8位
                frame_data = frame_data[:height*width].reshape((height, width)).astype(np.uint8)
            elif pixel_type in [PixelType_Gvsp_Mono16]:
                # 单通道16位灰度图像
                frame_data = frame_data.view(np.uint16)[:height*width].reshape((height, width))
            elif pixel_type in [PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed]:
                # 3通道RGB/BGR图像
                frame_data = frame_data[:height*width*3].reshape((height, width, 3))
            elif pixel_type in [PixelType_Gvsp_RGBA8_Packed, PixelType_Gvsp_BGRA8_Packed]:
                # 4通道RGBA/BGRA图像
                frame_data = frame_data[:height*width*4].reshape((height, width, 4))
            elif pixel_type in [PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8,
                              PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8]:
                # Bayer格式图像，转换为RGB
                bayer_data = frame_data[:height*width].reshape((height, width))
                frame_data = cv2.cvtColor(bayer_data, cv2.COLOR_BAYER_GR2RGB)
            else:
                # 其他格式，尝试默认转换
                print(f"未处理的像素格式: {pixel_type}，使用默认转换")
                frame_data = frame_data[:height*width*channels].reshape((height, width, channels))
            
            # 确保数据类型为uint8（OpenCV显示需要）
            if frame_data.dtype != np.uint8:
                # 归一化到0-255
                frame_data = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # 将图像复制到队列
            g_frame_queue.append(frame_data.copy())
            
            # 限制队列大小
            if len(g_frame_queue) > 10:
                g_frame_queue.pop(0)
                
        except Exception as e:
            print(f"图像处理错误: {e}")

# 显示图像线程
def display_thread():
    global g_bExit
    print("开始显示视频流，按 'q' 键退出")
    
    while not g_bExit:
        if g_frame_queue:
            # 从队列获取最新图像
            frame = g_frame_queue.pop()
            
            # 调整图像大小以适应窗口
            resized_frame = cv2.resize(frame, (800, 600))
            
            # 显示图像
            cv2.imshow('Hikvision Camera Stream', resized_frame)
            
            # 检查退出按键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                g_bExit = True
                break
        else:
            # 队列为空时短暂等待
            time.sleep(0.01)
    
    # 释放窗口
    cv2.destroyAllWindows()

def main():
    global g_bExit
    
    try:
        # 初始化SDK
        print("初始化SDK...")
        MvCamera.MV_CC_Initialize()
        
        # 获取SDK版本
        sdk_version = MvCamera.MV_CC_GetSDKVersion()
        print(f"SDK版本: 0x{sdk_version:x}")
        
        # 枚举设备
        print("枚举设备...")
        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer_type = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE | 
                      MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
        
        ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        if ret != 0:
            print(f"枚举设备失败! 错误码: 0x{ret:x}")
            return
        
        if device_list.nDeviceNum == 0:
            print("未找到设备!")
            return
        
        print(f"找到 {device_list.nDeviceNum} 个设备:")
        
        # 显示设备信息
        for i in range(device_list.nDeviceNum):
            device_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            
            if device_info.nTLayerType == MV_GIGE_DEVICE:
                print(f"\n设备 {i}: 网络相机")
                # 修复：使用ctypes.string_at转换c_ubyte_Array到字节字符串
                model_name = string_at(device_info.SpecialInfo.stGigEInfo.chModelName, 32).decode('gbk', errors='ignore').strip('\x00')
                print(f"  型号: {model_name}")
                print(f"  IP地址: {device_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xFF}.{(device_info.SpecialInfo.stGigEInfo.nCurrentIp >> 8) & 0xFF}.{(device_info.SpecialInfo.stGigEInfo.nCurrentIp >> 16) & 0xFF}.{(device_info.SpecialInfo.stGigEInfo.nCurrentIp >> 24) & 0xFF}")
            elif device_info.nTLayerType == MV_USB_DEVICE:
                print(f"\n设备 {i}: USB相机")
                # 修复：使用ctypes.string_at转换c_ubyte_Array到字节字符串
                model_name = string_at(device_info.SpecialInfo.stUsb3VInfo.chModelName, 32).decode('gbk', errors='ignore').strip('\x00')
                serial_number = string_at(device_info.SpecialInfo.stUsb3VInfo.chSerialNumber, 32).decode('gbk', errors='ignore').strip('\x00')
                print(f"  型号: {model_name}")
                print(f"  序列号: {serial_number}")
            else:
                print(f"\n设备 {i}: 其他类型相机")
        
        # 选择第一个设备
        selected_device = 0
        print(f"\n选择设备 {selected_device}")
        
        # 创建相机实例
        print("创建相机实例...")
        cam = MvCamera()
        
        # 创建句柄
        device_info = cast(device_list.pDeviceInfo[selected_device], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = cam.MV_CC_CreateHandle(device_info)
        if ret != 0:
            print(f"创建句柄失败! 错误码: 0x{ret:x}")
            return
        
        # 打开设备
        print("打开设备...")
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print(f"打开设备失败! 错误码: 0x{ret:x}")
            cam.MV_CC_DestroyHandle()
            return
        
        # 设置网络相机最佳包大小
        if device_info.nTLayerType == MV_GIGE_DEVICE:
            packet_size = cam.MV_CC_GetOptimalPacketSize()
            if packet_size > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
                if ret != 0:
                    print(f"设置包大小失败! 错误码: 0x{ret:x}")
        
        # 设置触发模式为OFF（连续采集）
        print("设置触发模式为连续采集...")
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print(f"设置触发模式失败! 错误码: 0x{ret:x}")
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            return
        
        # 注册图像回调函数
        print("注册图像回调函数...")
        fun_ctype = get_platform_functype()
        frame_callback = fun_ctype(None, POINTER(MV_FRAME_OUT), c_void_p, c_bool)(image_callback)
        ret = cam.MV_CC_RegisterImageCallBackEx2(frame_callback, None, True)
        if ret != 0:
            print(f"注册回调失败! 错误码: 0x{ret:x}")
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            return
        
        # 开始取流
        print("开始取流...")
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"开始取流失败! 错误码: 0x{ret:x}")
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            return
        
        # 启动显示线程
        import threading
        display_thread_handle = threading.Thread(target=display_thread)
        display_thread_handle.start()
        
        # 等待显示线程结束
        display_thread_handle.join()
        
        # 停止取流
        print("停止取流...")
        ret = cam.MV_CC_StopGrabbing()
        if ret != 0:
            print(f"停止取流失败! 错误码: 0x{ret:x}")
        
        # 关闭设备
        print("关闭设备...")
        ret = cam.MV_CC_CloseDevice()
        if ret != 0:
            print(f"关闭设备失败! 错误码: 0x{ret:x}")
        
        # 销毁句柄
        print("销毁句柄...")
        ret = cam.MV_CC_DestroyHandle()
        if ret != 0:
            print(f"销毁句柄失败! 错误码: 0x{ret:x}")
        
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 反初始化SDK
        print("反初始化SDK...")
        MvCamera.MV_CC_Finalize()
        
        print("程序结束")

if __name__ == "__main__":
    main()