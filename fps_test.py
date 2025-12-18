# -- coding: utf-8 --

import sys
import os
import time
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
# 导入相机参数结构体
from CameraParams_header import *

# 全局变量
g_bExit = False
g_frame_count = 0
g_start_time = 0

# 图像回调函数 - 极简版本，只计数
def image_callback(pstFrame, pUser, bAutoFree):
    global g_frame_count, g_start_time
    stFrame = cast(pstFrame, POINTER(MV_FRAME_OUT)).contents
    
    if stFrame.pBufAddr:
        try:
            g_frame_count += 1
            
            # 每秒打印一次帧率
            current_time = time.time()
            if current_time - g_start_time >= 1.0:
                fps = g_frame_count / (current_time - g_start_time)
                print(f"当前帧率: {fps:.2f} fps")
                # 重置计数器
                g_frame_count = 0
                g_start_time = current_time
                    
        except Exception as e:
            print(f"图像处理错误: {e}")

# 设置相机参数以提高帧率
def set_camera_params(cam, device_info):
    try:
        # 获取并显示当前像素格式
        print("检查像素格式...")
        stEnumValue = MVCC_ENUMVALUE()
        ret = cam.MV_CC_GetEnumValue("PixelFormat", stEnumValue)
        if ret == 0:
            current_format = stEnumValue.nCurValue
            print(f"当前像素格式: {current_format}")
            
            # 检查是否已经是8位Bayer格式
            if current_format in [PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerGR8, 
                                PixelType_Gvsp_BayerBG8, PixelType_Gvsp_BayerGB8]:
                print("当前已经是8位Bayer格式，无需转换")
            else:
                # 尝试设置为BayerRG8格式（官网推荐的格式）
                print("尝试设置为BayerRG8格式...")
                ret = cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BayerRG8)
                if ret == 0:
                    print("成功设置为BayerRG8格式")
                else:
                    print(f"设置像素格式失败! 错误码: 0x{ret:x}")
        
        # 获取并设置分辨率
        print("\n检查分辨率...")
        stIntValue = MVCC_INTVALUE()
        
        # 获取当前宽度
        ret = cam.MV_CC_GetIntValue("Width", stIntValue)
        if ret == 0:
            current_width = stIntValue.nCurValue
            print(f"当前宽度: {current_width} 像素")
        
        # 获取当前高度
        ret = cam.MV_CC_GetIntValue("Height", stIntValue)
        if ret == 0:
            current_height = stIntValue.nCurValue
            print(f"当前高度: {current_height} 像素")
        
        # 设置较低的分辨率（将原始分辨率减半）
        new_width = current_width // 2
        new_height = current_height // 2
        
        print(f"\n尝试设置分辨率为 {new_width}x{new_height}...")
        
        # 设置宽度
        ret = cam.MV_CC_SetIntValue("Width", new_width)
        if ret == 0:
            print(f"成功设置宽度为: {new_width} 像素")
        else:
            print(f"设置宽度失败! 错误码: 0x{ret:x}")
        
        # 设置高度
        ret = cam.MV_CC_SetIntValue("Height", new_height)
        if ret == 0:
            print(f"成功设置高度为: {new_height} 像素")
        else:
            print(f"设置高度失败! 错误码: 0x{ret:x}")
        
        # 根据官网信息直接设置帧率为19.1 fps
        print("\n设置帧率...")
        target_fps = 19.1  # 官网标称的最大帧率
        ret = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", target_fps)
        if ret == 0:
            print(f"成功设置帧率为: {target_fps:.1f} fps")
        else:
            print(f"设置帧率失败! 错误码: 0x{ret:x}")
            
            # 尝试使用另一个可能的帧率参数
            print("尝试使用另一个帧率参数...")
            ret = cam.MV_CC_SetFloatValue("AcquisitionFrameRateAbs", target_fps)
            if ret == 0:
                print(f"成功设置AcquisitionFrameRateAbs为: {target_fps:.1f} fps")
            else:
                print(f"设置AcquisitionFrameRateAbs失败! 错误码: 0x{ret:x}")
        
        # 关闭自动曝光和自动增益
        print("\n关闭自动曝光和自动增益...")
        ret = cam.MV_CC_SetEnumValue("ExposureAuto", 0)  # 关闭自动曝光
        if ret != 0:
            print(f"关闭自动曝光失败! 错误码: 0x{ret:x}")
        
        ret = cam.MV_CC_SetEnumValue("GainAuto", 0)  # 关闭自动增益
        if ret != 0:
            print(f"关闭自动增益失败! 错误码: 0x{ret:x}")
        
        # 设置合理的曝光时间（根据19.1 fps计算，单帧最大曝光时间约为52ms）
        print("\n设置曝光时间...")
        exposure_time = 20000  # 20毫秒，远小于52ms的最大限制
        ret = cam.MV_CC_SetFloatValue("ExposureTime", exposure_time)
        if ret == 0:
            print(f"成功设置曝光时间: {exposure_time:.0f} μs")
        else:
            print(f"设置曝光时间失败! 错误码: 0x{ret:x}")
        
        # 设置较低的增益值
        gain_value = 10
        ret = cam.MV_CC_SetFloatValue("Gain", gain_value)
        if ret == 0:
            print(f"成功设置增益值: {gain_value:.0f}")
        else:
            print(f"设置增益值失败! 错误码: 0x{ret:x}")
        
        # 对于网络相机，优化网络设置
        if device_info.nTLayerType == MV_GIGE_DEVICE:
            print("\n优化网络相机设置...")
            
            # 设置最佳网络包大小
            packet_size = cam.MV_CC_GetOptimalPacketSize()
            if packet_size > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
                if ret == 0:
                    print(f"已设置最佳网络包大小: {packet_size}")
                else:
                    print(f"设置网络包大小失败! 错误码: 0x{ret:x}")
            
            # 启用数据包重传
            ret = cam.MV_CC_SetEnumValue("GevStreamChannelSelector", 0)
            if ret == 0:
                cam.MV_CC_SetBoolValue("GevSCBWR", True)  # 启用带宽预留
                cam.MV_CC_SetIntValue("GevSCPD", 50000)    # 设置流延迟
                print("已启用网络流优化设置")
        
        # 关闭不需要的功能以提高性能
        print("\n关闭不必要的图像处理功能...")
        cam.MV_CC_SetEnumValue("GammaEnable", 0)  # 关闭伽马校正
        cam.MV_CC_SetEnumValue("SharpenEnable", 0)  # 关闭锐化
        cam.MV_CC_SetEnumValue("DenoiseEnable", 0)  # 关闭降噪
        
    except Exception as e:
        print(f"设置相机参数时发生错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    global g_bExit, g_frame_count, g_start_time
    
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
        
        print(f"找到 {device_list.nDeviceNum} 个设备")
        
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
        
        # 设置相机参数
        set_camera_params(cam, device_info)
        
        # 设置触发模式为OFF（连续采集）
        print("\n设置触发模式为连续采集...")
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
        
        # 初始化帧率计数
        g_frame_count = 0
        g_start_time = time.time()
        
        # 开始取流
        print("\n开始取流...")
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"开始取流失败! 错误码: 0x{ret:x}")
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            return
        
        print("\n开始计数帧率，按 Ctrl+C 退出...")
        
        # 运行30秒
        time.sleep(30)
        
        # 停止取流
        print("\n停止取流...")
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
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 反初始化SDK
        print("反初始化SDK...")
        MvCamera.MV_CC_Finalize()
        
        print("程序结束")

if __name__ == "__main__":
    main()