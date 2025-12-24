# -- coding: utf-8 --
"""
香橙派海康威视相机流媒体显示程序
适用于香橙派平台的相机驱动程序
"""

import sys
import os
import time
import cv2
import numpy as np
from ctypes import *
import collections
import threading

# 香橙派MVS SDK路径设置 - 使用项目中的Python_for_arm目录（注意大小写）
current_dir = os.path.dirname(os.path.abspath(__file__))

# 项目中的ARM版本MVS SDK路径 - 根据您提供的路径信息
mvs_sdk_path = os.path.join(current_dir, 'Python_for_arm', 'MvImport')
if mvs_sdk_path not in sys.path:
    sys.path.insert(0, mvs_sdk_path)
    print(f"添加MVS SDK路径: {mvs_sdk_path}")

# 导入MVS模块
try:
    from MvCameraControl_class import *
    from PixelType_header import *
    print("MVS SDK导入成功")
except ImportError as e:
    print(f"MVS SDK导入失败: {e}")
    print("请确保项目中包含Python_for_arm目录，其中包含MVS SDK的ARM版本")
    print("目录结构应类似: ")
    print("  project/")
    print("   ├── Python_for_arm/")  # 注意：目录名是Python_for_arm（大写P）
    print("   │   └── MvImport/")
    print("   │       ├── MvCameraControl_class.py")
    print("   │       ├── PixelType_header.py")
    print("   │       └── 其他MVS SDK文件...")
    print("   └── camera_stream_for_orangePi.py")
    sys.exit(1)

# 香橙派环境下的CUDA可用性检查（香橙派通常不使用CUDA）
try:
    # 香橙派通常不使用CUDA，而是使用NPU或其他加速器
    import numpy as np
    # 检查是否有硬件加速可用
    import subprocess
    
    # 检查是否有OpenCL可用
    try:
        import pyopencl as cl
        OPENCL_AVAILABLE = True
        print("OpenCL可用")
    except ImportError:
        OPENCL_AVAILABLE = False
        print("OpenCL不可用")
    
    # 检查OpenCV是否支持硬件加速
    print(f"OpenCV版本: {cv2.__version__}")
    print(f"OpenCV OpenCL支持: {cv2.ocl.haveOpenCL()}")
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        print("已启用OpenCV OpenCL加速")
        
except Exception as e:
    print(f"硬件加速检查失败: {e}")

# 香橙派全局变量
g_bExit = False
# 减少队列长度以节省内存
g_frame_queue = collections.deque(maxlen=3)  # 香橙派内存较小，减少缓存
g_target_fps = 30  # 香橙派处理能力有限，降低目标帧率
g_target_width = 1920  # 香橙派降低分辨率以提高性能
g_target_height = 1080  # 香橙派降低分辨率以提高性能
g_bayer_format = None

# 香橙派图像回调函数
def image_callback(pstFrame, pUser, bAutoFree):
    global g_frame_queue, g_bayer_format
    stFrame = cast(pstFrame, POINTER(MV_FRAME_OUT)).contents
    
    if stFrame.pBufAddr:
        try:
            # 获取图像基本信息
            width = stFrame.stFrameInfo.nWidth
            height = stFrame.stFrameInfo.nHeight
            pixel_type = stFrame.stFrameInfo.enPixelType
            frame_len = stFrame.stFrameInfo.nFrameLen
            
            # 调试信息（仅在首次执行时打印）
            if not hasattr(image_callback, "first_run"):
                print(f"收到图像 - 宽: {width}, 高: {height}, 像素格式: {pixel_type}, 帧长度: {frame_len}")
                image_callback.first_run = True
            
            processed_frame = None
            
            # 根据像素格式处理图像 - 针对香橙派优化
            if pixel_type == PixelType_Gvsp_Mono8:
                # 单通道8位灰度图像（最快）
                processed_frame = np.ctypeslib.as_array(stFrame.pBufAddr, (height, width))
            elif pixel_type in [PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed]:
                # RGB/BGR图像
                processed_frame = np.ctypeslib.as_array(stFrame.pBufAddr, (height, width, 3))
            elif pixel_type in [PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8,
                              PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8]:
                # 8位Bayer格式 - 香橙派优化处理
                bayer_data = np.ctypeslib.as_array(stFrame.pBufAddr, (height, width))
                if pixel_type == PixelType_Gvsp_BayerGR8:
                    conversion_code = cv2.COLOR_BAYER_GR2RGB
                elif pixel_type == PixelType_Gvsp_BayerRG8:
                    conversion_code = cv2.COLOR_BAYER_RG2RGB
                elif pixel_type == PixelType_Gvsp_BayerGB8:
                    conversion_code = cv2.COLOR_BAYER_GB2RGB
                elif pixel_type == PixelType_Gvsp_BayerBG8:
                    conversion_code = cv2.COLOR_BAYER_BG2RGB
                
                # 使用OpenCV的Bayer转换 - 香橙派优化
                processed_frame = cv2.cvtColor(bayer_data, conversion_code)
            else:
                # 其他格式，尝试快速转换 - 香橙派兼容
                try:
                    # 直接获取原始数据
                    raw_data = np.ctypeslib.as_array(stFrame.pBufAddr, (frame_len,))
                    # 尝试转换为8位
                    if pixel_type in [PixelType_Gvsp_Mono10_Packed, PixelType_Gvsp_Mono12_Packed]:
                        # 压缩格式，简单处理
                        processed_frame = (raw_data[:height*width] >> 4).astype(np.uint8)
                        processed_frame = processed_frame.reshape((height, width))
                    else:
                        # 其他格式，尝试直接重塑
                        processed_frame = raw_data[:height*width].reshape((height, width))
                        if processed_frame.dtype != np.uint8:
                            processed_frame = (processed_frame >> 4).astype(np.uint8)
                except Exception as e:
                    print(f"无法处理像素格式: {pixel_type}, 错误: {e}")
                    return
            
            # 将图像复制到队列（仅保留最新的图像）
            if processed_frame is not None and processed_frame.size > 0:
                # 香橙派内存管理优化
                g_frame_queue.append(processed_frame.copy())
                    
        except Exception as e:
            print(f"图像处理错误: {e}")
            import traceback
            traceback.print_exc()

# 香橙派显示图像线程
def display_thread():
    global g_bExit, g_frame_queue
    
    print("开始显示视频流，按 'q' 键退出")
    
    # 帧率计算变量
    fps_count = 0
    fps_start_time = time.time()
    current_fps = 0
    
    while not g_bExit:
        if g_frame_queue:
            # 记录开始时间
            start_time = time.time()
            
            # 从队列获取最新图像
            frame = g_frame_queue[-1]  # 获取最新帧，不删除
            g_frame_queue.clear()  # 清空队列，只保留最新帧
            
            # 香橙派优化的图像缩放 - 使用更高效的算法
            # 首先检查是否有OpenCL可用
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                resized_frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_LINEAR)
            else:
                # 使用标准OpenCV缩放
                resized_frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_LINEAR)
            
            # 计算帧率
            fps_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time >= 1.0:
                current_fps = fps_count / elapsed_time
                fps_count = 0
                fps_start_time = time.time()
            
            # 在图像上显示帧率
            fps_text = f"FPS: {current_fps:.2f}"
            if len(resized_frame.shape) == 2:  # 灰度图像
                # 创建彩色图像用于显示帧率
                display_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2BGR)
            else:
                display_frame = resized_frame.copy()
            
            # 添加帧率文本
            cv2.putText(display_frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 显示图像
            cv2.imshow('OrangePi Hikvision Camera Stream', display_frame)
            
            # 检查退出按键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                g_bExit = True
                break
        else:
            # 队列为空时短暂等待
            time.sleep(0.001)
    
    # 释放窗口
    cv2.destroyAllWindows()

# 香橙派相机参数设置 - 针对ARM处理器优化
def set_camera_params(cam, device_info):
    global g_target_fps, g_target_width, g_target_height
    
    try:
        print("开始设置香橙派相机参数...")
        
        # 设置较低的分辨率以适应香橙派性能
        print(f"设置目标分辨率为 {g_target_width}x{g_target_height}...")
        
        # 设置宽度
        ret = cam.MV_CC_SetIntValue("Width", g_target_width)
        if ret != 0:
            print(f"设置宽度失败! 尝试默认值")
            # 尝试设置为较小的分辨率
            ret = cam.MV_CC_SetIntValue("Width", 640)
            if ret == 0:
                print("已设置为640宽度")
            else:
                print(f"设置640宽度也失败，错误码: 0x{ret:x}")
        
        # 设置高度
        ret = cam.MV_CC_SetIntValue("Height", g_target_height)
        if ret != 0:
            print(f"设置高度失败! 尝试默认值")
            # 尝试设置为较小的分辨率
            ret = cam.MV_CC_SetIntValue("Height", 480)
            if ret == 0:
                print("已设置为480高度")
            else:
                print(f"设置480高度也失败，错误码: 0x{ret:x}")
        
        # 验证分辨率设置
        width_param = MVCC_INTVALUE()
        height_param = MVCC_INTVALUE()
        if (cam.MV_CC_GetIntValue("Width", width_param) == 0 and 
            cam.MV_CC_GetIntValue("Height", height_param) == 0):
            print(f"分辨率设置成功: {width_param.nCurValue}x{height_param.nCurValue}")
        
        # 设置较低的帧率以适应香橙派性能
        print(f"设置目标帧率为 {g_target_fps} fps...")
        ret = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", g_target_fps)
        if ret != 0:
            print(f"设置帧率失败! 错误码: 0x{ret:x}")
            # 尝试设置为更低的帧率
            ret = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 10)
            if ret == 0:
                print("已设置为10fps")
        
        # 为香橙派选择最简单的像素格式
        print("设置香橙派优化的像素格式...")
        pixel_format_success = False
        
        # 优先尝试灰度格式以提高性能
        ret = cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8)
        if ret == 0:
            print("已将像素格式设置为8位灰度（香橙派优化）")
            pixel_format_success = True
        else:
            print(f"设置灰度格式失败，错误码: 0x{ret:x}")
            # 如果灰度不行，尝试RGB
            ret = cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed)
            if ret == 0:
                print("已将像素格式设置为8位RGB")
                pixel_format_success = True
            else:
                print(f"设置RGB格式也失败，错误码: 0x{ret:x}")
        
        if not pixel_format_success:
            print("警告: 无法设置优化的像素格式，将使用相机默认格式")
        
        # 香橙派网络优化
        if device_info.nTLayerType == MV_GIGE_DEVICE:
            print("优化香橙派网络相机设置...")
            # 设置较小的网络包大小以适应网络环境
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", 1500)  # 标准以太网包大小
            if ret == 0:
                print("已设置网络包大小为1500")
            else:
                print(f"设置网络包大小失败，错误码: 0x{ret:x}")
        
        # 香橙派性能优化设置
        print("设置香橙派性能优化参数...")
        # 关闭自动曝光和自动增益
        ret = cam.MV_CC_SetEnumValue("ExposureAuto", 1)  # 关闭自动曝光
        if ret != 0:
            print(f"关闭自动曝光失败，错误码: 0x{ret:x}")
        
        ret = cam.MV_CC_SetEnumValue("GainAuto", 1)     # 关闭自动增益
        if ret != 0:
            print(f"关闭自动增益失败，错误码: 0x{ret:x}")
        
        # 设置合理的曝光和增益值
        ret = cam.MV_CC_SetFloatValue("ExposureTime", 10000)  # 10毫秒
        if ret != 0:
            print(f"设置曝光时间失败，错误码: 0x{ret:x}")
        
        ret = cam.MV_CC_SetFloatValue("Gain", 15)  # 中等增益
        if ret != 0:
            print(f"设置增益失败，错误码: 0x{ret:x}")
        
        # 关闭占用资源的功能
        cam.MV_CC_SetEnumValue("GammaEnable", 0)      # 关闭伽马校正
        cam.MV_CC_SetEnumValue("SharpenEnable", 0)    # 关闭锐化
        cam.MV_CC_SetEnumValue("DenoiseEnable", 0)    # 关闭降噪
        cam.MV_CC_SetEnumValue("ColorTransformationEnable", 0)  # 关闭颜色转换
        
        print("香橙派相机参数设置完成")
        
    except Exception as e:
        print(f"设置香橙派相机参数时发生错误: {e}")
        import traceback
        traceback.print_exc()

def check_system_requirements():
    """检查香橙派系统要求"""
    print("检查香橙派系统要求...")
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查OpenCV版本和功能
    print(f"OpenCV版本: {cv2.__version__}")
    
    # 检查系统内存
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"系统内存: {memory.total / (1024**3):.2f} GB")
        print(f"可用内存: {memory.available / (1024**3):.2f} GB")
    except ImportError:
        print("无法获取系统内存信息（可选依赖psutil未安装）")
    
    # 检查是否存在MVS SDK目录
    mvs_sdk_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Python_for_arm', 'MvImport')
    if os.path.exists(mvs_sdk_path):
        print(f"MVS SDK目录存在: {mvs_sdk_path}")
        sdk_files = os.listdir(mvs_sdk_path)
        print(f"SDK文件: {sdk_files[:10]}...")  # 显示前10个文件
    else:
        print(f"警告: MVS SDK目录不存在: {mvs_sdk_path}")
    
    # 检查是否有USB权限
    print("检查设备权限...")
    try:
        usb_devices = os.popen('lsusb').read()
        print(f"USB设备数量: {len(usb_devices.split(chr(10)))-1}")
    except:
        print("无法检查USB设备")
    
    return True

def main():
    global g_bExit, g_bayer_format
    
    print("香橙派海康威视相机启动程序")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本目录: {os.path.dirname(os.path.abspath(__file__))}")
    
    # 检查系统要求
    if not check_system_requirements():
        print("系统要求检查失败")
        return
    
    try:
        # 初始化SDK
        print("初始化MVS SDK...")
        ret = MvCamera.MV_CC_Initialize()
        if ret != 0:
            print(f"SDK初始化失败! 错误码: 0x{ret:x}")
            return
        
        # 获取SDK版本
        try:
            sdk_version = MvCamera.MV_CC_GetSDKVersion()
            print(f"SDK版本: 0x{sdk_version:x}")
        except:
            print("无法获取SDK版本")
        
        # 枚举设备
        print("枚举相机设备...")
        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer_type = (MV_GIGE_DEVICE | MV_USB_DEVICE)  # 香橙派主要支持的设备类型
        
        ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        if ret != 0:
            print(f"枚举设备失败! 错误码: 0x{ret:x}")
            return
        
        if device_list.nDeviceNum == 0:
            print("未找到海康威视相机设备!")
            print("请检查:")
            print("1. 相机是否正确连接")
            print("2. MVS SDK的Python_for_arm目录是否存在于项目根目录")
            print("3. 设备权限是否正确设置")
            print("4. 相机驱动是否正确安装")
            return
        
        print(f"找到 {device_list.nDeviceNum} 个设备:")
        
        # 显示设备信息
        for i in range(device_list.nDeviceNum):
            device_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            
            if device_info.nTLayerType == MV_GIGE_DEVICE:
                print(f"\n设备 {i}: 网络相机")
                try:
                    model_name = string_at(device_info.SpecialInfo.stGigEInfo.chModelName, 32).decode('utf-8', errors='ignore').strip('\x00')
                    print(f"  型号: {model_name}")
                    ip_parts = [
                        (device_info.SpecialInfo.stGigEInfo.nCurrentIp >> 24) & 0xFF,
                        (device_info.SpecialInfo.stGigEInfo.nCurrentIp >> 16) & 0xFF,
                        (device_info.SpecialInfo.stGigEInfo.nCurrentIp >> 8) & 0xFF,
                        (device_info.SpecialInfo.stGigEInfo.nCurrentIp >> 0) & 0xFF
                    ]
                    ip_address = ".".join(map(str, ip_parts))
                    print(f"  IP地址: {ip_address}")
                except:
                    print("  无法获取详细信息")
            elif device_info.nTLayerType == MV_USB_DEVICE:
                print(f"\n设备 {i}: USB相机")
                try:
                    model_name = string_at(device_info.SpecialInfo.stUsb3VInfo.chModelName, 32).decode('utf-8', errors='ignore').strip('\x00')
                    serial_number = string_at(device_info.SpecialInfo.stUsb3VInfo.chSerialNumber, 32).decode('utf-8', errors='ignore').strip('\x00')
                    print(f"  型号: {model_name}")
                    print(f"  序列号: {serial_number}")
                except:
                    print("  无法获取详细信息")
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
            # 尝试使用其他访问模式
            ret = cam.MV_CC_OpenDevice(MV_ACCESS_ReadWrite, 0)
            if ret != 0:
                print(f"使用读写模式打开设备也失败! 错误码: 0x{ret:x}")
                cam.MV_CC_DestroyHandle()
                return
            else:
                print("使用读写模式成功打开设备")
        
        # 为香橙派设置相机参数
        set_camera_params(cam, device_info)
        
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
        
        print("相机启动成功，启动显示线程...")
        
        # 启动显示线程
        display_thread_handle = threading.Thread(target=display_thread)
        display_thread_handle.daemon = True  # 设置为守护线程
        display_thread_handle.start()
        
        print("按 'q' 键在显示窗口中退出程序")
        
        # 等待显示线程结束
        display_thread_handle.join()
        
        # 停止取流
        print("停止取流...")
        g_bExit = True
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
        print("\n用户中断程序 (Ctrl+C)")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 反初始化SDK
        print("反初始化SDK...")
        try:
            MvCamera.MV_CC_Finalize()
        except:
            pass
        
        print("香橙派相机程序结束")

if __name__ == "__main__":
    main()