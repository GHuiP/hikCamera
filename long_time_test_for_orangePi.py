# -- coding: utf-8 --
"""
香橙派海康威视相机长期稳定性测试程序
适用于香橙派平台的相机稳定性测试
"""

import sys
import os
import time
import cv2
import numpy as np
from ctypes import *
import collections
import threading
import logging
import psutil
import traceback
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orange_pi_camera_stability_test.log'),
        logging.StreamHandler()
    ]
)

# 香橙派MVS SDK路径设置 - 使用项目中的Python_for_arm目录（注意大小写）
current_dir = os.path.dirname(os.path.abspath(__file__))

# 项目中的ARM版本MVS SDK路径 - 根据您提供的路径信息
mvs_sdk_path = os.path.join(current_dir, 'Python_for_arm', 'MvImport')
if mvs_sdk_path not in sys.path:
    sys.path.insert(0, mvs_sdk_path)
    logging.info(f"添加MVS SDK路径: {mvs_sdk_path}")

# 导入MVS模块
try:
    from MvCameraControl_class import *
    from PixelType_header import *
    logging.info("MVS SDK导入成功")
except ImportError as e:
    logging.error(f"MVS SDK导入失败: {e}")
    logging.error("请确保项目中包含Python_for_arm目录，其中包含MVS SDK的ARM版本")
    logging.error("目录结构应类似: ")
    logging.error("  project/")
    logging.error("   ├── Python_for_arm/")  # 注意：目录名是Python_for_arm（大写P）
    logging.error("   │   └── MvImport/")
    logging.error("   │       ├── MvCameraControl_class.py")
    logging.error("   │       ├── PixelType_header.py")
    logging.error("   │       └── 其他MVS SDK文件...")
    logging.error("   └── long_timg_test_orange_pi.py")
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
        logging.info("OpenCL可用")
    except ImportError:
        OPENCL_AVAILABLE = False
        logging.info("OpenCL不可用")
    
    # 检查OpenCV是否支持硬件加速
    logging.info(f"OpenCV版本: {cv2.__version__}")
    logging.info(f"OpenCV OpenCL支持: {cv2.ocl.haveOpenCL()}")
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        logging.info("已启用OpenCV OpenCL加速")
        
except Exception as e:
    logging.error(f"硬件加速检查失败: {e}")

# 香橙派全局变量
g_bExit = False
g_frame_queue = collections.deque(maxlen=3)  # 香橙派内存较小，减少缓存
g_target_fps = 30  # 香橙派处理能力有限，降低目标帧率
g_target_width = 1920  # 香橙派降低分辨率以提高性能
g_target_height = 1080  # 香橙派降低分辨率以提高性能
g_bayer_format = None  # 自动检测的Bayer格式

# 稳定性测试统计变量
g_total_frames = 0
g_fps_list = []
g_start_time = None
g_last_frame_time = None
g_memory_usage = []
g_error_count = 0
g_test_duration = 3600  # 默认测试时长：1小时（3600秒）
g_frame_loss_count = 0
g_last_frame_count = 0

# 图像回调函数
def image_callback(pstFrame, pUser, bAutoFree):
    global g_frame_queue, g_bayer_format, g_total_frames, g_last_frame_time, g_frame_loss_count
    
    try:
        stFrame = cast(pstFrame, POINTER(MV_FRAME_OUT)).contents
        
        if stFrame.pBufAddr:
            # 获取图像基本信息
            width = stFrame.stFrameInfo.nWidth
            height = stFrame.stFrameInfo.nHeight
            pixel_type = stFrame.stFrameInfo.enPixelType
            frame_len = stFrame.stFrameInfo.nFrameLen
            
            # 调试信息（仅在首次执行时打印）
            if not hasattr(image_callback, "first_run"):
                logging.info(f"收到图像 - 宽: {width}, 高: {height}, 像素格式: {pixel_type}, 帧长度: {frame_len}")
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
                    logging.error(f"无法处理像素格式: {pixel_type}，错误: {e}")
                    return
            
            # 将图像复制到队列（仅保留最新的图像）- 香橙派内存管理优化
            if processed_frame is not None and processed_frame.size > 0:
                g_frame_queue.append(processed_frame.copy())
                g_total_frames += 1
                g_last_frame_time = time.time()
                    
    except Exception as e:
        global g_error_count
        g_error_count += 1
        logging.error(f"图像处理错误: {e}")
        logging.error(traceback.format_exc())

# 显示图像线程（可选，用于监控）
def display_thread():
    global g_bExit, g_frame_queue
    
    logging.info("开始显示视频流")
    
    # 帧率计算变量
    fps_count = 0
    fps_start_time = time.time()
    current_fps = 0
    
    while not g_bExit:
        if g_frame_queue:
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
                # g_fps_list.append(current_fps)
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

# 健康监控线程
def health_monitor_thread():
    global g_bExit, g_total_frames, g_memory_usage, g_last_frame_count, g_frame_loss_count
    global g_fps_count, g_fps_start_time, g_fps_list
    
    logging.info("健康监控线程已启动")
    # 帧率计算变量
    local_frame_count = 0
    fps_start_time = time.time()
    current_fps = 0

    while not g_bExit:
        try:
            # 检查内存使用情况
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)  # 转换为MB
            g_memory_usage.append(memory_mb)
            
            # 计算并记录帧率（无论是否显示）
            current_frame_count = g_total_frames
            if current_frame_count > local_frame_count:
                elapsed_time = time.time() - fps_start_time
                if elapsed_time >= 1.0:
                    current_fps = (current_frame_count - local_frame_count) / elapsed_time
                    g_fps_list.append(current_fps)
                    local_frame_count = current_frame_count
                    fps_start_time = time.time()

            # 输出健康监控日志
            logging.info(f"健康监控 - 总帧数: {g_total_frames}, 当前帧率: {current_fps:.2f} fps, 内存使用: {memory_mb:.2f} MB, 错误数: {g_error_count}")
            
            # 检查帧丢失
            if g_total_frames == g_last_frame_count:
                g_frame_loss_count += 1
                if g_frame_loss_count > 10:  # 连续10秒没有新帧
                    logging.warning(f"检测到帧丢失，连续 {g_frame_loss_count} 秒没有收到新帧")
            else:
                g_frame_loss_count = 0
            g_last_frame_count = g_total_frames
            
            # 睡眠1秒
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"健康监控线程错误: {e}")
            logging.error(traceback.format_exc())
            time.sleep(1)

# 香橙派相机参数设置 - 针对ARM处理器优化
def set_camera_params(cam, device_info):
    global g_target_fps, g_target_width, g_target_height
    
    try:
        logging.info("开始设置香橙派相机参数...")
        
        # 设置较低的分辨率以适应香橙派性能
        logging.info(f"设置目标分辨率为 {g_target_width}x{g_target_height}...")
        
        # 设置宽度
        ret = cam.MV_CC_SetIntValue("Width", g_target_width)
        if ret != 0:
            logging.error(f"设置宽度失败! 尝试默认值")
            # 尝试设置为较小的分辨率
            ret = cam.MV_CC_SetIntValue("Width", 640)
            if ret == 0:
                logging.info("已设置为640宽度")
            else:
                logging.error(f"设置640宽度也失败，错误码: 0x{ret:x}")
        
        # 设置高度
        ret = cam.MV_CC_SetIntValue("Height", g_target_height)
        if ret != 0:
            logging.error(f"设置高度失败! 尝试默认值")
            # 尝试设置为较小的分辨率
            ret = cam.MV_CC_SetIntValue("Height", 480)
            if ret == 0:
                logging.info("已设置为480高度")
            else:
                logging.error(f"设置480高度也失败，错误码: 0x{ret:x}")
        
        # 验证分辨率设置
        width_param = MVCC_INTVALUE()
        height_param = MVCC_INTVALUE()
        if (cam.MV_CC_GetIntValue("Width", width_param) == 0 and 
            cam.MV_CC_GetIntValue("Height", height_param) == 0):
            logging.info(f"分辨率设置成功: {width_param.nCurValue}x{height_param.nCurValue}")
        
        # 设置较低的帧率以适应香橙派性能
        logging.info(f"设置目标帧率为 {g_target_fps} fps...")
        ret = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", g_target_fps)
        if ret != 0:
            logging.error(f"设置帧率失败! 错误码: 0x{ret:x}")
            # 尝试设置为更低的帧率
            ret = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 10)
            if ret == 0:
                logging.info("已设置为10fps")
        
        # 为香橙派选择最简单的像素格式
        logging.info("设置香橙派优化的像素格式...")
        pixel_format_success = False
        
        # 优先尝试灰度格式以提高性能
        ret = cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8)
        if ret == 0:
            logging.info("已将像素格式设置为8位灰度（香橙派优化）")
            pixel_format_success = True
        else:
            logging.error(f"设置灰度格式失败，错误码: 0x{ret:x}")
            # 如果灰度不行，尝试RGB
            ret = cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed)
            if ret == 0:
                logging.info("已将像素格式设置为8位RGB")
                pixel_format_success = True
            else:
                logging.error(f"设置RGB格式也失败，错误码: 0x{ret:x}")
        
        if not pixel_format_success:
            logging.warning("警告: 无法设置优化的像素格式，将使用相机默认格式")
        
        # 香橙派网络优化
        if device_info.nTLayerType == MV_GIGE_DEVICE:
            logging.info("优化香橙派网络相机设置...")
            # 设置较小的网络包大小以适应网络环境
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", 1500)  # 标准以太网包大小
            if ret == 0:
                logging.info("已设置网络包大小为1500")
            else:
                logging.error(f"设置网络包大小失败，错误码: 0x{ret:x}")
        
        # 香橙派性能优化设置
        logging.info("设置香橙派性能优化参数...")
        # 关闭自动曝光和自动增益
        ret = cam.MV_CC_SetEnumValue("ExposureAuto", 1)  # 关闭自动曝光
        if ret != 0:
            logging.error(f"关闭自动曝光失败，错误码: 0x{ret:x}")
        
        ret = cam.MV_CC_SetEnumValue("GainAuto", 1)     # 关闭自动增益
        if ret != 0:
            logging.error(f"关闭自动增益失败，错误码: 0x{ret:x}")
        
        # 设置合理的曝光和增益值
        ret = cam.MV_CC_SetFloatValue("ExposureTime", 10000)  # 10毫秒
        if ret != 0:
            logging.error(f"设置曝光时间失败，错误码: 0x{ret:x}")
        
        ret = cam.MV_CC_SetFloatValue("Gain", 15)  # 中等增益
        if ret != 0:
            logging.error(f"设置增益失败，错误码: 0x{ret:x}")
        
        # 关闭占用资源的功能
        cam.MV_CC_SetEnumValue("GammaEnable", 0)      # 关闭伽马校正
        cam.MV_CC_SetEnumValue("SharpenEnable", 0)    # 关闭锐化
        cam.MV_CC_SetEnumValue("DenoiseEnable", 0)    # 关闭降噪
        cam.MV_CC_SetEnumValue("ColorTransformationEnable", 0)  # 关闭颜色转换
        
        logging.info("香橙派相机参数设置完成")
        return True
        
    except Exception as e:
        logging.error(f"设置香橙派相机参数时发生错误: {e}")
        logging.error(traceback.format_exc())
        return False

# 生成测试报告
def generate_test_report():
    global g_total_frames, g_fps_list, g_memory_usage, g_error_count, g_start_time
    
    logging.info("生成测试报告...")
    
    # 计算统计信息
    current_time = time.time()
    total_test_time = current_time - g_start_time
    
    if g_fps_list:
        avg_fps = sum(g_fps_list) / len(g_fps_list)
        min_fps = min(g_fps_list)
        max_fps = max(g_fps_list)
    else:
        avg_fps = 0
        min_fps = 0
        max_fps = 0
    
    if g_memory_usage:
        avg_memory = sum(g_memory_usage) / len(g_memory_usage)
        max_memory = max(g_memory_usage)
    else:
        avg_memory = 0
        max_memory = 0
    
    # 计算帧率稳定性
    if len(g_fps_list) > 1:
        fps_std = np.std(g_fps_list)
        fps_variation = (fps_std / avg_fps) * 100 if avg_fps > 0 else 0
    else:
        fps_std = 0
        fps_variation = 0
    
    # 生成报告
    report = f"""
    香橙派相机稳定性测试报告
    ========================
    测试时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(g_start_time))} 到 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}
    测试时长: {total_test_time:.2f} 秒 ({total_test_time/3600:.2f} 小时)
    总帧数: {g_total_frames}
    平均帧率: {avg_fps:.2f} fps
    最低帧率: {min_fps:.2f} fps
    最高帧率: {max_fps:.2f} fps
    帧率稳定性: {fps_variation:.2f}% (变异系数)
    平均内存使用: {avg_memory:.2f} MB
    最大内存使用: {max_memory:.2f} MB
    错误计数: {g_error_count}
    帧丢失次数: {g_frame_loss_count}
    
    测试结果: {'通过' if g_error_count == 0 and g_frame_loss_count < 100 else '未通过'}
    """
    
    logging.info(report)
    
    # 保存报告到文件
    with open('orange_pi_camera_stability_report.txt', 'w') as f:
        f.write(report)
    
    return report

# 主函数
def main():
    global g_bExit, g_bayer_format, g_start_time, g_test_duration
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='香橙派相机稳定性测试程序')
    parser.add_argument('--duration', type=int, default=3600, help='测试时长（秒），默认3600秒（1小时）')
    parser.add_argument('--no-display', action='store_true', help='不显示视频流')
    args = parser.parse_args()
    
    g_test_duration = args.duration
    logging.info(f"开始香橙派相机稳定性测试，测试时长: {g_test_duration} 秒")
    
    try:
        # 初始化SDK
        logging.info("初始化MVS SDK...")
        ret = MvCamera.MV_CC_Initialize()
        if ret != 0:
            logging.error(f"SDK初始化失败! 错误码: 0x{ret:x}")
            return
        
        # 获取SDK版本
        try:
            sdk_version = MvCamera.MV_CC_GetSDKVersion()
            logging.info(f"SDK版本: 0x{sdk_version:x}")
        except:
            logging.error("无法获取SDK版本")
        
        # 枚举设备
        logging.info("枚举相机设备...")
        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer_type = (MV_GIGE_DEVICE | MV_USB_DEVICE)  # 香橙派主要支持的设备类型
        
        ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        if ret != 0:
            logging.error(f"枚举设备失败! 错误码: 0x{ret:x}")
            return
        
        if device_list.nDeviceNum == 0:
            logging.error("未找到海康威视相机设备!")
            logging.error("请检查:")
            logging.error("1. 相机是否正确连接")
            logging.error("2. MVS SDK的Python_for_arm目录是否存在于项目根目录")
            logging.error("3. 设备权限是否正确设置")
            logging.error("4. 相机驱动是否正确安装")
            return
        
        logging.info(f"找到 {device_list.nDeviceNum} 个设备:")
        
        # 显示设备信息
        for i in range(device_list.nDeviceNum):
            device_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            
            if device_info.nTLayerType == MV_GIGE_DEVICE:
                logging.info(f"\n设备 {i}: 网络相机")
                try:
                    model_name = string_at(device_info.SpecialInfo.stGigEInfo.chModelName, 32).decode('utf-8', errors='ignore').strip('\x00')
                    logging.info(f"  型号: {model_name}")
                    ip_parts = [
                        (device_info.SpecialInfo.stGigEInfo.nCurrentIp >> 24) & 0xFF,
                        (device_info.SpecialInfo.stGigEInfo.nCurrentIp >> 16) & 0xFF,
                        (device_info.SpecialInfo.stGigEInfo.nCurrentIp >> 8) & 0xFF,
                        (device_info.SpecialInfo.stGigEInfo.nCurrentIp >> 0) & 0xFF
                    ]
                    ip_address = ".".join(map(str, ip_parts))
                    logging.info(f"  IP地址: {ip_address}")
                except:
                    logging.error("  无法获取详细信息")
            elif device_info.nTLayerType == MV_USB_DEVICE:
                logging.info(f"\n设备 {i}: USB相机")
                try:
                    model_name = string_at(device_info.SpecialInfo.stUsb3VInfo.chModelName, 32).decode('utf-8', errors='ignore').strip('\x00')
                    serial_number = string_at(device_info.SpecialInfo.stUsb3VInfo.chSerialNumber, 32).decode('utf-8', errors='ignore').strip('\x00')
                    logging.info(f"  型号: {model_name}")
                    logging.info(f"  序列号: {serial_number}")
                except:
                    logging.error("  无法获取详细信息")
            else:
                logging.info(f"\n设备 {i}: 其他类型相机")
        
        # 选择第一个设备
        selected_device = 0
        logging.info(f"\n选择设备 {selected_device}")
        
        # 创建相机实例
        logging.info("创建相机实例...")
        cam = MvCamera()
        
        # 创建句柄
        device_info = cast(device_list.pDeviceInfo[selected_device], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = cam.MV_CC_CreateHandle(device_info)
        if ret != 0:
            logging.error(f"创建句柄失败! 错误码: 0x{ret:x}")
            return
        
        # 打开设备
        logging.info("打开设备...")
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            logging.error(f"打开设备失败! 错误码: 0x{ret:x}")
            # 尝试使用其他访问模式
            ret = cam.MV_CC_OpenDevice(MV_ACCESS_ReadWrite, 0)
            if ret != 0:
                logging.error(f"使用读写模式打开设备也失败! 错误码: 0x{ret:x}")
                cam.MV_CC_DestroyHandle()
                return
            else:
                logging.info("使用读写模式成功打开设备")
        
        # 为香橙派设置相机参数
        if not set_camera_params(cam, device_info):
            logging.error("设置相机参数失败，退出测试")
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            return
        
        # 设置触发模式为OFF（连续采集）
        logging.info("设置触发模式为连续采集...")
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            logging.error(f"设置触发模式失败! 错误码: 0x{ret:x}")
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            return
        
        # 注册图像回调函数
        logging.info("注册图像回调函数...")
        fun_ctype = get_platform_functype()
        frame_callback = fun_ctype(None, POINTER(MV_FRAME_OUT), c_void_p, c_bool)(image_callback)
        ret = cam.MV_CC_RegisterImageCallBackEx2(frame_callback, None, True)
        if ret != 0:
            logging.error(f"注册回调失败! 错误码: 0x{ret:x}")
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            return
        
        # 开始取流
        logging.info("开始取流...")
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            logging.error(f"开始取流失败! 错误码: 0x{ret:x}")
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            return
        
        # 记录开始时间
        g_start_time = time.time()
        
        # 启动显示线程（可选）
        if not args.no_display:
            display_thread_handle = threading.Thread(target=display_thread)
            display_thread_handle.daemon = True  # 设置为守护线程
            display_thread_handle.start()
        
        # 启动健康监控线程
        health_thread_handle = threading.Thread(target=health_monitor_thread)
        health_thread_handle.daemon = True  # 设置为守护线程
        health_thread_handle.start()
        
        # 等待测试结束
        logging.info(f"测试将在 {g_test_duration} 秒后结束，或按 'q' 键手动结束")
        start_time = time.time()
        while not g_bExit:
            elapsed_time = time.time() - start_time
            if elapsed_time >= g_test_duration:
                logging.info(f"测试时长已到 {g_test_duration} 秒，结束测试")
                g_bExit = True
                break
            time.sleep(0.1)

        # 等待健康监控线程结束
        logging.info("等待健康监控线程结束...")
        health_thread_handle.join()
        
        # 如果有显示线程，也等待其结束
        if not args.no_display:
            logging.info("等待显示线程结束...")
            display_thread_handle.join()
        
        # 停止取流
        logging.info("停止取流...")
        ret = cam.MV_CC_StopGrabbing()
        if ret != 0:
            logging.error(f"停止取流失败! 错误码: 0x{ret:x}")
        
        # 关闭设备
        logging.info("关闭设备...")
        ret = cam.MV_CC_CloseDevice()
        if ret != 0:
            logging.error(f"关闭设备失败! 错误码: 0x{ret:x}")
        
        # 销毁句柄
        logging.info("销毁句柄...")
        ret = cam.MV_CC_DestroyHandle()
        if ret != 0:
            logging.error(f"销毁句柄失败! 错误码: 0x{ret:x}")
        
        # 生成测试报告
        generate_test_report()
        
    except KeyboardInterrupt:
        logging.info("\n用户中断程序 (Ctrl+C)")
    except Exception as e:
        logging.error(f"发生错误: {e}")
        logging.error(traceback.format_exc())
    finally:
        # 反初始化SDK
        logging.info("反初始化SDK...")
        try:
            MvCamera.MV_CC_Finalize()
        except:
            pass
        
        logging.info("香橙派相机稳定性测试程序结束")

if __name__ == "__main__":
    main()