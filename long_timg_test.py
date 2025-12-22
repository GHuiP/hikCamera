# -- coding: utf-8 --

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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('camera_stability_test.log'),
        logging.StreamHandler()
    ]
)

# 设置本地MVS SDK路径
current_dir = os.path.dirname(os.path.abspath(__file__))
mv_import_path = os.path.join(current_dir, 'Python', 'MvImport')
if mv_import_path not in sys.path:
    sys.path.append(mv_import_path)

# 导入MVS模块
from MvCameraControl_class import *
# 导入像素格式定义
from PixelType_header import *

# 尝试导入CUDA相关库
try:
    import torch
    import torch.cuda as cuda
    CUDA_AVAILABLE = cuda.is_available()
    logging.info(f"CUDA 可用: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")
except ImportError:
    CUDA_AVAILABLE = False
    DEVICE = None
    logging.info("CUDA 不可用，使用CPU处理")

# 全局变量
g_bExit = False
g_frame_queue = collections.deque(maxlen=5)  # 线程安全的队列，最多保存5帧
g_target_fps = 30  # 目标帧率
g_target_width = 1920  # 目标宽度
g_target_height = 1080  # 目标高度
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
            
            # 根据像素格式处理图像
            if pixel_type == PixelType_Gvsp_Mono8:
                # 单通道8位灰度图像（最快）
                processed_frame = np.ctypeslib.as_array(stFrame.pBufAddr, (height, width))
            elif pixel_type in [PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed]:
                # RGB/BGR图像
                processed_frame = np.ctypeslib.as_array(stFrame.pBufAddr, (height, width, 3))
            elif pixel_type in [PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8,
                              PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8]:
                # 8位Bayer格式
                bayer_data = np.ctypeslib.as_array(stFrame.pBufAddr, (height, width))
                if pixel_type == PixelType_Gvsp_BayerGR8:
                    conversion_code = cv2.COLOR_BAYER_GR2RGB
                elif pixel_type == PixelType_Gvsp_BayerRG8:
                    conversion_code = cv2.COLOR_BAYER_RG2RGB
                elif pixel_type == PixelType_Gvsp_BayerGB8:
                    conversion_code = cv2.COLOR_BAYER_GB2RGB
                elif pixel_type == PixelType_Gvsp_BayerBG8:
                    conversion_code = cv2.COLOR_BAYER_BG2RGB
                
                # 使用OpenCV的Bayer转换
                processed_frame = cv2.cvtColor(bayer_data, conversion_code)
            else:
                # 其他格式，尝试快速转换
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
            
            # 将图像复制到队列（仅保留最新的图像）
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
            
            # 调整图像大小
            if CUDA_AVAILABLE:
                # 使用PyTorch的CUDA加速调整大小
                frame_tensor = torch.from_numpy(frame).to(DEVICE)
                if len(frame_tensor.shape) == 2:  # 灰度图像
                    frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # 添加通道和批量维度
                    is_gray = True
                elif len(frame_tensor.shape) == 3:  # 彩色图像
                    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW -> BCHW
                    is_gray = False
                
                # 进行插值调整大小
                resized_tensor = torch.nn.functional.interpolate(
                    frame_tensor.float() / 255.0,
                    size=(600, 800),
                    mode='bilinear',
                    align_corners=False
                )
                
                # 转换回HWC格式并缩放回0-255范围
                if is_gray:
                    resized_frame = (resized_tensor.squeeze(0).squeeze(0) * 255).byte().cpu().numpy()
                else:
                    resized_frame = (resized_tensor.squeeze(0).permute(1, 2, 0) * 255).byte().cpu().numpy()
            else:
                # 使用OpenCV调整大小
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
            cv2.imshow('Hikvision Camera Stream', display_frame)
            
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

# 在健康监控线程中添加帧率计算逻辑
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
# 设置相机参数
def set_camera_params(cam, device_info):
    global g_target_fps, g_target_width, g_target_height
    
    try:
        # 设置分辨率
        logging.info(f"设置目标分辨率为 {g_target_width}x{g_target_height}...")
        
        # 获取当前宽度
        width_param = MVCC_INTVALUE()
        ret = cam.MV_CC_GetIntValue("Width", width_param)
        if ret == 0:
            logging.info(f"当前宽度: {width_param.nCurValue}, 范围: {width_param.nMin} - {width_param.nMax}, 步长: {width_param.nInc}")
        
        # 获取当前高度
        height_param = MVCC_INTVALUE()
        ret = cam.MV_CC_GetIntValue("Height", height_param)
        if ret == 0:
            logging.info(f"当前高度: {height_param.nCurValue}, 范围: {height_param.nMin} - {height_param.nMax}, 步长: {height_param.nInc}")
        
        # 设置宽度
        ret = cam.MV_CC_SetIntValue("Width", g_target_width)
        if ret != 0:
            logging.error(f"设置宽度失败! 错误码: 0x{ret:x}")
        
        # 设置高度
        ret = cam.MV_CC_SetIntValue("Height", g_target_height)
        if ret != 0:
            logging.error(f"设置高度失败! 错误码: 0x{ret:x}")
        
        # 验证分辨率设置
        width_param = MVCC_INTVALUE()
        height_param = MVCC_INTVALUE()
        if cam.MV_CC_GetIntValue("Width", width_param) == 0 and cam.MV_CC_GetIntValue("Height", height_param) == 0:
            logging.info(f"分辨率设置成功: {width_param.nCurValue}x{height_param.nCurValue}")
        
        # 设置帧率
        logging.info(f"设置目标帧率为 {g_target_fps} fps...")
        ret = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", g_target_fps)
        if ret != 0:
            logging.error(f"设置帧率失败! 错误码: 0x{ret:x}")
            # 尝试获取当前帧率范围
            try:
                min_val, max_val = cam.MV_CC_GetFloatValueRange("AcquisitionFrameRate")
                logging.info(f"相机支持的帧率范围: {min_val:.1f} - {max_val:.1f} fps")
                # 使用最大帧率
                ret = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", max_val)
                if ret == 0:
                    logging.info(f"已设置为相机支持的最大帧率: {max_val:.1f} fps")
            except Exception as e:
                logging.error(f"获取帧率范围失败: {e}")
        
        # 强制设置8位像素格式
        logging.info("强制设置8位像素格式...")
        pixel_format_success = False
        
        # 尝试设置为8位灰度
        ret = cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8)
        if ret == 0:
            logging.info("已将像素格式设置为8位灰度")
            pixel_format_success = True
        else:
            # 尝试设置为8位RGB
            ret = cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed)
            if ret == 0:
                logging.info("已将像素格式设置为8位RGB")
                pixel_format_success = True
            else:
                # 尝试设置为8位Bayer
                ret = cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BayerGR8)
                if ret == 0:
                    logging.info("已将像素格式设置为8位BayerGR")
                    pixel_format_success = True
        
        if not pixel_format_success:
            logging.warning("警告: 无法设置为8位像素格式，将使用原始格式")
        
        # 对于网络相机，启用数据包丢失重传和优化
        if device_info.nTLayerType == MV_GIGE_DEVICE:
            logging.info("优化网络相机设置...")
            # 设置最佳网络包大小
            packet_size = cam.MV_CC_GetOptimalPacketSize()
            if packet_size > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
                if ret == 0:
                    logging.info(f"已设置最佳网络包大小: {packet_size}")
            
            # 启用流控制
            ret = cam.MV_CC_SetEnumValue("GevStreamChannelSelector", 0)
            if ret == 0:
                cam.MV_CC_SetEnumValue("GevSCPD", 10000)  # 设置流延迟参数
                cam.MV_CC_SetEnumValue("GevSCFTD", 5000)  # 设置流帧超时
        
        # 关闭自动曝光和自动增益
        logging.info("设置曝光和增益参数...")
        cam.MV_CC_SetEnumValue("ExposureAuto", 1)  # 关闭自动曝光
        cam.MV_CC_SetEnumValue("GainAuto", 1)  # 关闭自动增益
        
        # 设置曝光时间和增益
        cam.MV_CC_SetFloatValue("ExposureTime", 5000)  # 5毫秒
        cam.MV_CC_SetFloatValue("Gain", 10)  # 较低的增益
        
        # 关闭不需要的功能
        cam.MV_CC_SetEnumValue("GammaEnable", 0)  # 关闭伽马校正
        cam.MV_CC_SetEnumValue("SharpenEnable", 0)  # 关闭锐化
        cam.MV_CC_SetEnumValue("DenoiseEnable", 0)  # 关闭降噪
        
        return True
        
    except Exception as e:
        logging.error(f"设置相机参数时发生错误: {e}")
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
    相机稳定性测试报告
    ====================
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
    with open('camera_stability_report.txt', 'w') as f:
        f.write(report)
    
    return report

# 主函数
def main():
    global g_bExit, g_bayer_format, g_start_time, g_test_duration
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='相机稳定性测试程序')
    parser.add_argument('--duration', type=int, default=3600, help='测试时长（秒），默认3600秒（1小时）')
    parser.add_argument('--no-display', action='store_true', help='不显示视频流')
    args = parser.parse_args()
    
    g_test_duration = args.duration
    logging.info(f"开始相机稳定性测试，测试时长: {g_test_duration} 秒")
    
    try:
        # 初始化SDK
        logging.info("初始化SDK...")
        MvCamera.MV_CC_Initialize()
        
        # 获取SDK版本
        sdk_version = MvCamera.MV_CC_GetSDKVersion()
        logging.info(f"SDK版本: 0x{sdk_version:x}")
        
        # 枚举设备
        logging.info("枚举设备...")
        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer_type = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE | 
                      MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
        
        ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        if ret != 0:
            logging.error(f"枚举设备失败! 错误码: 0x{ret:x}")
            return
        
        if device_list.nDeviceNum == 0:
            logging.error("未找到设备!")
            return
        
        logging.info(f"找到 {device_list.nDeviceNum} 个设备:")
        
        # 显示设备信息
        for i in range(device_list.nDeviceNum):
            device_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            
            if device_info.nTLayerType == MV_GIGE_DEVICE:
                logging.info(f"\n设备 {i}: 网络相机")
                model_name = string_at(device_info.SpecialInfo.stGigEInfo.chModelName, 32).decode('gbk', errors='ignore').strip('\x00')
                logging.info(f"  型号: {model_name}")
                logging.info(f"  IP地址: {(device_info.SpecialInfo.stGigEInfo.nCurrentIp >> 24) & 0xFF}.{(device_info.SpecialInfo.stGigEInfo.nCurrentIp >> 16) & 0xFF}.{(device_info.SpecialInfo.stGigEInfo.nCurrentIp >> 8) & 0xFF}.{(device_info.SpecialInfo.stGigEInfo.nCurrentIp >> 0) & 0xFF}")
            elif device_info.nTLayerType == MV_USB_DEVICE:
                logging.info(f"\n设备 {i}: USB相机")
                model_name = string_at(device_info.SpecialInfo.stUsb3VInfo.chModelName, 32).decode('gbk', errors='ignore').strip('\x00')
                serial_number = string_at(device_info.SpecialInfo.stUsb3VInfo.chSerialNumber, 32).decode('gbk', errors='ignore').strip('\x00')
                logging.info(f"  型号: {model_name}")
                logging.info(f"  序列号: {serial_number}")
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
            cam.MV_CC_DestroyHandle()
            return
        
        # 设置网络相机最佳包大小
        if device_info.nTLayerType == MV_GIGE_DEVICE:
            packet_size = cam.MV_CC_GetOptimalPacketSize()
            if packet_size > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
                if ret != 0:
                    logging.error(f"设置包大小失败! 错误码: 0x{ret:x}")
        
        # 设置相机参数
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
            display_thread_handle.start()
        
        # 启动健康监控线程
        health_thread_handle = threading.Thread(target=health_monitor_thread)
        health_thread_handle.start()
        
        # 等待测试结束
        logging.info(f"测试将在 {g_test_duration} 秒后结束，或按 'q' 键手动结束")
        while not g_bExit:
            elapsed_time = time.time() - g_start_time
            if elapsed_time >= g_test_duration:
                logging.info(f"测试时长已到 {g_test_duration} 秒，结束测试")
                g_bExit = True # 显示设置退出标志
                break
            time.sleep(0.1)

        # 2. 等待健康监控线程结束
        logging.info("等待健康监控线程结束...")
        health_thread_handle.join()
        
        # 3. 如果有显示线程，也等待其结束
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
        
    except Exception as e:
        logging.error(f"发生错误: {e}")
        logging.error(traceback.format_exc())
    finally:
        # 反初始化SDK
        logging.info("反初始化SDK...")
        MvCamera.MV_CC_Finalize()
        
        logging.info("程序结束")

if __name__ == "__main__":
    main()