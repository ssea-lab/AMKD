# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------------
	File Name:         click_see_img_grayscale
	Description:       用於paddlelabel色彩化
	Author:            dell
	date:              2023/8/29
--------------------------------------------------------------------------------
           Change Activity: 2023/8/29
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
from tifffile import TiffFile


def on_click(event):
    if event.inaxes is not None:
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        pixel_value = cdata[y, x]
        print(f"Pixel value at (x={x}, y={y}): {pixel_value}")


# 关闭所有图形窗口
plt.close('all')

# 读取图像数据
a_i_path = os.path.join('segmentation_mask', 'annotation', '宫颈炎',
                        'M22102_2023_P0000132_circle_3_0x3_1_C11_S11_frame_1.png')

cdata = mpimg.imread(a_i_path)

# 创建图像显示窗口
fig, ax = plt.subplots()
ax.imshow(cdata)
ax.set_yticklabels([])  # 设置y轴刻度标签为空，使得图像y轴方向正常

# 绑定点击事件处理函数
fig.canvas.mpl_connect('button_press_event', on_click)

# 显示图像
plt.show()

img = cv2.imread(a_i_path, cv2.IMREAD_GRAYSCALE)
print(img.shape)
print(img)
print(img.sum())
