import tkinter as tk


def calculate_color(value_1, value_2, value_3):
    # 计算渐变蓝色
    red = int(value_1 * 255)
    green = int(value_2 * 255)
    blue = int(value_3 * 255)
    # 将RGB值转换为十六进制
    hex_color = "#{:02X}{:02X}{:02X}".format(red, green, blue)
    return hex_color


def update_color(value_1, value_2, value_3):
    hex_color = calculate_color(value_1, value_2, value_3)
    color_label.config(text=hex_color)
    color_frame.config(bg=hex_color)

# 创建主窗口
window = tk.Tk()
window.title("渐变蓝色窗口")

# 创建一个Frame用于显示颜色
color_frame = tk.Frame(window, width=500, height=500)
color_frame.pack(padx=20, pady=20)

# 创建一个Label用于显示颜色的十六进制码
color_label = tk.Label(window, text="#0000FF", font=("Helvetica", 16))
color_label.pack()

# 创建一个滑块用于控制颜色渐变
slider_1 = tk.Scale(window, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, length=200, label="颜色渐变")
slider_2 = tk.Scale(window, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, length=200, label="颜色渐变")
slider_3 = tk.Scale(window, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, length=200, label="颜色渐变")

slider_1.pack()
slider_2.pack()
slider_3.pack()

# 添加滑块值变化的事件处理函数
slider_1.bind("<Motion>", lambda event: update_color(slider_1.get(), slider_2.get(), slider_3.get()))
slider_2.bind("<Motion>", lambda event: update_color(slider_1.get(), slider_2.get(), slider_3.get()))
slider_3.bind("<Motion>", lambda event: update_color(slider_1.get(), slider_2.get(), slider_3.get()))
# 初始更新一次颜色
update_color(slider_1.get(), slider_2.get(), slider_3.get())

# 启动主循环
window.mainloop()



