import math
import numpy as np

def get_direction(matrices, finger_matrices,previous_det):
    finger_x, finger_y = finger_matrices

    up_left_x, up_left_y = matrices[[0],[0]][0]
    down_left_x, down_left_y = matrices[[1],[0]][0]

    down_right_x, down_right_y = matrices[[2],[0]][0]
    up_right_x, up_right_y = matrices[[3],[0]][0]

    # 判斷有無在以上數值內
    # 計算線性方程式的m(斜率)跟b(截距) y = mx+b
    # 點到線公式
    if finger_y < down_left_y and finger_y > up_left_y:# finger_x > down_left_x and finger_x < up_left_x and
        left_m = (down_left_y - up_left_y) / (down_left_x - up_left_x)
        left_b = up_left_y - (left_m * up_left_x)
        to_left = abs(left_m * finger_x - finger_y + left_b) / (math.sqrt(pow(left_m, 2) + pow(left_b, 2)))
        if to_left*100 < 10:
            return "left", False
            print(f"distance to left= {to_left}")
            points.append(to_left)
    if finger_y < down_right_y and finger_y > up_right_y: # finger_x > down_right_x and finger_x < up_right_x and
        right_m = (up_right_y - down_right_y) / (up_right_x - down_right_x)
        right_b = down_right_y - (right_m * down_right_x)
        to_right = abs(right_m * finger_x - finger_y + right_b) / (math.sqrt(pow(right_m, 2) + pow(right_b, 2)))
        if to_right*100 < 10:
            return "right", False
            print(f"distance to right= {to_right}")
            points.append(to_right)
    if finger_x < up_right_x and finger_x > up_left_x:#  and finger_y < up_right_y and finger_y > up_left_y
        up_m = (up_left_y - up_right_y) / (up_left_x - up_right_x)
        up_b = up_right_y - (up_m * up_right_x)
        to_up = abs(up_m * finger_x - finger_y + up_b) / (math.sqrt(pow(up_m, 2) + pow(up_b, 2)))
        if to_up*100 < 10:
            return "up", False
            print(f"distance to up= {to_up}")
            points.append(to_up)
    if finger_x < down_right_x and finger_x > down_left_x:#  and finger_y < down_right_y and finger_y > down_left_y
        down_m = (down_left_y - down_right_y) / (down_left_x - down_right_x)
        down_b = down_right_y - (down_m * down_right_x)
        to_down = abs(down_m * finger_x - finger_y + down_b) / (math.sqrt(pow(down_m, 2) + pow(down_b, 2)))
        if to_down*100 < 15:
            print(f"distance to down= {to_down}")
            return "down", False
            points.append(to_down)
    return previous_det, True


