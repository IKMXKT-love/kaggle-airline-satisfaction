"""Run the frozen second-round 10-fold recipe; keep main.py's previous recipe intact."""
from optimize_v2 import cv
from verify_v2 import verify


if __name__ == '__main__':
    print('开始第二轮 10 折预测，自动复用已完成且指纹一致的结果。', flush=True)
    cv()
    verify()
    print('已生成并核验 submission_v2.csv；旧提交文件保留。', flush=True)
