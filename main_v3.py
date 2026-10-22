"""Reproduce the frozen source-informed, ten-fold prediction recipe."""
from optimize_v3 import cv
from verify_v3 import verify

if __name__=='__main__':
    print('开始最终 10 折预测，将自动复用已完成且指纹匹配的结果。',flush=True)
    cv();verify()
    print('结果已核验：submission_v3.csv。旧版本均保留。',flush=True)
