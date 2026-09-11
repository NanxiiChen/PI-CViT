#!/bin/bash
# LDC 有限 Re 池消融 sweep(单卡版, 手工收敛判断, 不改动 train 脚本)
#
# 用法(开两个终端分别执行):
#   GPU=0 K_LIST="1 4 16 64 128" bash sweep_finite_u_ldc.sh
#   GPU=1 K_LIST="2 8 32"        bash sweep_finite_u_ldc.sh
#
# 小 k 时 batch=k, 单卡即可; 两张卡各自独立串行跑各自的 k 列表。
#
# 运行期间监控 tensorboard 或 console 输出, 认为当前 k 已收敛时:
#   在该终端按【一次】Ctrl+C —— 当前训练被中断, 自动进入下一个 k
#   (连按会连续跳过多个 k; 想终止该 GPU 的整个 sweep 按 Ctrl+\)
#
# 每个 k 的 console 输出同步写入 logs/ldc_finite_u_gpu<GPU>/finite_u<k>.log

cd "$(dirname "$0")"   # 保证从任意目录启动都能找到 python 包

set -u
GPU="${GPU:-0}"
K_LIST="${K_LIST:-1 4 16 64 128}"
SAVE_DIR="/root/autodl-tmp/tf-logs/ldc/cvit/pure_physics_finite_u"
LOG_DIR="logs/ldc_finite_u_gpu${GPU}"
mkdir -p "${LOG_DIR}"

# 关键: 捕获 SIGINT。没有这个 trap, Ctrl+C 会把整个 for 循环一起杀掉;
# 有了它, Ctrl+C 只终结当前前台训练进程, 循环继续到下一个 k
trap 'echo; echo ">> [Ctrl+C] 中断当前 k, 跳到下一个"' INT

for k in ${K_LIST}; do
    LOG="${LOG_DIR}/finite_u${k}.log"
    echo "=================================================="
    echo ">> GPU ${GPU}: num_u_pool = ${k}  (console log: ${LOG})"
    echo ">> 收敛后按一次 Ctrl+C 跳到下一个 k"
    echo "=================================================="
    CUDA_VISIBLE_DEVICES="${GPU}" python -u -m ldc.train \
        --configs train_cvit \
        --set num_u_pool="${k}" \
        --set test_every=200 \
        --set save_name="finite_u${k}" \
        --set save_dir="${SAVE_DIR}" \
        --set use_multi_gpu=False \
        --set num_u_samples=16 \
        2>&1 | tee "${LOG}"
    echo ">> num_u_pool=${k} 结束 (train 退出码 ${PIPESTATUS[0]}), 继续下一个"
done

echo ">> GPU ${GPU} 的 sweep 全部完成。"
