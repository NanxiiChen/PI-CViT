#!/bin/bash
# 有限函数池消融 sweep(手动收敛判断版,不改动 train 脚本)
#
# 用法:
#   bash sweep_finite_u.sh
#   K_LIST="32 64" bash sweep_finite_u.sh     # 自定义 k 列表
#
# 运行期间监控 tensorboard 或 console 输出,认为当前 k 已收敛时:
#   在跑脚本的终端按【一次】Ctrl+C —— 当前训练被中断,sweep 自动进入下一个 k
#   (连按会连续跳过多个 k;想终止整个 sweep 按 Ctrl+\)
#
# 每个 k 的 console 输出同步写入 logs/swe_finite_u/finite_u<k>.log

cd "$(dirname "$0")"   # 保证从任意目录启动都能找到 python 包

set -u
K_LIST="${K_LIST:-32 64 128 256 512 1024}"
SAVE_DIR="/root/autodl-tmp/tf-logs/swe/cvit/pure_physics_finite_u"
LOG_DIR="logs/swe_finite_u"
mkdir -p "${LOG_DIR}"

# 关键: 捕获 SIGINT。没有这个 trap, Ctrl+C 会把整个 for 循环一起杀掉;
# 有了它, Ctrl+C 只终结当前前台训练进程, 循环继续到下一个 k
trap 'echo; echo ">> [Ctrl+C] 中断当前 k, 跳到下一个"' INT

for k in ${K_LIST}; do
    LOG="${LOG_DIR}/finite_u${k}.log"
    echo "=================================================="
    echo ">> num_u_pool = ${k}  (console log: ${LOG})"
    echo ">> 收敛后按一次 Ctrl+C 跳到下一个 k"
    echo "=================================================="
    python -u -m swe.train \
        --configs train_cvit \
        --set num_u_pool="${k}" \
        --set test_every=200 \
        --set save_name="finite_u${k}" \
        --set save_dir="${SAVE_DIR}" \
        2>&1 | tee "${LOG}"
    echo ">> num_u_pool=${k} 结束 (train 退出码 ${PIPESTATUS[0]}), 继续下一个"
done

echo ">> sweep 全部完成。"
