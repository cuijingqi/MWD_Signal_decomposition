# ══════════════════════════════════════════════════════════════
# main.py  ·  总入口
# ══════════════════════════════════════════════════════════════
#
# 运行顺序：
#   Step0  前处理   ：读取稳定段 → 取最长段 → 两两截齐配对
#   Step1  两级分解 ：WPD + Wasserstein → 粗重构 → CWT + FQI + 软掩模 → FFT精确重构
#
# 可选技术点均在 config.py §3 中配置，此文件无需修改。
# ══════════════════════════════════════════════════════════════

import os
import sys

# Windows PowerShell 下 Python 输出缓冲导致日志重复的根本修复：
# 设置环境变量 PYTHONUNBUFFERED 使底层 C 缓冲关闭
os.environ["PYTHONUNBUFFERED"] = "1"
# 同时设置 reconfigure 使 Python 层也无缓冲
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from config import OUTPUT_FOLDER, OUTPUT_STEP0, OUTPUT_STEP1, OUTPUT_STEP2


class _Tee:
    """同时写入原始流和文件，实现控制台+文件双输出。"""
    def __init__(self, stream, fobj):
        self._stream = stream
        self._fobj   = fobj

    def write(self, data):
        self._stream.write(data)
        self._fobj.write(data)
        self._fobj.flush()

    def flush(self):
        self._stream.flush()
        self._fobj.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _p(msg=""):
    """打印并立即刷新，消除 Windows PowerShell 缓冲重复输出。"""
    print(msg)
    sys.stdout.flush()


def main():
    for d in [OUTPUT_FOLDER, OUTPUT_STEP0, OUTPUT_STEP1, OUTPUT_STEP2]:
        os.makedirs(d, exist_ok=True)

    log_path = os.path.join(OUTPUT_FOLDER, "output.md")
    _log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)

    _p("══════════════════════════════════════════════════════")
    _p("  MWD 信号两级渐进分解分析")
    _p("  方法：WPD + Wasserstein + FQI + 双向软掩模")
    _p("══════════════════════════════════════════════════════")

    # ── Step0：前处理 ──────────────────────────────────────────
    _p("\n【步骤0】前处理：读取最长稳定段 → 两两截齐配对")
    from step0_preprocess import run_preprocess
    hole_data, pair_data = run_preprocess()
    sys.stdout.flush()

    # ── Step1：两级渐进分解 ────────────────────────────────────
    _p("\n【步骤1】两级渐进分解：WPD → Wasserstein分类 → 粗重构 → SST + FQI → 四分量重构")
    from step1_line1 import run_line1
    pair_results = run_line1(pair_data)
    sys.stdout.flush()

    sys.stdout.flush()

    _p("\n══════════════════════════════════════════════════════")
    _p(f"  全部完成，输出保存在：{OUTPUT_FOLDER}/")
    _p("══════════════════════════════════════════════════════")

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _log_file.close()
    print(f"日志已保存至：{log_path}")


if __name__ == "__main__":
    main()