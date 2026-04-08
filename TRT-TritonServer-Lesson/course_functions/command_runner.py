import subprocess
import sys

def run_command(command, output: bool = True) -> None:
    """
    执行一个 shell 命令并可选择打印输出。

    参数：
        command (str or list): 要执行的命令，可以是字符串或字符串列表。
        output (bool, 可选): 是否打印命令输出，默认为 True。
    """
    # 判断命令是字符串还是列表
    if isinstance(command, str):
        # 如果是字符串，使用 shell=True
        shell = True
    else:
        # 如果是列表，不需要 shell=True
        shell = False

    # 使用 Popen 执行命令，禁用缓冲
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # 以文本模式读取输出
        bufsize=1,  # 行缓冲
        shell=shell,
        universal_newlines=True  # 确保文本模式跨平台兼容
    )

    # 实时读取输出
    while True:
        # 读取标准输出
        stdout_line = process.stdout.readline()
        if stdout_line == '' and process.poll() is not None:
            break
        if stdout_line and output:
            print(stdout_line.strip(), flush=True)  # 确保实时刷新输出

    # 等待命令执行完成
    process.wait()

    # 检查命令是否成功执行
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)