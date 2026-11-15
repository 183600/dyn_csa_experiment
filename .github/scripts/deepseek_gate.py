#!/usr/bin/env python3
"""DeepSeek 把关：审查已暂存(staged)的改动，决定是否允许推送到远程。

用法: deepseek_gate.py [标签]
输出(stdout 单行):
  APPROVE          —— 通过（或暂存区没有改动）
  REJECT:<理由>    —— 拦截本次推送
  UNAVAILABLE      —— DeepSeek API 不可用（由调用方决定保留改动跳过推送）
环境: DEEPSEEK_API_KEY
"""
import json
import os
import re
import subprocess
import sys
import urllib.request

MAX_DIFF_CHARS = 60000


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True).stdout


def call_deepseek(key, model, prompt):
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 300,
            "stream": False,
        }).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"].strip()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "auto commit"
    diff = sh("git", "diff", "--cached")
    if not diff.strip():
        print("APPROVE")
        return

    stat = sh("git", "diff", "--cached", "--stat")
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + (
            f"\n... [diff 已截断，完整长度 {len(diff)} 字符，请结合 --stat 概览判断]")

    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        print("UNAVAILABLE")
        return

    prompt = f"""你是代码把关人。一个自主 AI 循环即将把以下改动推送到 main 分支（提交标签: {label}）。

改动概览 (git diff --cached --stat):
{stat}

改动内容 (diff):
{diff}

审查标准（只在满足拦截条件时才 REJECT）：
1. 明显破坏性：删除大量关键文件/清空核心代码/破坏入口或 CI/注入恶意或完全无关内容；
2. 明显破坏功能语义：实现变空壳、无脑 return、篡改或削弱测试断言。
普通的性能优化、重构、小幅修改、文档/注释变化 → 一律 APPROVE。

只输出一行 JSON，不要任何其他文字：
{{"verdict":"APPROVE","reason":"一句话说明"}}
或
{{"verdict":"REJECT","reason":"一句话拦截理由"}}"""

    content = ""
    try:
        try:
            content = call_deepseek(key, "deepseek-chat", prompt)
        except Exception:
            # 模型别名失效：从 /models 取第一个可用模型重试一次
            with urllib.request.urlopen(urllib.request.Request(
                    "https://api.deepseek.com/models",
                    headers={"Authorization": "Bearer " + key}), timeout=30) as r:
                ids = [m.get("id") for m in json.load(r).get("data", []) if m.get("id")]
            if not ids:
                print("UNAVAILABLE")
                return
            content = call_deepseek(key, ids[0], prompt)
    except Exception:
        print("UNAVAILABLE")
        return

    m = re.search(r'\{[^{}]*"verdict"\s*:\s*"(APPROVE|REJECT)"[^{}]*\}', content, re.S)
    if m:
        try:
            j = json.loads(m.group(0))
            reason = str(j.get("reason", ""))[:300].replace("\n", " ")
            print(f"{j['verdict']}:{reason}" if j["verdict"] == "REJECT" else "APPROVE")
            return
        except Exception:
            pass
    if "REJECT" in content:
        print("REJECT:" + content[:200].replace("\n", " "))
    else:
        print("APPROVE")


if __name__ == "__main__":
    main()
