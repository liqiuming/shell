#!/usr/bin/env python3
"""Warm up an OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable, List, Tuple


DEFAULT_BASE_URL = "https://maas.mlsvcloud.com:8559/maas/ai/aiFactoryServer/v1/apis/1/v1"
DEFAULT_MODEL = "Kimi-K2.6-mls"
DEFAULT_SYSTEM_PROMPT = "你是一个服务预热助手。请稳定、简短地回答。"
DEFAULT_TIMEOUT = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="预热 OpenAI 兼容的大模型接口，支持普通和流式请求。"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"接口基础地址，默认: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"模型名，默认: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MAAS_API_KEY", ""),
        help="API Key；也可以通过环境变量 MAAS_API_KEY 提供。",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="每组预热提示词执行轮数，默认 2。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="默认 max_tokens，默认 64。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="默认 temperature，默认 0.0。",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"请求超时秒数，默认 {DEFAULT_TIMEOUT}。",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="使用 stream=true 预热流式返回路径。",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="跳过 HTTPS 证书校验，仅排障时使用。",
    )
    parser.add_argument(
        "--path",
        default="/chat/completions",
        help="OpenAI 兼容接口路径，默认 /chat/completions。",
    )
    parser.add_argument(
        "--prompt-file",
        help="从文件读取预热提示词，一行一个；不传则使用内置三组提示词。",
    )
    return parser.parse_args()


def build_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}"


def default_prompts() -> List[Tuple[str, str]]:
    long_prompt = (
        "下面是一段用于预热的长上下文，请阅读后只输出一行“预热完成”。"
        " 背景：当前服务基于 Ray + vLLM 多机部署，推理过程中可能涉及跨节点通信、"
        " Triton 内核 JIT 编译、KV cache 初始化、prefix cache 命中以及流式输出路径。"
        " 本次请求的目标不是获取高质量答案，而是尽可能覆盖真实业务中常见的中长上下文形状，"
        " 让服务在正式接入流量前完成主要算子和通信路径的热身。"
    )
    return [
        ("short", "你好，请只回复“预热成功”。"),
        ("medium", "请用两句话说明为什么大模型服务在刚启动时会感觉更慢。"),
        ("long", long_prompt),
    ]


def load_prompts(prompt_file: str | None) -> List[Tuple[str, str]]:
    if not prompt_file:
        return default_prompts()

    prompts: List[Tuple[str, str]] = []
    with open(prompt_file, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            text = line.strip()
            if text:
                prompts.append((f"file-{index}", text))
    if not prompts:
        raise ValueError("prompt 文件为空，至少需要一行有效提示词。")
    return prompts


def make_ssl_context(insecure: bool) -> ssl.SSLContext:
    if insecure:
        return ssl._create_unverified_context()  # noqa: SLF001
    return ssl.create_default_context()


def build_payload(
    model: str,
    prompt: str,
    stream: bool,
    max_tokens: int,
    temperature: float,
) -> bytes:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def make_request(
    url: str,
    api_key: str,
    payload: bytes,
) -> urllib.request.Request:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    return urllib.request.Request(url=url, data=payload, headers=headers, method="POST")


def read_streaming_response(response: Iterable[bytes], started_at: float) -> Tuple[float | None, int, str]:
    ttft = None
    chunk_count = 0
    preview = ""

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if not line or not line.startswith("data:"):
            continue

        data = line[5:].strip()
        if data == "[DONE]":
            break

        chunk_count += 1
        if ttft is None:
            ttft = time.perf_counter() - started_at

        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            if not preview:
                preview = data[:120]
            continue

        choices = event.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            if content and not preview:
                preview = content[:120]

    return ttft, chunk_count, preview


def read_normal_response(body: bytes) -> str:
    try:
        data = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return body.decode("utf-8", errors="ignore")[:200]

    choices = data.get("choices") or []
    if not choices:
        return json.dumps(data, ensure_ascii=False)[:200]

    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        content = "".join(parts)
    return str(content)[:200]


def perform_request(
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    stream: bool,
    max_tokens: int,
    temperature: float,
    timeout: int,
    ssl_context: ssl.SSLContext,
) -> Tuple[float, float | None, str]:
    payload = build_payload(model, prompt, stream, max_tokens, temperature)
    request = make_request(url, api_key, payload)
    started_at = time.perf_counter()

    with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
        if stream:
            ttft, chunk_count, preview = read_streaming_response(response, started_at)
            summary = f"stream chunks={chunk_count}, preview={preview or '<empty>'}"
        else:
            body = response.read()
            ttft = None
            preview = read_normal_response(body)
            summary = f"preview={preview or '<empty>'}"

    elapsed = time.perf_counter() - started_at
    return elapsed, ttft, summary


def validate_args(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise ValueError("请通过 --api-key 或环境变量 MAAS_API_KEY 提供 API Key。")
    if args.rounds < 1:
        raise ValueError("--rounds 必须大于等于 1。")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens 必须大于等于 1。")
    if args.timeout < 1:
        raise ValueError("--timeout 必须大于等于 1。")


def main() -> int:
    try:
        args = parse_args()
        validate_args(args)
        prompts = load_prompts(args.prompt_file)
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2

    url = build_url(args.base_url, args.path)
    ssl_context = make_ssl_context(args.insecure)

    print(f"预热地址: {url}")
    print(f"模型名称: {args.model}")
    print(f"预热模式: {'stream' if args.stream else 'normal'}")
    print(f"提示词数量: {len(prompts)}，轮数: {args.rounds}")
    print("-" * 72)

    total_requests = 0
    success_requests = 0
    total_elapsed = 0.0

    for round_index in range(1, args.rounds + 1):
        for label, prompt in prompts:
            total_requests += 1
            try:
                elapsed, ttft, summary = perform_request(
                    url=url,
                    api_key=args.api_key,
                    model=args.model,
                    prompt=prompt,
                    stream=args.stream,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    ssl_context=ssl_context,
                )
                success_requests += 1
                total_elapsed += elapsed
                ttft_text = f"{ttft:.3f}s" if ttft is not None else "n/a"
                print(
                    f"[OK] round={round_index} prompt={label:<6} "
                    f"elapsed={elapsed:.3f}s ttft={ttft_text} {summary}"
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")[:300]
                print(
                    f"[HTTP {exc.code}] round={round_index} prompt={label:<6} body={body}",
                    file=sys.stderr,
                )
            except urllib.error.URLError as exc:
                print(
                    f"[URL ERROR] round={round_index} prompt={label:<6} reason={exc.reason}",
                    file=sys.stderr,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[ERROR] round={round_index} prompt={label:<6} detail={exc}",
                    file=sys.stderr,
                )

    print("-" * 72)
    avg_elapsed = total_elapsed / success_requests if success_requests else 0.0
    print(
        f"完成: success={success_requests}/{total_requests}, "
        f"avg_elapsed={avg_elapsed:.3f}s"
    )
    return 0 if success_requests == total_requests else 1


if __name__ == "__main__":
    raise SystemExit(main())
