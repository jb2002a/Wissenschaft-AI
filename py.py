import sys

import modal

app = modal.App("example-hello-world")

@app.function()
def f(i):
    if i % 2 == 0:
        print("hello", i)
    else:
        print("world", i, file=sys.stderr)

    return i * i


@app.local_entrypoint()
def main():
    """로컬에서 `f` 동작 검증. 실행: modal run py.py"""
    cases = (0, 1, 2, 3, 10, -5)
    for n in cases:
        got = f.local(n)
        assert got == n * n, f"f.local({n!r}) -> {got!r}, expected {n * n!r}"
    print("local ok:", [f.local(n) for n in (0, 2, 4)])