import httpx
import pytest

from src.server.cv.perception import GrokPerception, PerceptionError


def reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def make(handler) -> GrokPerception:
    p = GrokPerception(keys=["A", "B"], model="m")
    p.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.x.ai/v1"
    )
    return p


async def test_detect_parses_json():
    async def handler(request):
        return reply('{"state_now": true, "evidence": "cat on table"}')

    obs = await make(handler).detect(b"jpegbytes", "a cat is on the table")
    assert (obs.state, obs.evidence) == (True, "cat on table")


async def test_normalize_rule():
    async def handler(request):
        return reply(
            '{"predicate": "a cat is on the table", '
            '"direction": "rising", "is_transition": true}'
        )

    rule = await make(handler).normalize("the cat jumps onto the table")
    assert (rule.predicate, rule.direction) == ("a cat is on the table", "rising")


async def test_retry_on_429_then_error():
    seen = []

    async def handler(request):
        seen.append(request.headers["authorization"][-1])
        return httpx.Response(429, json={"error": "slow down"})

    with pytest.raises(PerceptionError):
        await make(handler).detect(b"x", "p")
    assert seen == ["A", "B"]


async def test_failover_sticks_to_backup_key():
    seen = []

    async def handler(request):
        key = request.headers["authorization"][-1]
        seen.append(key)
        if key == "A":
            return httpx.Response(401, json={"error": "bad key"})
        return reply('{"state_now": false, "evidence": ""}')

    p = make(handler)
    await p.detect(b"x", "p")
    await p.detect(b"x", "p")  # stays on B, no retry through A
    assert seen == ["A", "B", "B"]


async def test_usage_is_read_from_response():
    async def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"state_now": false}'}}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 12},
            },
        )

    obs = await make(handler).detect(b"x", "p")
    assert obs.usage.as_dict() == {
        "prompt": 300,
        "completion": 12,
        "total": 312,
        "calls": 1,
        "usd": 0.00108,  # 300*$3 + 12*$15 per 1M
    }
