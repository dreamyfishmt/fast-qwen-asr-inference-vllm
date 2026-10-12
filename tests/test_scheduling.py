"""Inference scheduling: stream priority, streams vs. uploads, cancellation."""

import asyncio
import threading
import time

from asr_server.service import BATCH, STREAM, PrioritySemaphore, Service

from .conftest import FakeEngine


def test_priority_semaphore_serves_streams_first():
    async def main():
        sem = PrioritySemaphore(1)
        order = []

        async def job(name, priority):
            async with sem.slot(priority):
                order.append(name)
                await asyncio.sleep(0.01)

        async with sem.slot(BATCH):
            tasks = [asyncio.create_task(job("batch", BATCH)), asyncio.create_task(job("stream", STREAM))]
            await asyncio.sleep(0.01)  # both are waiting now
        await asyncio.gather(*tasks)
        return order

    assert asyncio.run(main()) == ["stream", "batch"]


def test_priority_semaphore_cancelled_waiter():
    async def main():
        sem = PrioritySemaphore(1)
        async with sem.slot(BATCH):
            waiter = asyncio.create_task(sem.slot(STREAM).__aenter__())
            await asyncio.sleep(0.01)
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        async with sem.slot(BATCH):  # the slot was not lost
            return True

    assert asyncio.run(asyncio.wait_for(main(), 1))


def _blocking(seconds, log, name):
    def run(*_):
        log.append((name, "start"))
        time.sleep(seconds)
        log.append((name, "end"))
        return name
    return run


def test_non_exclusive_engine_streams_bypass_uploads():
    engine = FakeEngine()
    engine.exclusive = False
    log = []

    async def main():
        service = Service(engine)
        upload = asyncio.create_task(service.run_infer(_blocking(0.3, log, "upload")))
        await asyncio.sleep(0.05)
        await service.run_stream(_blocking(0, log, "stream"))
        log.append(("upload running", not upload.done()))
        await upload

    asyncio.run(main())
    assert ("upload running", True) in log


def test_cancelled_call_keeps_its_slot_until_the_thread_returns():
    log = []
    running = threading.Lock()

    def guarded(name, seconds):
        def run(*_):
            assert running.acquire(blocking=False), "engine calls overlapped"
            try:
                return _blocking(seconds, log, name)()
            finally:
                running.release()
        return run

    async def main():
        service = Service(FakeEngine())  # exclusive
        first = asyncio.create_task(service.run_stream(guarded("first", 0.3)))
        await asyncio.sleep(0.05)
        first.cancel()
        await service.run_infer(guarded("second", 0))
        assert first.cancelled()

    asyncio.run(main())
    assert log == [("first", "start"), ("first", "end"), ("second", "start"), ("second", "end")]
