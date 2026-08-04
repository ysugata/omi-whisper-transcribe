"""Tests for ModelManager GPU lock and acquire timeout."""

import asyncio
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_manager import ModelManager


@pytest.fixture
def mm(monkeypatch):
    """Create a ModelManager with a fake model loader."""
    manager = ModelManager("test-model", "float16")

    # Patch _load_model to avoid GPU dependency
    async def _fake_load(self=manager):
        self._model = "fake-model"

    monkeypatch.setattr(manager, "_load_model", _fake_load)
    return manager


@pytest.mark.asyncio
async def test_acquire_loads_model(mm):
    """acquire() should load the model if not loaded."""
    model, executor = await mm.acquire()
    assert model == "fake-model"
    assert mm.session_count == 1
    mm.release()
    assert mm.session_count == 0


@pytest.mark.asyncio
async def test_acquire_timeout_when_gpu_locked(mm):
    """acquire() should raise TimeoutError when gpu_lock is held."""
    # Simulate batch operation holding gpu_lock
    await mm.gpu_lock.acquire()

    with pytest.raises(TimeoutError, match="GPU busy"):
        await mm.acquire(timeout=0.1)

    mm.gpu_lock.release()


@pytest.mark.asyncio
async def test_acquire_succeeds_after_gpu_lock_released(mm):
    """acquire() should succeed once gpu_lock is released."""
    await mm.gpu_lock.acquire()

    async def release_later():
        await asyncio.sleep(0.1)
        mm.gpu_lock.release()

    asyncio.create_task(release_later())

    # Should succeed (timeout > release delay)
    model, executor = await mm.acquire(timeout=2.0)
    assert model == "fake-model"
    mm.release()


@pytest.mark.asyncio
async def test_force_unload_clears_model(mm):
    """force_unload() should clear the model."""
    await mm.acquire()
    assert mm.loaded
    mm.release()

    async with mm.gpu_lock:
        await mm.force_unload()

    assert not mm.loaded


@pytest.mark.asyncio
async def test_acquire_does_not_hold_gpu_lock_after_return(mm):
    """After acquire() returns, gpu_lock should be released."""
    await mm.acquire()

    # gpu_lock should be free — another task should be able to acquire it
    acquired = mm.gpu_lock.locked()
    assert not acquired, "gpu_lock should not be held after acquire()"

    mm.release()


@pytest.mark.asyncio
async def test_acquire_caller_cancelled_does_not_leak_lock(mm):
    """Cancelling the caller mid-wait must not orphan the lock acquisition.

    asyncio.wait does not cancel the inner acquire task; without cleanup the
    orphan acquires gpu_lock later and nothing ever releases it.
    """
    await mm.gpu_lock.acquire()  # batch holds the lock

    caller = asyncio.ensure_future(mm.acquire(timeout=30))
    await asyncio.sleep(0.05)  # caller is now waiting on gpu_lock
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    # Batch finishes → the orphaned acquire (if leaked) would grab the lock
    mm.gpu_lock.release()
    await asyncio.sleep(0.05)  # let any orphaned task run + done callbacks

    # The lock must be free for the next real session
    model, _ = await asyncio.wait_for(mm.acquire(timeout=1), timeout=2)
    assert model == "fake-model"
    mm.release()
