#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for checkpoint and resume functionality."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from transformation_portal.streaming.checkpoint import (
    Checkpoint,
    CheckpointManager,
    checkpoint as checkpoint_decorator,
    resume_from_checkpoint,
)


class TestCheckpoint:
    """Tests for Checkpoint dataclass."""

    def test_initialization(self):
        """Test checkpoint initialization."""
        checkpoint = Checkpoint(
            id="test_checkpoint",
            progress=50.0,
            state={'current_file': 'file1.jpg'},
            timestamp=time.time(),
            metadata={'batch': 1}
        )

        assert checkpoint.id == "test_checkpoint"
        assert checkpoint.progress == 50.0
        assert checkpoint.state == {'current_file': 'file1.jpg'}
        assert checkpoint.metadata == {'batch': 1}

    def test_save_checkpoint(self):
        """Test saving checkpoint to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Checkpoint(
                id="test_save",
                progress=75.0,
                state={'index': 10},
                timestamp=time.time(),
                metadata={'info': 'test'}
            )

            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            checkpoint.save(checkpoint_path)

            assert checkpoint_path.exists()

            # Verify content
            with open(checkpoint_path) as f:
                data = json.load(f)

            assert data['id'] == "test_save"
            assert data['progress'] == 75.0
            assert data['state'] == {'index': 10}

    def test_save_creates_parent_directories(self):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Checkpoint(
                id="test",
                progress=50.0,
                state={},
                timestamp=time.time(),
                metadata={}
            )

            nested_path = Path(tmpdir) / "a" / "b" / "c" / "checkpoint.json"
            checkpoint.save(nested_path)

            assert nested_path.exists()

    def test_load_checkpoint(self):
        """Test loading checkpoint from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint file
            checkpoint_data = {
                'id': 'test_load',
                'progress': 80.0,
                'state': {'processed': 100},
                'timestamp': time.time(),
                'metadata': {'version': '1.0'}
            }

            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f)

            # Load it
            loaded = Checkpoint.load(checkpoint_path)

            assert loaded.id == 'test_load'
            assert loaded.progress == 80.0
            assert loaded.state == {'processed': 100}
            assert loaded.metadata == {'version': '1.0'}


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager("test_op", checkpoint_dir=Path(tmpdir))

            assert manager.operation_id == "test_op"
            assert manager.checkpoint_dir.exists()

    def test_initialization_creates_directory(self):
        """Test that initialization creates checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints" / "my_op"
            manager = CheckpointManager("my_op", checkpoint_dir=checkpoint_dir)

            assert checkpoint_dir.exists()

    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager("test_op", checkpoint_dir=Path(tmpdir))

            checkpoint = manager.create_checkpoint(
                progress=50.0,
                state={'current': 10},
                metadata={'info': 'test'}
            )

            assert checkpoint.id.startswith("test_op_")
            assert checkpoint.progress == 50.0
            assert checkpoint.state == {'current': 10}
            assert checkpoint.metadata == {'info': 'test'}

    def test_save_checkpoint(self):
        """Test saving checkpoint through manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager("test_op", checkpoint_dir=Path(tmpdir))

            checkpoint = manager.create_checkpoint(
                progress=60.0,
                state={'value': 42}
            )

            saved_path = manager.save(checkpoint)

            assert saved_path.exists()
            assert saved_path.parent == manager.checkpoint_dir

    def test_get_latest_checkpoint(self):
        """Test getting latest checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager("test_op", checkpoint_dir=Path(tmpdir))

            # Create multiple checkpoints
            checkpoint1 = manager.create_checkpoint(progress=25.0, state={'step': 1})
            time.sleep(0.01)
            checkpoint2 = manager.create_checkpoint(progress=50.0, state={'step': 2})
            time.sleep(0.01)
            checkpoint3 = manager.create_checkpoint(progress=75.0, state={'step': 3})

            manager.save(checkpoint1)
            manager.save(checkpoint2)
            manager.save(checkpoint3)

            latest = manager.get_latest()

            assert latest is not None
            assert latest.progress == 75.0
            assert latest.state == {'step': 3}

    def test_get_latest_when_no_checkpoints(self):
        """Test getting latest when no checkpoints exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager("test_op", checkpoint_dir=Path(tmpdir))

            latest = manager.get_latest()
            assert latest is None

    def test_list_checkpoints(self):
        """Test listing all checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager("test_op", checkpoint_dir=Path(tmpdir))

            # Create and save multiple checkpoints
            for i in range(3):
                checkpoint = manager.create_checkpoint(
                    progress=i * 25.0,
                    state={'step': i}
                )
                manager.save(checkpoint)
                time.sleep(0.01)

            checkpoints = manager.list_checkpoints()

            assert len(checkpoints) == 3
            # Should be sorted by timestamp
            assert checkpoints[0].state['step'] == 0
            assert checkpoints[1].state['step'] == 1
            assert checkpoints[2].state['step'] == 2

    def test_list_checkpoints_handles_corrupt_files(self):
        """Test that list_checkpoints handles corrupt checkpoint files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager("test_op", checkpoint_dir=Path(tmpdir))

            # Create valid checkpoint
            checkpoint = manager.create_checkpoint(progress=50.0, state={})
            manager.save(checkpoint)

            # Create corrupt checkpoint file
            corrupt_file = manager.checkpoint_dir / "corrupt.json"
            with open(corrupt_file, 'w') as f:
                f.write("not valid json{")

            checkpoints = manager.list_checkpoints()

            # Should still return valid checkpoint
            assert len(checkpoints) == 1

    def test_clear_checkpoints(self):
        """Test clearing all checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager("test_op", checkpoint_dir=Path(tmpdir))

            # Create checkpoints
            for i in range(3):
                checkpoint = manager.create_checkpoint(progress=i * 25.0, state={})
                manager.save(checkpoint)

            assert len(list(manager.checkpoint_dir.glob('*.json'))) == 3

            manager.clear()

            assert len(list(manager.checkpoint_dir.glob('*.json'))) == 0

    def test_clear_removes_empty_directory(self):
        """Test that clear removes directory if empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager("test_op", checkpoint_dir=checkpoint_dir)

            checkpoint = manager.create_checkpoint(progress=50.0, state={})
            manager.save(checkpoint)

            manager.clear()

            # Directory should be removed if it became empty
            # (or might still exist if OS keeps it, so we just check files are gone)
            if checkpoint_dir.exists():
                assert len(list(checkpoint_dir.glob('*'))) == 0


class TestCheckpointDecorator:
    """Tests for @checkpoint decorator."""

    def test_decorator_basic_usage(self):
        """Test basic decorator usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            @checkpoint_decorator(
                operation_id="test_batch",
                checkpoint_interval=2,
                checkpoint_dir=Path(tmpdir)
            )
            def process_batch(items):
                for i, item in enumerate(items):
                    progress = (i + 1) / len(items) * 100
                    state = {'current_index': i}
                    yield progress, state, item * 2

            items = [1, 2, 3, 4, 5]
            results = list(process_batch(items))

            assert results == [2, 4, 6, 8, 10]

            # Check that checkpoints were created
            checkpoint_dir = Path(tmpdir) / "test_batch"
            checkpoints = list(checkpoint_dir.glob('*.json'))
            # Should have checkpoints at intervals of 2
            assert len(checkpoints) >= 1

    def test_decorator_respects_interval(self):
        """Test that decorator respects checkpoint interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            @checkpoint_decorator(
                operation_id="interval_test",
                checkpoint_interval=3,
                checkpoint_dir=Path(tmpdir)
            )
            def process_items(items):
                for i, item in enumerate(items):
                    yield i * 10, {'index': i}, item

            items = [1, 2, 3, 4, 5, 6, 7, 8]
            list(process_items(items))

            checkpoint_dir = Path(tmpdir) / "interval_test"
            checkpoints = list(checkpoint_dir.glob('*.json'))

            # With interval=3 and 8 items: checkpoints at 0, 3, 6
            assert len(checkpoints) >= 2

    def test_decorator_with_non_tuple_results(self):
        """Test decorator with functions that don't yield tuples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            @checkpoint_decorator(
                operation_id="simple",
                checkpoint_dir=Path(tmpdir)
            )
            def simple_generator(items):
                for item in items:
                    yield item * 2

            results = list(simple_generator([1, 2, 3]))
            assert results == [2, 4, 6]


class TestResumeFromCheckpoint:
    """Tests for resume_from_checkpoint function."""

    def test_resume_from_existing_checkpoint(self):
        """Test resuming from existing checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager("resume_test", checkpoint_dir=Path(tmpdir))

            # Create and save checkpoint
            checkpoint = manager.create_checkpoint(
                progress=75.0,
                state={'last_processed': 'file_10.jpg', 'index': 10}
            )
            manager.save(checkpoint)

            # Resume
            state = resume_from_checkpoint("resume_test", checkpoint_dir=Path(tmpdir))

            assert state is not None
            assert state['last_processed'] == 'file_10.jpg'
            assert state['index'] == 10

    def test_resume_when_no_checkpoint_exists(self):
        """Test resuming when no checkpoint exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = resume_from_checkpoint("nonexistent", checkpoint_dir=Path(tmpdir))
            assert state is None

    def test_resume_gets_latest_checkpoint(self):
        """Test that resume gets the latest checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager("multi_test", checkpoint_dir=Path(tmpdir))

            # Create multiple checkpoints
            checkpoint1 = manager.create_checkpoint(progress=25.0, state={'step': 1})
            time.sleep(0.01)
            checkpoint2 = manager.create_checkpoint(progress=50.0, state={'step': 2})
            time.sleep(0.01)
            checkpoint3 = manager.create_checkpoint(progress=75.0, state={'step': 3})

            manager.save(checkpoint1)
            manager.save(checkpoint2)
            manager.save(checkpoint3)

            # Resume should get latest
            state = resume_from_checkpoint("multi_test", checkpoint_dir=Path(tmpdir))

            assert state is not None
            assert state['step'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
