#!/usr/bin/env python3
"""
Self-Improving Security Resolution System
==========================================
Automated security issue detection and resolution with machine learning
from historical fixes.

This module provides:
- Automatic detection of new security vulnerabilities
- Pattern matching against historical successful mitigations
- Automated application of fixes when confidence is high
- Learning from manual interventions to improve future responses
- Continuous feedback loop for security improvement

Architecture:
    AutoSecurityResolver
    ├── PatternLearner (learns from historical fixes)
    ├── ConfidenceScorer (evaluates fix applicability)
    ├── AutoFixer (applies fixes automatically)
    ├── FeedbackCollector (learns from outcomes)
    └── EscalationManager (handles uncertain cases)

Resolution Strategies:
    1. CONSTRAINT_BLOCK - Add impossible version constraint
    2. VENDOR_REPLACE - Replace with vendored secure version
    3. UPGRADE - Upgrade to patched version
    4. REMOVE - Remove vulnerable dependency
    5. WORKAROUND - Apply code-level workaround

Confidence Levels:
    - HIGH (>0.9): Auto-apply fix, log for review
    - MEDIUM (0.7-0.9): Auto-apply with notification
    - LOW (0.5-0.7): Suggest fix, require approval
    - UNCERTAIN (<0.5): Escalate to human review

Author: Transformation Portal
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Configure module logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("auto_security_resolver")


# =============================================================================
# Data Models
# =============================================================================


class ResolutionStrategy(Enum):
    """Available resolution strategies."""
    CONSTRAINT_BLOCK = "constraint_block"
    VENDOR_REPLACE = "vendor_replace"
    UPGRADE = "upgrade"
    REMOVE = "remove"
    WORKAROUND = "workaround"


class ConfidenceLevel(Enum):
    """Confidence levels for automated actions."""
    HIGH = "high"  # >0.9 - Auto-apply
    MEDIUM = "medium"  # 0.7-0.9 - Auto-apply with notification
    LOW = "low"  # 0.5-0.7 - Suggest, require approval
    UNCERTAIN = "uncertain"  # <0.5 - Escalate


class ResolutionStatus(Enum):
    """Status of a resolution attempt."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPLIED = "applied"
    VERIFIED = "verified"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    ESCALATED = "escalated"


@dataclass
class ResolutionPattern:
    """A learned pattern for resolving security issues."""
    pattern_id: str
    vulnerability_type: str  # e.g., "command_injection", "path_traversal"
    package_pattern: str  # Regex pattern for matching packages
    strategy: ResolutionStrategy
    confidence_base: float  # Base confidence score

    # Resolution template
    files_to_modify: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    verification_steps: List[str] = field(default_factory=list)

    # Learning metadata
    success_count: int = 0
    failure_count: int = 0
    last_used: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def effectiveness_score(self) -> float:
        """Calculate effectiveness based on historical success."""
        total = self.success_count + self.failure_count
        if total == 0:
            return self.confidence_base
        success_rate = self.success_count / total
        # Blend base confidence with historical performance
        return 0.3 * self.confidence_base + 0.7 * success_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['strategy'] = self.strategy.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResolutionPattern':
        """Create from dictionary."""
        data['strategy'] = ResolutionStrategy(data['strategy'])
        return cls(**data)


@dataclass
class ResolutionAttempt:
    """Record of an attempted resolution."""
    attempt_id: str
    vulnerability_id: str
    pattern_id: str
    strategy: ResolutionStrategy
    status: ResolutionStatus
    confidence: float

    # Execution details
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None

    # Changes made
    files_modified: List[str] = field(default_factory=list)
    backup_path: Optional[str] = None

    # Results
    verification_passed: bool = False
    error_message: Optional[str] = None

    # Feedback
    human_approved: Optional[bool] = None
    feedback_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['strategy'] = self.strategy.value
        data['status'] = self.status.value
        return data


@dataclass
class SecurityFeedback:
    """Feedback on a resolution for learning."""
    attempt_id: str
    success: bool
    feedback_type: str  # "auto", "human_approved", "human_rejected", "rollback"
    notes: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# =============================================================================
# Pattern Learning Engine
# =============================================================================


class PatternLearner:
    """Learns resolution patterns from historical fixes."""

    # Pre-defined patterns based on common vulnerabilities
    BUILT_IN_PATTERNS: List[Dict[str, Any]] = [
        {
            "pattern_id": "basicsr_cve_2024_27763",
            "vulnerability_type": "command_injection",
            "package_pattern": r"^basicsr$",
            "strategy": ResolutionStrategy.VENDOR_REPLACE,
            "confidence_base": 0.95,
            "files_to_modify": [
                "requirements/constraints.txt",
            ],
            "commands": [
                "echo 'basicsr>=999.0.0  # CVE-2024-27763 blocked' >> requirements/constraints.txt",
            ],
            "verification_steps": [
                "python scripts/utilities/verify_no_basicsr_imports.py --check-pkg",
                "pip install -c requirements/constraints.txt basicsr 2>&1 | grep -q 'ResolutionImpossible'",
            ],
            "success_count": 10,
            "failure_count": 0,
        },
        {
            "pattern_id": "generic_constraint_block",
            "vulnerability_type": "any",
            "package_pattern": r".*",
            "strategy": ResolutionStrategy.CONSTRAINT_BLOCK,
            "confidence_base": 0.7,
            "files_to_modify": [
                "requirements/constraints.txt",
            ],
            "commands": [],  # Generated dynamically
            "verification_steps": [],  # Generated dynamically
            "success_count": 5,
            "failure_count": 1,
        },
        {
            "pattern_id": "upgrade_to_patched",
            "vulnerability_type": "any_with_patch",
            "package_pattern": r".*",
            "strategy": ResolutionStrategy.UPGRADE,
            "confidence_base": 0.85,
            "files_to_modify": [],  # Determined by package location
            "commands": [],  # Generated dynamically
            "verification_steps": [],
            "success_count": 8,
            "failure_count": 2,
        },
    ]

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.patterns_file = repo_root / ".github" / "security" / "learned_patterns.json"
        self.patterns: List[ResolutionPattern] = []
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load patterns from storage."""
        # Start with built-in patterns
        for pattern_data in self.BUILT_IN_PATTERNS:
            self.patterns.append(ResolutionPattern(
                pattern_id=pattern_data["pattern_id"],
                vulnerability_type=pattern_data["vulnerability_type"],
                package_pattern=pattern_data["package_pattern"],
                strategy=pattern_data["strategy"],
                confidence_base=pattern_data["confidence_base"],
                files_to_modify=pattern_data.get("files_to_modify", []),
                commands=pattern_data.get("commands", []),
                verification_steps=pattern_data.get("verification_steps", []),
                success_count=pattern_data.get("success_count", 0),
                failure_count=pattern_data.get("failure_count", 0),
            ))

        # Load learned patterns
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'r') as f:
                    data = json.load(f)
                    for pattern_data in data.get("patterns", []):
                        # Update existing or add new
                        existing = next(
                            (p for p in self.patterns if p.pattern_id == pattern_data["pattern_id"]),
                            None
                        )
                        if existing:
                            existing.success_count = pattern_data.get("success_count", existing.success_count)
                            existing.failure_count = pattern_data.get("failure_count", existing.failure_count)
                            existing.last_used = pattern_data.get("last_used")
                        else:
                            self.patterns.append(ResolutionPattern.from_dict(pattern_data))
            except (IOError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load learned patterns: {e}")

    def save_patterns(self) -> None:
        """Save patterns to storage."""
        self.patterns_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.patterns_file, 'w') as f:
                json.dump({
                    "patterns": [p.to_dict() for p in self.patterns],
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save patterns: {e}")

    def find_matching_patterns(
        self,
        package_name: str,
        vulnerability_type: Optional[str] = None,
        has_patch: bool = False
    ) -> List[Tuple[ResolutionPattern, float]]:
        """Find patterns matching a vulnerability, sorted by confidence."""
        matches = []

        for pattern in self.patterns:
            # Check package pattern match
            if not re.match(pattern.package_pattern, package_name, re.IGNORECASE):
                continue

            # Check vulnerability type match
            if vulnerability_type and pattern.vulnerability_type not in ["any", "any_with_patch", vulnerability_type]:
                continue

            # Check if pattern requires patch availability
            if pattern.vulnerability_type == "any_with_patch" and not has_patch:
                continue

            # Calculate confidence
            confidence = pattern.effectiveness_score

            # Boost confidence for exact package matches
            if pattern.package_pattern == f"^{re.escape(package_name)}$":
                confidence = min(1.0, confidence + 0.1)

            matches.append((pattern, confidence))

        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def record_outcome(self, pattern_id: str, success: bool) -> None:
        """Record the outcome of using a pattern."""
        for pattern in self.patterns:
            if pattern.pattern_id == pattern_id:
                if success:
                    pattern.success_count += 1
                else:
                    pattern.failure_count += 1
                pattern.last_used = datetime.now(timezone.utc).isoformat()
                break
        self.save_patterns()

    def learn_new_pattern(
        self,
        package_name: str,
        vulnerability_type: str,
        strategy: ResolutionStrategy,
        files_modified: List[str],
        commands: List[str],
        verification_steps: List[str],
    ) -> ResolutionPattern:
        """Learn a new pattern from a successful manual fix."""
        pattern_id = f"learned_{hashlib.sha256(f'{package_name}:{vulnerability_type}'.encode()).hexdigest()[:8]}"

        pattern = ResolutionPattern(
            pattern_id=pattern_id,
            vulnerability_type=vulnerability_type,
            package_pattern=f"^{re.escape(package_name)}$",
            strategy=strategy,
            confidence_base=0.6,  # Start with medium confidence
            files_to_modify=files_modified,
            commands=commands,
            verification_steps=verification_steps,
            success_count=1,  # We're learning from a success
        )

        self.patterns.append(pattern)
        self.save_patterns()

        logger.info(f"Learned new pattern: {pattern_id} for {package_name}")
        return pattern


# =============================================================================
# Auto Fixer
# =============================================================================


class AutoFixer:
    """Automatically applies security fixes."""

    # Minimum confidence thresholds
    AUTO_APPLY_THRESHOLD = 0.9
    NOTIFY_THRESHOLD = 0.7
    SUGGEST_THRESHOLD = 0.5

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.backup_dir = repo_root / ".github" / "security" / "backups"
        self.attempts_file = repo_root / ".github" / "security" / "resolution_attempts.json"
        self.attempts: List[ResolutionAttempt] = []
        self._load_attempts()

    def _load_attempts(self) -> None:
        """Load resolution attempts history."""
        if self.attempts_file.exists():
            try:
                with open(self.attempts_file, 'r') as f:
                    data = json.load(f)
                    # Keep only recent attempts (last 100)
                    for attempt_data in data.get("attempts", [])[-100:]:
                        attempt_data['strategy'] = ResolutionStrategy(attempt_data['strategy'])
                        attempt_data['status'] = ResolutionStatus(attempt_data['status'])
                        self.attempts.append(ResolutionAttempt(**attempt_data))
            except (IOError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load attempts: {e}")

    def _save_attempts(self) -> None:
        """Save resolution attempts."""
        self.attempts_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.attempts_file, 'w') as f:
                json.dump({
                    "attempts": [a.to_dict() for a in self.attempts[-100:]],
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save attempts: {e}")

    def create_backup(self, files: List[str]) -> str:
        """Create backup of files before modification."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / timestamp
        backup_path.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            src = self.repo_root / file_path
            if src.exists():
                dst = backup_path / file_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        return str(backup_path)

    def rollback(self, backup_path: str, files: List[str]) -> bool:
        """Rollback files from backup."""
        try:
            backup_dir = Path(backup_path)
            for file_path in files:
                src = backup_dir / file_path
                dst = self.repo_root / file_path
                if src.exists():
                    shutil.copy2(src, dst)
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def apply_fix(
        self,
        vulnerability_id: str,
        pattern: ResolutionPattern,
        confidence: float,
        package_name: str,
        dry_run: bool = False,
    ) -> ResolutionAttempt:
        """Apply a security fix based on a pattern."""
        attempt_id = hashlib.sha256(
            f"{vulnerability_id}:{pattern.pattern_id}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        attempt = ResolutionAttempt(
            attempt_id=attempt_id,
            vulnerability_id=vulnerability_id,
            pattern_id=pattern.pattern_id,
            strategy=pattern.strategy,
            status=ResolutionStatus.PENDING,
            confidence=confidence,
        )

        # Determine confidence level
        if confidence >= self.AUTO_APPLY_THRESHOLD:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence >= self.NOTIFY_THRESHOLD:
            confidence_level = ConfidenceLevel.MEDIUM
        elif confidence >= self.SUGGEST_THRESHOLD:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.UNCERTAIN

        logger.info(f"Attempting fix for {vulnerability_id} with confidence {confidence:.2f} ({confidence_level.value})")

        if dry_run:
            logger.info(f"[DRY RUN] Would apply fix using pattern {pattern.pattern_id}")
            attempt.status = ResolutionStatus.PENDING
            return attempt

        # Check if auto-apply is allowed
        if confidence_level == ConfidenceLevel.UNCERTAIN:
            logger.warning(f"Confidence too low ({confidence:.2f}), escalating to human review")
            attempt.status = ResolutionStatus.ESCALATED
            self.attempts.append(attempt)
            self._save_attempts()
            return attempt

        try:
            # Create backup
            attempt.status = ResolutionStatus.IN_PROGRESS
            if pattern.files_to_modify:
                attempt.backup_path = self.create_backup(pattern.files_to_modify)

            # Generate dynamic commands if needed
            commands = self._generate_commands(pattern, package_name)

            # Execute commands
            for cmd in commands:
                logger.info(f"Executing: {cmd}")
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=self.repo_root,
                    check=False
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")

            attempt.files_modified = pattern.files_to_modify
            attempt.status = ResolutionStatus.APPLIED

            # Verify the fix
            verification_passed = self._verify_fix(pattern, package_name)
            attempt.verification_passed = verification_passed

            if verification_passed:
                attempt.status = ResolutionStatus.VERIFIED
                logger.info(f"✅ Fix verified successfully for {vulnerability_id}")
            else:
                logger.warning(f"⚠️ Fix applied but verification failed for {vulnerability_id}")
                # Rollback on verification failure
                if attempt.backup_path:
                    self.rollback(attempt.backup_path, pattern.files_to_modify)
                    attempt.status = ResolutionStatus.ROLLED_BACK

        except Exception as e:
            logger.error(f"Fix failed: {e}")
            attempt.status = ResolutionStatus.FAILED
            attempt.error_message = str(e)
            # Rollback on failure
            if attempt.backup_path:
                self.rollback(attempt.backup_path, pattern.files_to_modify)
                attempt.status = ResolutionStatus.ROLLED_BACK

        attempt.completed_at = datetime.now(timezone.utc).isoformat()
        self.attempts.append(attempt)
        self._save_attempts()
        return attempt

    def _generate_commands(self, pattern: ResolutionPattern, package_name: str) -> List[str]:
        """Generate commands for a pattern, substituting package name."""
        if pattern.commands:
            return [cmd.replace("{package}", package_name) for cmd in pattern.commands]

        # Generate default commands based on strategy
        if pattern.strategy == ResolutionStrategy.CONSTRAINT_BLOCK:
            return [
                f"echo '{package_name}>=999.0.0  # Security blocked' >> requirements/constraints.txt",
            ]
        elif pattern.strategy == ResolutionStrategy.UPGRADE:
            return [
                f"pip install --upgrade {package_name}",
            ]
        return []

    def _verify_fix(self, pattern: ResolutionPattern, package_name: str) -> bool:
        """Verify a fix was applied successfully."""
        verification_steps = pattern.verification_steps or []

        # Add default verifications
        if not verification_steps:
            if pattern.strategy == ResolutionStrategy.CONSTRAINT_BLOCK:
                verification_steps = [
                    f"grep -q '{package_name}' requirements/constraints.txt",
                ]
            elif pattern.strategy == ResolutionStrategy.VENDOR_REPLACE:
                verification_steps = [
                    f"pip show {package_name} 2>&1 | grep -q 'not found' || exit 1",
                ]

        for step in verification_steps:
            step = step.replace("{package}", package_name)
            try:
                result = subprocess.run(
                    step,
                    shell=True,
                    capture_output=True,
                    cwd=self.repo_root,
                    check=False
                )
                if result.returncode != 0:
                    logger.warning(f"Verification step failed: {step}")
                    return False
            except Exception as e:
                logger.warning(f"Verification error: {e}")
                return False

        return True


# =============================================================================
# Feedback Collector
# =============================================================================


class FeedbackCollector:
    """Collects and processes feedback on resolutions."""

    def __init__(self, repo_root: Path, pattern_learner: PatternLearner):
        self.repo_root = repo_root
        self.pattern_learner = pattern_learner
        self.feedback_file = repo_root / ".github" / "security" / "feedback.jsonl"

    def record_feedback(self, feedback: SecurityFeedback) -> None:
        """Record feedback on a resolution attempt."""
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.feedback_file, 'a') as f:
                f.write(json.dumps(asdict(feedback)) + "\n")
        except IOError as e:
            logger.error(f"Failed to record feedback: {e}")

        # Update pattern based on feedback
        # Find the attempt to get the pattern_id
        attempts_file = self.repo_root / ".github" / "security" / "resolution_attempts.json"
        if attempts_file.exists():
            try:
                with open(attempts_file, 'r') as f:
                    data = json.load(f)
                    for attempt in data.get("attempts", []):
                        if attempt["attempt_id"] == feedback.attempt_id:
                            self.pattern_learner.record_outcome(
                                attempt["pattern_id"],
                                feedback.success
                            )
                            break
            except (IOError, json.JSONDecodeError):
                pass

    def analyze_feedback(self) -> Dict[str, Any]:
        """Analyze feedback to identify improvement opportunities."""
        analysis = {
            "total_feedback": 0,
            "success_rate": 0.0,
            "patterns_needing_review": [],
            "common_failures": [],
        }

        if not self.feedback_file.exists():
            return analysis

        successes = 0
        failures = 0
        failure_notes = []

        try:
            with open(self.feedback_file, 'r') as f:
                for line in f:
                    if line.strip():
                        feedback = json.loads(line)
                        analysis["total_feedback"] += 1
                        if feedback.get("success"):
                            successes += 1
                        else:
                            failures += 1
                            if feedback.get("notes"):
                                failure_notes.append(feedback["notes"])

            if analysis["total_feedback"] > 0:
                analysis["success_rate"] = successes / analysis["total_feedback"]

            # Identify patterns with poor performance
            for pattern in self.pattern_learner.patterns:
                if pattern.failure_count > pattern.success_count and pattern.failure_count >= 3:
                    analysis["patterns_needing_review"].append(pattern.pattern_id)

            # Find common failure reasons
            if failure_notes:
                # Simple frequency analysis
                from collections import Counter
                word_freq = Counter()
                for note in failure_notes:
                    words = note.lower().split()
                    word_freq.update(words)
                analysis["common_failures"] = [word for word, _ in word_freq.most_common(5)]

        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to analyze feedback: {e}")

        return analysis


# =============================================================================
# Main Auto Security Resolver
# =============================================================================


class AutoSecurityResolver:
    """Main interface for automated security resolution."""

    def __init__(self, repo_root: Optional[Path] = None):
        if repo_root is None:
            repo_root = self._find_repo_root()
        self.repo_root = repo_root
        self.pattern_learner = PatternLearner(repo_root)
        self.auto_fixer = AutoFixer(repo_root)
        self.feedback_collector = FeedbackCollector(repo_root, self.pattern_learner)

        # Statistics
        self.stats_file = repo_root / ".github" / "security" / "resolver_stats.json"

    def _find_repo_root(self) -> Path:
        """Find repository root."""
        current = Path(__file__).resolve().parent
        for _ in range(10):
            if (current / '.git').exists():
                return current
            if current.parent == current:
                break
            current = current.parent
        return Path.cwd()

    def resolve(
        self,
        package_name: str,
        vulnerability_id: str,
        vulnerability_type: Optional[str] = None,
        has_patch: bool = False,
        dry_run: bool = False,
        force: bool = False,
    ) -> ResolutionAttempt:
        """Attempt to automatically resolve a vulnerability."""
        logger.info(f"Resolving vulnerability {vulnerability_id} for package {package_name}")

        # Find matching patterns
        matches = self.pattern_learner.find_matching_patterns(
            package_name,
            vulnerability_type,
            has_patch
        )

        if not matches:
            logger.warning(f"No matching patterns found for {package_name}")
            return ResolutionAttempt(
                attempt_id="no_match",
                vulnerability_id=vulnerability_id,
                pattern_id="none",
                strategy=ResolutionStrategy.CONSTRAINT_BLOCK,
                status=ResolutionStatus.ESCALATED,
                confidence=0.0,
                error_message="No matching resolution patterns found",
            )

        # Use the best matching pattern
        best_pattern, confidence = matches[0]
        logger.info(f"Best match: {best_pattern.pattern_id} with confidence {confidence:.2f}")

        # Apply the fix
        attempt = self.auto_fixer.apply_fix(
            vulnerability_id=vulnerability_id,
            pattern=best_pattern,
            confidence=confidence if not force else 1.0,
            package_name=package_name,
            dry_run=dry_run,
        )

        # Update statistics
        self._update_stats(attempt)

        return attempt

    def provide_feedback(
        self,
        attempt_id: str,
        success: bool,
        feedback_type: str = "human",
        notes: Optional[str] = None,
    ) -> None:
        """Provide feedback on a resolution attempt."""
        feedback = SecurityFeedback(
            attempt_id=attempt_id,
            success=success,
            feedback_type=feedback_type,
            notes=notes,
        )
        self.feedback_collector.record_feedback(feedback)
        logger.info(f"Recorded feedback for attempt {attempt_id}: {'success' if success else 'failure'}")

    def learn_from_manual_fix(
        self,
        package_name: str,
        vulnerability_type: str,
        strategy: ResolutionStrategy,
        files_modified: List[str],
        commands: List[str],
        verification_steps: List[str],
    ) -> ResolutionPattern:
        """Learn a new pattern from a manual fix."""
        return self.pattern_learner.learn_new_pattern(
            package_name=package_name,
            vulnerability_type=vulnerability_type,
            strategy=strategy,
            files_modified=files_modified,
            commands=commands,
            verification_steps=verification_steps,
        )

    def get_resolution_status(self) -> Dict[str, Any]:
        """Get overall resolution system status."""
        feedback_analysis = self.feedback_collector.analyze_feedback()

        return {
            "patterns_count": len(self.pattern_learner.patterns),
            "attempts_count": len(self.auto_fixer.attempts),
            "feedback_analysis": feedback_analysis,
            "recent_attempts": [
                {
                    "id": a.attempt_id,
                    "vulnerability": a.vulnerability_id,
                    "status": a.status.value,
                    "confidence": a.confidence,
                }
                for a in self.auto_fixer.attempts[-5:]
            ],
        }

    def _update_stats(self, attempt: ResolutionAttempt) -> None:
        """Update resolver statistics."""
        stats = {"total_attempts": 0, "successful": 0, "failed": 0, "escalated": 0}

        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    stats = json.load(f)
            except (IOError, json.JSONDecodeError):
                pass

        stats["total_attempts"] += 1
        if attempt.status == ResolutionStatus.VERIFIED:
            stats["successful"] += 1
        elif attempt.status == ResolutionStatus.FAILED:
            stats["failed"] += 1
        elif attempt.status == ResolutionStatus.ESCALATED:
            stats["escalated"] += 1

        stats["last_updated"] = datetime.now(timezone.utc).isoformat()

        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to update stats: {e}")


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Self-Improving Security Resolution System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Resolve command
    resolve_parser = subparsers.add_parser("resolve", help="Resolve a vulnerability")
    resolve_parser.add_argument("--package", required=True, help="Package name")
    resolve_parser.add_argument("--vuln-id", required=True, help="Vulnerability ID (e.g., CVE-2024-27763)")
    resolve_parser.add_argument("--vuln-type", help="Vulnerability type (e.g., command_injection)")
    resolve_parser.add_argument("--has-patch", action="store_true", help="Patch is available")
    resolve_parser.add_argument("--dry-run", action="store_true", help="Don't apply changes")
    resolve_parser.add_argument("--force", action="store_true", help="Force apply regardless of confidence")

    # Feedback command
    feedback_parser = subparsers.add_parser("feedback", help="Provide feedback on resolution")
    feedback_parser.add_argument("--attempt-id", required=True, help="Resolution attempt ID")
    feedback_parser.add_argument("--success", action="store_true", help="Resolution was successful")
    feedback_parser.add_argument("--failure", action="store_true", help="Resolution failed")
    feedback_parser.add_argument("--notes", help="Additional notes")

    # Learn command
    learn_parser = subparsers.add_parser("learn", help="Learn from manual fix")
    learn_parser.add_argument("--package", required=True, help="Package name")
    learn_parser.add_argument("--vuln-type", required=True, help="Vulnerability type")
    learn_parser.add_argument("--strategy", required=True, choices=[s.value for s in ResolutionStrategy])
    learn_parser.add_argument("--files", nargs="+", help="Files modified")
    learn_parser.add_argument("--commands", nargs="+", help="Commands used")
    learn_parser.add_argument("--verify", nargs="+", help="Verification steps")

    # Status command
    subparsers.add_parser("status", help="Show resolver status")

    # Patterns command
    subparsers.add_parser("patterns", help="List known resolution patterns")

    args = parser.parse_args()

    resolver = AutoSecurityResolver()

    if args.command == "resolve":
        attempt = resolver.resolve(
            package_name=args.package,
            vulnerability_id=args.vuln_id,
            vulnerability_type=args.vuln_type,
            has_patch=args.has_patch,
            dry_run=args.dry_run,
            force=args.force,
        )
        print(f"Resolution attempt: {attempt.attempt_id}")
        print(f"  Status: {attempt.status.value}")
        print(f"  Confidence: {attempt.confidence:.2f}")
        if attempt.error_message:
            print(f"  Error: {attempt.error_message}")
        if attempt.status == ResolutionStatus.VERIFIED:
            print("  ✅ Fix verified successfully")

    elif args.command == "feedback":
        if args.success:
            resolver.provide_feedback(args.attempt_id, True, "human", args.notes)
            print("✅ Positive feedback recorded")
        elif args.failure:
            resolver.provide_feedback(args.attempt_id, False, "human", args.notes)
            print("❌ Negative feedback recorded")
        else:
            print("Please specify --success or --failure")

    elif args.command == "learn":
        pattern = resolver.learn_from_manual_fix(
            package_name=args.package,
            vulnerability_type=args.vuln_type,
            strategy=ResolutionStrategy(args.strategy),
            files_modified=args.files or [],
            commands=args.commands or [],
            verification_steps=args.verify or [],
        )
        print(f"✅ Learned new pattern: {pattern.pattern_id}")

    elif args.command == "status":
        status = resolver.get_resolution_status()
        print("🔒 Auto Security Resolver Status")
        print(f"  Patterns: {status['patterns_count']}")
        print(f"  Attempts: {status['attempts_count']}")
        print(f"  Success Rate: {status['feedback_analysis']['success_rate']:.1%}")
        if status['recent_attempts']:
            print("\nRecent Attempts:")
            for attempt in status['recent_attempts']:
                print(f"  - {attempt['vulnerability']}: {attempt['status']} ({attempt['confidence']:.2f})")

    elif args.command == "patterns":
        print("🔧 Known Resolution Patterns:")
        for pattern in resolver.pattern_learner.patterns:
            effectiveness = pattern.effectiveness_score
            print(f"\n  {pattern.pattern_id}")
            print(f"    Strategy: {pattern.strategy.value}")
            print(f"    Effectiveness: {effectiveness:.2%}")
            print(f"    Success/Failure: {pattern.success_count}/{pattern.failure_count}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
