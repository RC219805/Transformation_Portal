"""
Security Management Module for Transformation Portal
=====================================================

Proactive Dependabot security warning management with RAG integration
and self-improving automated resolution.

This package provides:
- Vulnerability scanning and tracking
- Automated mitigation suggestions
- RAG-based security knowledge base
- CI/CD security verification
- Self-improving auto-resolution system
- Continuous security verification

Usage:
    # Basic security management
    from scripts.security import SecurityManager
    manager = SecurityManager()
    report = manager.scan()

    # Automated resolution
    from scripts.security import AutoSecurityResolver
    resolver = AutoSecurityResolver()
    attempt = resolver.resolve(
        package_name="basicsr",
        vulnerability_id="CVE-2024-27763",
        vulnerability_type="command_injection"
    )

    # Continuous verification
    from scripts.security import ContinuousSecurityVerifier
    verifier = ContinuousSecurityVerifier()
    report = verifier.quick_check()  # For pre-commit
    report = verifier.full_audit()   # For CI/CD
"""

from .security_manager import (
    SecurityManager,
    DependencyScanner,
    MitigationEngine,
    SecurityKnowledgeBase,
    Vulnerability,
    Mitigation,
    SecurityPolicy,
    SecurityReport,
    SeverityLevel,
    MitigationStatus,
    MitigationType,
)

from .auto_resolver import (
    AutoSecurityResolver,
    PatternLearner,
    AutoFixer,
    FeedbackCollector,
    ResolutionPattern,
    ResolutionAttempt,
    SecurityFeedback,
    ResolutionStrategy,
    ResolutionStatus,
    ConfidenceLevel,
)

from .continuous_security import (
    ContinuousSecurityVerifier,
    ImportScanner,
    PackageAuditor,
    ConstraintVerifier,
    CodePatternScanner,
    SecurityCheckResult,
    SecurityHealthReport,
    SecurityCheckType,
    SecurityStatus,
    security_guard,
)

__all__ = [
    # Security Manager
    "SecurityManager",
    "DependencyScanner",
    "MitigationEngine",
    "SecurityKnowledgeBase",
    "Vulnerability",
    "Mitigation",
    "SecurityPolicy",
    "SecurityReport",
    "SeverityLevel",
    "MitigationStatus",
    "MitigationType",
    # Auto Resolver
    "AutoSecurityResolver",
    "PatternLearner",
    "AutoFixer",
    "FeedbackCollector",
    "ResolutionPattern",
    "ResolutionAttempt",
    "SecurityFeedback",
    "ResolutionStrategy",
    "ResolutionStatus",
    "ConfidenceLevel",
    # Continuous Security
    "ContinuousSecurityVerifier",
    "ImportScanner",
    "PackageAuditor",
    "ConstraintVerifier",
    "CodePatternScanner",
    "SecurityCheckResult",
    "SecurityHealthReport",
    "SecurityCheckType",
    "SecurityStatus",
    "security_guard",
]
