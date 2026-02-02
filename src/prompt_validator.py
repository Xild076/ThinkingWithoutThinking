"""
Prompt Validator Module

Validates prompt mutations to prevent training corruption from:
- Memorizing test case content
- Incorporating meta-instructions about training
- Injecting irrelevant technical jargon
- Semantic drift from original purpose
"""

import re
import logging
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)


class PromptValidator:
    """Validates prompt mutations before accepting them into the system."""
    
    # Maximum allowed semantic drift as ratio (0.0 = identical, 1.0 = completely different)
    MAX_DRIFT_RATIO = 0.5
    
    # Blocklist of terms that should NEVER appear in prompts
    # These indicate training artifacts or test case leakage
    BLOCKLIST = [
        # Training/CI artifacts
        "github actions", "ci/cd", "ci platform", "helm", "kubernetes", "docker",
        "pull request", "commit", "deployment", "visual diagram", "feature-branch",
        "branch workflow", "pipeline diagram",
        # Memorized test content  
        "matrix factorization", "union-find", "nuclear deterrence", "united nations formation",
        "cold-start mitigation", "batch-plus-stream", "deep learning embeddings",
        "redis caching", "sparse matrices", "directed graphs",
        # Meta-references to training process
        "stress test", "validation criteria", "expected answer", "test case",
        "training loop", "grader", "scoring", "evaluation rubric",
        # Suspiciously specific technical requirements
        "security scanning", "static analysis", "dynamic analysis",
        "dependency vulnerability", "scalable deployment",
    ]
    
    # Required placeholders per block - mutations MUST preserve these
    REQUIRED_PLACEHOLDERS = {
        "planner_prompt_block": ["{prompt}"],
        "tool_router_block": ["{tools}", "{prompt}", "{plan}"],
        "response_synthesizer_block": ["{prompt}", "{plan}", "{sources}"],
        "self_critique_block": ["{initial_task}", "{input}", "{output}"],
        "improvement_critique_block": ["{initial_task}", "{input}", "{output}", "{critique}"],
    }
    
    # Patterns that indicate test case leakage
    LEAKAGE_PATTERNS = [
        r'\(NASA,?\s*\d{4}\)',      # (NASA, 2024)
        r'\(ESA,?\s*\d{4}\)',       # (ESA, 2024)  
        r'\(\w+\s+et\s+al\.?,?\s*\d{4}\)',  # (Smith et al., 2024)
        r'https?://[^\s\)]+',       # URLs (should not be hardcoded in prompts)
        r'\d{1,2}\.\d{1,2}\.\d{1,4}',  # Version numbers like 1.2.3
        r'\$[\d,]+\.?\d*',          # Dollar amounts
    ]
    
    @classmethod
    def validate_mutation(
        cls, 
        block_id: str, 
        original: str, 
        mutated: str,
        strict: bool = True
    ) -> tuple[bool, list[str]]:
        """Validate a prompt mutation before accepting it.
        
        Args:
            block_id: The identifier of the prompt block being mutated
            original: The original prompt text
            mutated: The proposed mutated prompt text
            strict: If True, apply all validation rules. If False, only critical rules.
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # 1. Check blocklist terms
        blocklist_violations = cls._check_blocklist(mutated)
        violations.extend(blocklist_violations)
        
        # 2. Check required placeholders
        placeholder_violations = cls._check_placeholders(block_id, mutated)
        violations.extend(placeholder_violations)
        
        # 3. Check for test case leakage
        leakage_violations = cls._check_leakage(mutated)
        violations.extend(leakage_violations)
        
        if strict:
            # 4. Check semantic drift (only in strict mode)
            drift = cls._compute_drift(original, mutated)
            if drift > cls.MAX_DRIFT_RATIO:
                violations.append(
                    f"Excessive semantic drift: {drift:.1%} (max allowed: {cls.MAX_DRIFT_RATIO:.0%})"
                )
            
            # 5. Check for excessive length growth
            length_ratio = len(mutated) / max(len(original), 1)
            if length_ratio > 3.0:
                violations.append(
                    f"Excessive length growth: {length_ratio:.1f}x original size"
                )
        
        is_valid = len(violations) == 0
        
        if not is_valid:
            logger.warning(f"Prompt validation failed for {block_id}: {violations}")
        
        return is_valid, violations
    
    @classmethod
    def _check_blocklist(cls, text: str) -> list[str]:
        """Check for blocklisted terms."""
        violations = []
        text_lower = text.lower()
        
        for term in cls.BLOCKLIST:
            if term in text_lower:
                violations.append(f"Contains blocklisted term: '{term}'")
        
        return violations
    
    @classmethod
    def _check_placeholders(cls, block_id: str, text: str) -> list[str]:
        """Check that required placeholders are present."""
        violations = []
        required = cls.REQUIRED_PLACEHOLDERS.get(block_id, [])
        
        for placeholder in required:
            if placeholder not in text:
                violations.append(f"Missing required placeholder: {placeholder}")
        
        return violations
    
    @classmethod
    def _check_leakage(cls, text: str) -> list[str]:
        """Check for patterns indicating test case leakage."""
        violations = []
        
        for pattern in cls.LEAKAGE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                violations.append(
                    f"Contains leaked test content pattern: {matches[:2]}"
                )
        
        return violations
    
    @classmethod
    def _compute_drift(cls, original: str, mutated: str) -> float:
        """Compute normalized semantic drift between prompts.
        
        Returns:
            Float between 0.0 (identical) and 1.0 (completely different)
        """
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, original, mutated).ratio()
        return 1.0 - similarity
    
    @classmethod
    def validate_response_citations(
        cls,
        response: str,
        sources: str,
        strict: bool = True
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Validate that response citations are grounded in actual sources.
        
        Args:
            response: The synthesized response text
            sources: The actual sources/evidence provided
            strict: If True, flag any citation without sources as hallucination
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if we have real sources
        has_sources = bool(sources and sources.strip())
        no_sources_marker = "[NO EXTERNAL SOURCES" in sources if sources else True
        
        if not has_sources or no_sources_marker:
            # No sources were provided - check for hallucinated citations
            
            # Check for URLs
            urls = re.findall(r'https?://[^\s\)\]]+', response)
            if urls:
                issues.append({
                    "type": "hallucinated_url",
                    "count": len(urls),
                    "examples": urls[:3],
                    "severity": "critical"
                })
            
            # Check for academic-style citations (Author, Year)
            citations = re.findall(
                r'\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?,?\s*\d{4}\)', 
                response
            )
            if citations:
                issues.append({
                    "type": "hallucinated_citation",
                    "count": len(citations),
                    "examples": citations[:3],
                    "severity": "critical"
                })
            
            # Check for source attributions like "according to NASA"
            attributions = re.findall(
                r'(?:according to|per|source:|from)\s+([A-Z][A-Za-z\s]+?)(?:\s+\(|\s*,|\s+report)',
                response,
                re.IGNORECASE
            )
            if attributions:
                issues.append({
                    "type": "hallucinated_attribution",
                    "count": len(attributions),
                    "examples": attributions[:3],
                    "severity": "high"
                })
            
            # Check for specific statistics without sources
            if strict:
                stats = re.findall(
                    r'\d+(?:\.\d+)?(?:\s*%|\s+percent|\s+million|\s+billion)',
                    response
                )
                # Only flag if there are many specific numbers
                if len(stats) > 3:
                    issues.append({
                        "type": "unverified_statistics",
                        "count": len(stats),
                        "examples": stats[:5],
                        "severity": "medium"
                    })
        
        is_valid = len([i for i in issues if i.get("severity") == "critical"]) == 0
        return is_valid, issues
    
    @classmethod
    def strip_hallucinated_citations(cls, response: str) -> str:
        """Remove hallucinated citations from a response.
        
        Args:
            response: The response text to clean
            
        Returns:
            Cleaned response with citations removed and disclaimer added
        """
        original = response
        cleaned = response
        
        # Remove URLs
        cleaned = re.sub(r'https?://[^\s\)\]]+', '[source unavailable]', cleaned)
        
        # Remove academic citations like (NASA, 2024), (Smith et al., 2023), (Author, Year)
        cleaned = re.sub(
            r'\s*\([A-Z][A-Za-z]*(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?,?\s*\d{4}\)', 
            '', 
            cleaned
        )
        
        # Remove inline citations like "NASA (2024)" or "according to NASA"
        cleaned = re.sub(
            r'\s+\(\d{4}\)',  # Just year in parens
            '',
            cleaned
        )
        
        # Clean up any doubled spaces
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        
        # Add disclaimer if we made changes
        if cleaned != original:
            disclaimer = (
                "\n\n*Note: This response is based on general knowledge. "
                "No external sources were consulted or verified.*"
            )
            if disclaimer not in cleaned:
                cleaned += disclaimer
        
        return cleaned.strip()


def validate_and_filter_improvements(
    current_prompts: dict[str, str],
    proposed_improvements: dict[str, str],
    strict: bool = True
) -> dict[str, str]:
    """Validate proposed prompt improvements and filter out invalid ones.
    
    Args:
        current_prompts: Current prompt dictionary
        proposed_improvements: Dictionary of proposed prompt changes
        strict: Whether to apply strict validation
        
    Returns:
        Filtered dictionary containing only valid improvements
    """
    valid_improvements = {}
    
    for block_id, proposed_prompt in proposed_improvements.items():
        original_prompt = current_prompts.get(block_id, "")
        
        is_valid, violations = PromptValidator.validate_mutation(
            block_id, original_prompt, proposed_prompt, strict=strict
        )
        
        if is_valid:
            valid_improvements[block_id] = proposed_prompt
            logger.info(f"Accepted improvement for {block_id}")
        else:
            logger.warning(
                f"Rejected improvement for {block_id}: {len(violations)} violations"
            )
            for v in violations:
                logger.warning(f"  - {v}")
    
    return valid_improvements
