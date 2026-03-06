import re
import sys
import logging
from dataclasses import dataclass
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from llm_provider import get_llm_client
from latex_utils import (
    escape_latex_keywords,
    count_keyword_occurrences,
    count_words_strict,
    count_characters_strict,
    validate_structure_basic,
    validate_full_resume
)

logger = logging.getLogger(__name__)

@dataclass
class ResumeSection:
    """Represents a major section of the resume"""
    name: str
    content: str
    start_line: int
    end_line: int
    original_content: str
    existing_keywords: List[str]

class ResumeOptimizer:
    def __init__(self):
        logger.info("Initializing LLM client...")
        self.client = get_llm_client()

    def extract_keywords_from_jd(self, job_description: str) -> str:
        """Extract relevant keywords from job description"""
        system_prompt = """Extract 8-10 most important technical keywords from this job description. 
        Focus on: programming languages, frameworks, technical skills, tools.
        Return only a comma-separated list."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Job Description:\n{job_description}")
        ]

        response = self.client.invoke(messages)
        keywords = response.content.strip()
        logger.info(f"Extracted keywords: {keywords}")
        return keywords

    def parse_keywords(self, keyword_content: str) -> List[str]:
        """Parse keywords from various formats"""
        if ',' in keyword_content:
            keywords = [kw.strip() for kw in keyword_content.split(',') if kw.strip()]
        else:
            keywords = [kw.strip() for kw in keyword_content.split('\n') if kw.strip()]

        # Escape LaTeX special characters
        return escape_latex_keywords(keywords)

    def extract_existing_keywords_in_component(self, component_content: str, target_keywords: List[str]) -> List[str]:
        """Extract keywords that already exist in a specific component using sophisticated matching"""
        existing = []

        for keyword in target_keywords:
            count = count_keyword_occurrences(component_content, keyword)
            if count > 0:
                existing.append(keyword)

        return existing

    def detect_changeable_components(self, latex_content: str, target_keywords: List[str]) -> List[ResumeSection]:
        """Use LLM to detect which components can be rewritten"""

        # Extract all bullet points from relevant sections
        bullets = []
        lines = latex_content.split('\n')
        current_section = "unknown"

        for i, line in enumerate(lines):
            # Track current section
            section_match = re.search(r'\\section\{([^}]+)\}', line, re.IGNORECASE)
            if section_match:
                current_section = section_match.group(1).lower()
                continue

            # Find bullet points in relevant sections
            if re.match(r'\s*\\item\s+', line) and any(keyword in current_section for keyword in ['experience', 'project', 'skill']):
                # Get full bullet content (may span multiple lines)
                bullet_lines = [line]
                j = i + 1
                while j < len(lines) and not re.match(r'\s*\\item\s+|\\end\{|\\section\{', lines[j].strip()):
                    bullet_lines.append(lines[j])
                    j += 1

                bullet_content = '\n'.join(bullet_lines)
                existing_keywords = self.extract_existing_keywords_in_component(
                    bullet_content, target_keywords)

                bullets.append(ResumeSection(
                    name=f"{current_section}_bullet_{len(bullets)}",
                    content=bullet_content,
                    start_line=i,
                    end_line=j-1,
                    original_content=bullet_content,
                    existing_keywords=existing_keywords
                ))

        if not bullets:
            return []

        # Use LLM to detect which bullets are changeable
        bullets_text = ""
        for idx, bullet in enumerate(bullets):
            clean_content = re.sub(r'\\item\s*', '', bullet.content).strip()
            bullets_text += f"[{idx}] {clean_content}\n\n"

        system_prompt = """Analyze these resume bullet points and identify which ones can be meaningfully rewritten to include technical keywords.

CHANGEABLE bullets typically describe:
- Software development work
- Technical projects with implementation details
- System design or architecture
- Programming tasks
- Technical achievements with measurable results

NON-CHANGEABLE bullets typically contain:
- Just job titles, dates, company names
- Education degrees and GPAs
- Awards with no technical content
- Generic responsibilities without technical detail

Return only the indices (numbers) of CHANGEABLE bullets, comma-separated."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Bullet points to analyze:\n{bullets_text}")
        ]

        try:
            response = self.client.invoke(messages)
            changeable_indices = []

            # Parse response for indices
            for part in response.content.split(','):
                try:
                    idx = int(part.strip())
                    if 0 <= idx < len(bullets):
                        changeable_indices.append(idx)
                except ValueError:
                    continue

            changeable_bullets = [bullets[i] for i in changeable_indices]

            logger.info(f"Detected {len(changeable_bullets)}/{len(bullets)} changeable components")
            for i, bullet in enumerate(changeable_bullets):
                logger.info(f"  [{i}] {bullet.name}: existing keywords {bullet.existing_keywords}")

            return changeable_bullets

        except Exception as e:
            logger.error(f"Error detecting changeable components: {e}")
            return bullets  # Fallback to all bullets

    def smart_keyword_matching(self, components: List[ResumeSection], target_keywords: List[str]) -> Dict[str, List[str]]:
        """Match keywords to components based on context and existing keyword tracking"""

        # Step 1: Track existing keyword usage across all components (improved counting)
        keyword_usage = {kw: 0 for kw in target_keywords}
        component_assignments = {comp.name: list(comp.existing_keywords) for comp in components}

        # Count existing keywords properly
        for component in components:
            for existing_kw in component.existing_keywords:
                if existing_kw in keyword_usage:
                    keyword_usage[existing_kw] += 1

        # Step 2: Get keywords that need placement (used < 2 times)
        keywords_to_place = [kw for kw, count in keyword_usage.items() if count < 2]

        if not keywords_to_place:
            logger.info("All keywords already at optimal usage")
            return component_assignments

        # Step 3: Use LLM to assign remaining keywords
        components_text = ""
        for idx, comp in enumerate(components):
            clean_content = re.sub(r'\\item\s*', '', comp.content).strip()
            existing_kw_str = ", ".join(comp.existing_keywords) if comp.existing_keywords else "none"
            
            # Simple context detection for LLM
            context_hints = []
            content_lower = clean_content.lower()
            if any(term in content_lower for term in ['unity', 'game', 'quest', 'vr', 'c#']):
                context_hints.append("Game/Unity")
            if any(term in content_lower for term in ['web', 'react', 'node', 'html']):
                context_hints.append("Web")
            if any(term in content_lower for term in ['ml', 'pytorch', 'ai', 'model']):
                context_hints.append("ML/AI")
            if any(term in content_lower for term in ['vision', 'opencv', 'image']):
                context_hints.append("Vision")
            if any(term in content_lower for term in ['data', 'sql', 'analysis']):
                context_hints.append("Data")
                
            context_str = ", ".join(context_hints) if context_hints else "General"
            components_text += f"[{idx}] Context: {context_str} | Existing: {existing_kw_str}\nContent: {clean_content}\n\n"

        keywords_text = ", ".join(keywords_to_place)

        system_prompt = f"""Match these technical keywords to the most appropriate resume components based on context.

Keywords to place: {keywords_text}

Rules:
1. Each keyword can be used maximum 2 times total across all components
2. Each component should get 1-2 new keywords maximum  
3. Match keywords based on technical relevance and context
4. Consider existing keywords in each component

Context matching guidelines:
- Game/Unity contexts → C#, GPU
- Web contexts → Next.js, HTML, API, SaaS
- ML/AI contexts → PyTorch, GPU, OpenCV
- Vision contexts → OpenCV, GPU, data science
- Data contexts → R, SQL, data science

Return the assignments in this exact format:
[component_index]: keyword1, keyword2
[component_index]: keyword1

Only return assignments for components that should get new keywords."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Components:\n{components_text}")
        ]

        try:
            response = self.client.invoke(messages)

            # Parse LLM response
            for line in response.content.strip().split('\n'):
                if ':' in line:
                    try:
                        comp_idx_str, keywords_str = line.split(':', 1)
                        comp_idx = int(comp_idx_str.strip().replace('[', '').replace(']', ''))

                        if 0 <= comp_idx < len(components):
                            component = components[comp_idx]
                            new_keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]

                            # Validate assignments
                            valid_keywords = []
                            for kw in new_keywords:
                                if kw in keywords_to_place and keyword_usage[kw] < 2:
                                    valid_keywords.append(kw)
                                    keyword_usage[kw] += 1

                            if valid_keywords:
                                component_assignments[component.name].extend(valid_keywords)

                    except (ValueError, IndexError):
                        continue

            # Log final assignments
            logger.info("\n--- KEYWORD ASSIGNMENT STRATEGY ---")
            for comp_name, assigned_kws in component_assignments.items():
                if assigned_kws:
                    logger.info(f"{comp_name}: {assigned_kws}")

            return component_assignments

        except Exception as e:
            logger.error(f"Error in smart keyword matching: {e}")
            return self.fallback_keyword_distribution(components, keywords_to_place)

    def fallback_keyword_distribution(self, components: List[ResumeSection], keywords_to_place: List[str]) -> Dict[str, List[str]]:
        """Fallback keyword distribution if LLM matching fails"""
        component_assignments = {comp.name: list(comp.existing_keywords) for comp in components}

        kw_idx = 0
        for component in components:
            if kw_idx >= len(keywords_to_place):
                break

            # Assign 1-2 keywords per component
            num_to_assign = min(2, len(keywords_to_place) - kw_idx)
            for _ in range(num_to_assign):
                if kw_idx < len(keywords_to_place):
                    component_assignments[component.name].append(keywords_to_place[kw_idx])
                    kw_idx += 1

        return component_assignments

    def optimize_component_strict(self, component: ResumeSection, assigned_keywords: List[str], strict_mode: bool = False) -> str:
        """Optimize single component with assigned keywords, preserving original meaning"""
        if not assigned_keywords:
            return component.original_content

        original_word_count = count_words_strict(component.original_content)
        original_char_count = count_characters_strict(component.original_content)
        
        # Set limits based on strict mode
        if strict_mode:
            max_chars = original_char_count + 8
            max_words = original_word_count + 4
        else:
            max_chars = original_char_count + 20
            max_words = original_word_count + 6
        
        # Remove duplicates while preserving order
        keywords = list(dict.fromkeys(assigned_keywords))
        new_keywords = [kw for kw in keywords if kw not in component.existing_keywords]

        if not new_keywords:
            return component.original_content

        system_prompt = f"""Rewrite this resume bullet to naturally include these keywords: {', '.join(new_keywords)}

CRITICAL RULES:
1. PRESERVE the original meaning and achievements
2. Replace generic terms with specific keywords where appropriate
3. Keep ALL LaTeX commands exactly (\\textbf, \\href, \\item, etc.)
4. Maximum {max_chars} characters (original: {original_char_count})
5. Maximum {max_words} words (original: {original_word_count})

INTEGRATION EXAMPLES:
- "built system" → "built distributed system" 
- "used framework" → "used PyTorch framework"
- "optimized performance" → "optimized GPU performance"
- "developed application" → "developed C# application"

Return ONLY the rewritten bullet point."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Original bullet ({original_word_count} words, {original_char_count} chars):
{component.original_content}

Keywords to integrate: {', '.join(new_keywords)}
Character limit: {max_chars}
Word limit: {max_words}""")
        ]

        # First attempt
        try:
            response = self.client.invoke(messages)
            optimized = response.content.strip()

            # Clean markdown artifacts
            optimized = re.sub(r'```latex\s*', '', optimized)
            optimized = re.sub(r'```\s*', '', optimized)
            optimized = optimized.strip()

            new_word_count = count_words_strict(optimized)
            new_char_count = count_characters_strict(optimized)

            # Check limits
            if strict_mode and new_char_count > max_chars:
                logger.warning(f"Character limit exceeded: {new_char_count} > {max_chars}. Trying fallback...")
                
                # SECOND ATTEMPT
                fallback_prompt = f"""Make this more concise while keeping keywords: {', '.join(new_keywords)}

Original: {component.original_content}
Previous attempt: {optimized}

STRICT character limit: {original_char_count + 2} characters."""
                
                fallback_messages = [
                    SystemMessage(content="You are a concise technical writer."),
                    HumanMessage(content=fallback_prompt)
                ]
                
                try:
                    fallback_response = self.client.invoke(fallback_messages)
                    optimized = fallback_response.content.strip()
                    optimized = re.sub(r'```.*', '', optimized).strip()
                    
                    new_char_count = count_characters_strict(optimized)
                    if strict_mode and new_char_count > original_char_count + 2:
                        logger.error(f"Fallback exceeded strict limit: {new_char_count} > {original_char_count + 2}")
                        return component.original_content
                        
                except Exception as e:
                    logger.error(f"Fallback optimization failed: {e}")
                    return component.original_content

            # Word count check
            if new_word_count > max_words:
                logger.warning(f"Word count exceeded: {new_word_count} > {max_words}")
                if strict_mode:
                    return component.original_content

            # Basic structure validation
            if not validate_structure_basic(component.original_content, optimized):
                logger.warning("Structure validation failed")
                return component.original_content

            return optimized

        except Exception as e:
            logger.error(f"Error optimizing component: {e}")
            return component.original_content

    def optimize_resume(self, latex_content: str, keywords_or_jd: str, is_jd: bool = False, strict: bool = False) -> str:
        """Main optimization method with smart keyword matching"""

        # Extract keywords
        if is_jd:
            keywords = self.extract_keywords_from_jd(keywords_or_jd)
            target_keywords = self.parse_keywords(keywords)
        else:
            target_keywords = self.parse_keywords(keywords_or_jd)

        logger.info(f"Target keywords: {target_keywords}")

        # Detect changeable components
        changeable_components = self.detect_changeable_components(latex_content, target_keywords)

        if not changeable_components:
            logger.warning("No changeable components detected")
            return latex_content

        # Smart keyword matching
        keyword_assignments = self.smart_keyword_matching(changeable_components, target_keywords)

        # Optimize components based on assignments
        optimized_components = {}
        changes_made = 0

        for component in changeable_components:
            assigned_keywords = keyword_assignments.get(component.name, [])
            new_keywords = [kw for kw in assigned_keywords if kw not in component.existing_keywords]

            if new_keywords:
                optimized_content = self.optimize_component_strict(
                    component, assigned_keywords, strict_mode=strict)

                if optimized_content != component.original_content:
                    optimized_components[component.name] = {
                        'content': optimized_content,
                        'start_line': component.start_line,
                        'end_line': component.end_line,
                        'keywords': assigned_keywords
                    }
                    changes_made += 1

                    orig_chars = count_characters_strict(component.original_content)
                    new_chars = count_characters_strict(optimized_content)
                    orig_words = count_words_strict(component.original_content)
                    new_words = count_words_strict(optimized_content)

                    logger.info(f"\n[CHANGE {changes_made}]")
                    logger.info(f"Component: {component.name}")
                    logger.info(f"Assigned keywords: {assigned_keywords}")
                    logger.info(f"Characters: {orig_chars} → {new_chars} (diff: +{new_chars - orig_chars})")
                    logger.info(f"Words: {orig_words} → {new_words}")

        # Reconstruct resume with optimized components
        if optimized_components:
            optimized_latex = self.reconstruct_resume_targeted(latex_content, optimized_components)
        else:
            optimized_latex = latex_content

        # Final validation
        if not validate_full_resume(latex_content, optimized_latex, logger):
            logger.error("Full resume validation failed!")
            return latex_content

        # Report final keyword usage
        final_usage = {}
        for keyword in target_keywords:
            count = count_keyword_occurrences(optimized_latex, keyword)
            final_usage[keyword] = count

        logger.info(f"\n--- KEYWORD USAGE SUMMARY ---")
        for keyword, count in final_usage.items():
            logger.info(f"'{keyword}': {count} times")

        total_used = sum(1 for count in final_usage.values() if count > 0)
        coverage = (total_used / len(target_keywords)) * 100 if target_keywords else 0

        logger.info(f"\n--- OPTIMIZATION SUMMARY ---")
        logger.info(f"Components modified: {changes_made}")
        logger.info(f"Keywords used: {total_used}/{len(target_keywords)} ({coverage:.1f}%)")

        return optimized_latex

    def reconstruct_resume_targeted(self, original_latex: str, optimized_components: Dict) -> str:
        """Reconstruct resume with targeted component changes"""
        lines = original_latex.split('\n')

        # Sort by start_line in reverse order to maintain indices
        components_sorted = sorted(optimized_components.items(),
                                   key=lambda x: x[1]['start_line'], reverse=True)

        for comp_name, comp_data in components_sorted:
            start_line = comp_data['start_line']
            end_line = comp_data['end_line']
            new_content = comp_data['content']

            # Replace the component lines
            new_lines = new_content.split('\n')
            lines[start_line:end_line+1] = new_lines

        return '\n'.join(lines)
