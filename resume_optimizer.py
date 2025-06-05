#!/usr/bin/env python3
"""
Simplified LaTeX Resume Optimizer
Optimizes resume by section blocks, not individual bullets.
"""

import argparse
import os
import sys
import re
import subprocess
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
import logging
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import config

# For tex to pdf conversion
import requests
import json
import base64
from pathlib import Path
from tex_to_pdf import LaTeXConverter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
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


class SimplifiedResumeOptimizer:
    def __init__(self):
        self.config = config.Config()

        logger.info("Initializing Azure OpenAI client...")

        if not self.config.AZURE_ENDPOINT or not self.config.AZURE_API_KEY:
            print("Error: AZURE_ENDPOINT and AZURE_API_KEY must be set.")
            sys.exit(1)

        try:
            self.client = AzureChatOpenAI(
                azure_deployment=self.config.AZURE_DEPLOYMENT,
                api_key=self.config.AZURE_API_KEY,
                api_version=self.config.AZURE_API_VERSION,
                azure_endpoint=self.config.AZURE_ENDPOINT,
                temperature=self.config.TEMPERATURE
            )
            logger.info("Azure OpenAI client initialized successfully!")
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI client: {e}")
            sys.exit(1)

    def read_file(self, filepath):
        """Read file content"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            sys.exit(1)

    def extract_name_from_resume(self, latex_content: str) -> str:
        """Extract name from LaTeX resume for PDF naming"""
        # Look for common name patterns
        patterns = [
            r'\\name\{([^}]+)\}',
            r'\\textbf\{\\huge\s+([^}]+)\}',
            r'\\begin\{center\}\\textbf\{\\Large\s+([^}]+)\}',
            r'\\textbf\{\\Large\s+([^}]+)\}',
            r'\\huge\{([^}]+)\}',
            r'\\Large\{([^}]+)\}'
        ]

        for pattern in patterns:
            match = re.search(pattern, latex_content, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean LaTeX commands from name
                name = re.sub(r'\\[a-zA-Z]+\s*', '', name).strip()
                if name and len(name.split()) <= 4:  # Reasonable name length
                    return name

        # Fallback: look for first capitalized words in document
        lines = latex_content.split('\n')
        for line in lines:
            if '\\begin{document}' in line:
                continue
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', line)
            if words and len(words[0].split()) <= 4:
                return words[0]

        return "Resume"

    def escape_latex_keywords(self, keywords: List[str]) -> List[str]:
        """Escape special LaTeX characters in keywords"""
        escaped = []
        for keyword in keywords:
            # Escape ampersand and other special characters
            escaped_keyword = keyword.replace('&', '\\&')
            escaped_keyword = escaped_keyword.replace('%', '\\%')
            escaped_keyword = escaped_keyword.replace('$', '\\$')
            escaped_keyword = escaped_keyword.replace('#', '\\#')
            escaped.append(escaped_keyword)
        return escaped

    def extract_keywords_from_jd(self, job_description):
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

    def parse_keywords(self, keyword_content):
        """Parse keywords from various formats"""
        if ',' in keyword_content:
            keywords = [kw.strip()
                        for kw in keyword_content.split(',') if kw.strip()]
        else:
            keywords = [kw.strip()
                        for kw in keyword_content.split('\n') if kw.strip()]

        # Escape LaTeX special characters
        return self.escape_latex_keywords(keywords)

    def extract_existing_keywords_in_component(self, component_content: str, target_keywords: List[str]) -> List[str]:
        """Extract keywords that already exist in a specific component"""
        existing = []
        content_lower = component_content.lower()

        for keyword in target_keywords:
            # Remove LaTeX escaping for comparison
            keyword_clean = keyword.replace('\\&', '&').replace(
                '\\%', '%').replace('\\$', '$').replace('\\#', '#')
            if keyword_clean.lower() in content_lower:
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
            section_match = re.search(
                r'\\section\{([^}]+)\}', line, re.IGNORECASE)
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

            logger.info(
                f"Detected {len(changeable_bullets)}/{len(bullets)} changeable components")
            for i, bullet in enumerate(changeable_bullets):
                logger.info(
                    f"  [{i}] {bullet.name}: existing keywords {bullet.existing_keywords}")

            return changeable_bullets

        except Exception as e:
            logger.error(f"Error detecting changeable components: {e}")
            return bullets  # Fallback to all bullets

    def smart_keyword_matching(self, components: List[ResumeSection], target_keywords: List[str]) -> Dict[str, List[str]]:
        """Intelligently match keywords to components based on context and existing keywords"""

        # Track keyword usage across all components
        keyword_usage = {kw: 0 for kw in target_keywords}
        component_assignments = {comp.name: [] for comp in components}

        # First, account for existing keywords
        for component in components:
            for existing_kw in component.existing_keywords:
                if existing_kw in keyword_usage:
                    keyword_usage[existing_kw] += 1
                    component_assignments[component.name].append(existing_kw)

        # Get keywords that need placement (used < 2 times)
        keywords_to_place = [kw for kw,
                             count in keyword_usage.items() if count < 2]

        if not keywords_to_place:
            logger.info("All keywords already at optimal usage")
            return component_assignments

        # Use LLM to match remaining keywords to components
        components_text = ""
        for idx, comp in enumerate(components):
            clean_content = re.sub(r'\\item\s*', '', comp.content).strip()
            existing_kw_str = ", ".join(
                comp.existing_keywords) if comp.existing_keywords else "none"
            components_text += f"[{idx}] Existing keywords: {existing_kw_str}\nContent: {clean_content}\n\n"

        keywords_text = ", ".join(keywords_to_place)

        system_prompt = f"""Match these technical keywords to the most appropriate resume components based on context and relevance.

Keywords to place: {keywords_text}

Rules:
1. Each keyword can be used maximum 2 times total across all components
2. Each component should get 1-2 new keywords maximum
3. Match keywords based on technical relevance and context
4. Consider existing keywords in each component

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
                        comp_idx = int(comp_idx_str.strip().replace(
                            '[', '').replace(']', ''))

                        if 0 <= comp_idx < len(components):
                            component = components[comp_idx]
                            new_keywords = [
                                kw.strip() for kw in keywords_str.split(',') if kw.strip()]

                            # Validate assignments
                            valid_keywords = []
                            for kw in new_keywords:
                                if kw in keywords_to_place and keyword_usage[kw] < 2:
                                    valid_keywords.append(kw)
                                    keyword_usage[kw] += 1

                            if valid_keywords:
                                component_assignments[component.name].extend(
                                    valid_keywords)

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
            # Fallback: simple distribution
            return self.fallback_keyword_distribution(components, keywords_to_place)

    def fallback_keyword_distribution(self, components: List[ResumeSection], keywords_to_place: List[str]) -> Dict[str, List[str]]:
        """Fallback keyword distribution if LLM matching fails"""
        component_assignments = {comp.name: list(
            comp.existing_keywords) for comp in components}

        kw_idx = 0
        for component in components:
            if kw_idx >= len(keywords_to_place):
                break

            # Assign 1-2 keywords per component
            num_to_assign = min(2, len(keywords_to_place) - kw_idx)
            for _ in range(num_to_assign):
                if kw_idx < len(keywords_to_place):
                    component_assignments[component.name].append(
                        keywords_to_place[kw_idx])
                    kw_idx += 1

        return component_assignments

    def count_words_strict(self, text: str) -> int:
        """Count words strictly excluding LaTeX commands"""
        # Remove all LaTeX commands and their arguments
        clean_text = re.sub(r'\\[a-zA-Z*]+(\[[^\]]*\])?(\{[^}]*\})*', '', text)
        # Remove URLs
        clean_text = re.sub(r'https?://[^\s}]+', '', clean_text)
        # Remove special characters and braces
        clean_text = re.sub(r'[{}\\&%]', '', clean_text)
        # Count actual words
        words = [w for w in clean_text.split() if w.strip()
                 and not w.isdigit()]
        return len(words)

    def optimize_component_strict(self, component: ResumeSection, assigned_keywords: List[str]) -> str:
        """Optimize component with upper word limit only"""
        if not assigned_keywords:
            return component.original_content

        original_word_count = self.count_words_strict(
            component.original_content)
        max_words = original_word_count + 4  # Only upper limit

        # Remove duplicates while preserving order
        keywords = list(dict.fromkeys(assigned_keywords))

        system_prompt = f"""Rewrite this resume bullet to include these keywords: {', '.join(keywords)}

STRICT RULES:
1. Maximum word count: {max_words} words (original: {original_word_count})
2. Keep ALL LaTeX commands exactly (\\textbf, \\href, \\item, etc.)
3. Preserve existing keywords that are already present
4. Replace generic terms with new keywords where appropriate
5. Maintain technical accuracy

EXAMPLES:
- "developed system" → "developed distributed system"
- "implemented framework" → "implemented robotics framework" 
- "optimized performance" → "optimized CPU performance"

Word count limit is CRITICAL. Do not exceed {max_words} words."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Original bullet ({original_word_count} words):
{component.original_content}

Keywords to integrate: {', '.join(keywords)}

Maximum word count: {max_words} words
Return ONLY the rewritten bullet.""")
        ]

        try:
            response = self.client.invoke(messages)
            optimized = response.content.strip()

            # Clean markdown
            optimized = re.sub(r'```latex\s*', '', optimized)
            optimized = re.sub(r'```\s*', '', optimized)

            # Word count validation (upper limit only)
            new_word_count = self.count_words_strict(optimized)

            if new_word_count > max_words:
                logger.warning(
                    f"Word count exceeded: {new_word_count} > {max_words}")
                return component.original_content

            # Basic structure validation
            if not self.validate_structure_basic(component.original_content, optimized):
                logger.warning("Structure validation failed")
                return component.original_content

            return optimized

        except Exception as e:
            logger.error(f"Error optimizing component: {e}")
            return component.original_content

    def validate_structure_basic(self, original: str, optimized: str) -> bool:
        """Basic LaTeX structure validation"""
        # Check \\item presence
        if '\\item' in original and '\\item' not in optimized:
            return False

        # Check major brace balance (allow 1 difference)
        orig_balance = original.count('{') - original.count('}')
        opt_balance = optimized.count('{') - optimized.count('}')

        if abs(orig_balance - opt_balance) > 1:
            return False

        return True

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
        changeable_components = self.detect_changeable_components(
            latex_content, target_keywords)

        if not changeable_components:
            logger.warning("No changeable components detected")
            return latex_content

        # Smart keyword matching
        keyword_assignments = self.smart_keyword_matching(
            changeable_components, target_keywords)

        # Optimize components based on assignments
        optimized_components = {}
        changes_made = 0

        for component in changeable_components:
            assigned_keywords = keyword_assignments.get(component.name, [])
            new_keywords = [
                kw for kw in assigned_keywords if kw not in component.existing_keywords]

            if new_keywords:
                optimized_content = self.optimize_component_strict(
                    component, assigned_keywords)

                if optimized_content != component.original_content:
                    optimized_components[component.name] = {
                        'content': optimized_content,
                        'start_line': component.start_line,
                        'end_line': component.end_line,
                        'keywords': assigned_keywords
                    }
                    changes_made += 1

                    logger.info(f"\n[CHANGE {changes_made}]")
                    logger.info(f"Component: {component.name}")
                    logger.info(f"Assigned keywords: {assigned_keywords}")
                    logger.info(
                        f"Words: {self.count_words_strict(component.original_content)} → {self.count_words_strict(optimized_content)}")

        # Reconstruct resume with optimized components
        if optimized_components:
            optimized_latex = self.reconstruct_resume_targeted(
                latex_content, optimized_components)
        else:
            optimized_latex = latex_content

        # Final validation
        if not self.validate_full_resume(latex_content, optimized_latex):
            logger.error("Full resume validation failed!")
            return latex_content

        # Report final keyword usage
        final_usage = {}
        for keyword in target_keywords:
            # Count clean keyword (without LaTeX escaping)
            clean_keyword = keyword.replace('\\&', '&').replace(
                '\\%', '%').replace('\\$', '$').replace('\\#', '#')
            count = optimized_latex.lower().count(clean_keyword.lower())
            final_usage[keyword] = count

        logger.info(f"\n--- KEYWORD USAGE SUMMARY ---")
        for keyword, count in final_usage.items():
            logger.info(f"'{keyword}': {count} times")

        total_used = sum(1 for count in final_usage.values() if count > 0)
        coverage = (total_used / len(target_keywords)) * 100

        logger.info(f"\n--- OPTIMIZATION SUMMARY ---")
        logger.info(f"Components modified: {changes_made}")
        logger.info(
            f"Keywords used: {total_used}/{len(target_keywords)} ({coverage:.1f}%)")

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

    def validate_full_resume(self, original: str, optimized: str) -> bool:
        """Validate full resume structure"""
        # Check item count
        orig_items = len(re.findall(r'\\item\s', original))
        opt_items = len(re.findall(r'\\item\s', optimized))

        if orig_items != opt_items:
            logger.error(f"Item count mismatch: {orig_items} vs {opt_items}")
            return False

        # Check document structure
        for cmd in ['\\documentclass', '\\begin{document}', '\\end{document}']:
            if original.count(cmd) != optimized.count(cmd):
                logger.error(f"Document structure corrupted: {cmd}")
                return False

        logger.info("Full resume validation passed")
        return True

    def save_optimized_resume(self, optimized_content, output_path):
        """Save optimized LaTeX to file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(optimized_content)
            logger.info(f"Optimized resume saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving optimized resume: {e}")
            sys.exit(1)

    def compile_pdf(self, latex_file, resume_name):
        """Compile LaTeX to PDF with industry standard naming"""
        try:
            logger.info("Compiling PDF...")
            work_dir = os.path.dirname(latex_file) or '.'
            tex_filename = os.path.basename(latex_file)

            for run in range(2):
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode',
                        '-file-line-error', tex_filename],
                    capture_output=True,
                    text=True,
                    cwd=work_dir
                )

                if result.returncode != 0:
                    logger.error(f"PDF compilation failed!")
                    # Show first error found
                    for line in result.stdout.split('\n'):
                        if 'Error:' in line or line.startswith('!'):
                            logger.error(line)
                            break
                    return False

            # Rename PDF to industry standard format
            original_pdf = latex_file.replace('.tex', '.pdf')
            standard_pdf = os.path.join(work_dir, f"Resume.pdf")

            if os.path.exists(original_pdf):
                if original_pdf != standard_pdf:
                    os.rename(original_pdf, standard_pdf)

                logger.info(
                    f"✅ PDF compiled successfully: {os.path.abspath(standard_pdf)}")
                logger.info(f"\n📄 RESUME READY!")
                logger.info(f"PDF Location: {os.path.abspath(standard_pdf)}")
                logger.info(f"\n🔧 To make further edits:")
                logger.info(
                    f"   1. Edit the optimized file: {os.path.abspath(latex_file)}")
                logger.info(f"   2. Recompile with: pdflatex {tex_filename}")
                logger.info(
                    f"   3. Run from directory: {os.path.abspath(work_dir)}")

                return True
            else:
                logger.error("PDF file not found after compilation")
                return False

        except FileNotFoundError:
            logger.error("pdflatex not found! Install LaTeX distribution.")
            return False
        except Exception as e:
            logger.error(f"Error compiling PDF: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Simplified LaTeX resume optimizer")
    parser.add_argument("resume", help="Path to LaTeX resume file (.tex)")
    parser.add_argument(
        "input_file", help="Path to keywords or job description file (.txt)")
    parser.add_argument("-o", "--output", help="Output LaTeX file path")
    parser.add_argument("--jd", action="store_true",
                        help="Input file contains job description")
    parser.add_argument("--pdf", action="store_true",
                        default=True, help="Compile to PDF")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip PDF compilation")
    parser.add_argument("--strict", action="store_true",
                        help="Enable strict mode with retry logic for character limits")

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.resume):
        print(f"Error: Resume file not found: {args.resume}")
        sys.exit(1)

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # Initialize optimizer
    optimizer = SimplifiedResumeOptimizer()

    # Read files
    logger.info(f"Reading resume: {args.resume}")
    latex_content = optimizer.read_file(args.resume)

    # Extract name for PDF naming
    resume_name = optimizer.extract_name_from_resume(latex_content)
    logger.info(f"Detected resume name: {resume_name}")

    # Set output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.resume))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"optimized_resume_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_optimized.tex")

    logger.info(f"Reading input: {args.input_file}")
    input_content = optimizer.read_file(args.input_file)

    # Optimize resume
    optimized_latex = optimizer.optimize_resume(
        latex_content,
        input_content,
        is_jd=args.jd,
        strict=args.strict
    )

    # Save optimized resume
    optimizer.save_optimized_resume(optimized_latex, output_path)

    # Compile PDF if requested
    if args.pdf and not args.no_pdf:
        converter = LaTeXConverter()
        output = converter.tex_file_to_pdf(
            tex_file_path=output_path,
            output_pdf_path=os.path.join(os.path.dirname(
                output_path), f"Resume.pdf"),
            compiler='xelatex'
        )
        if output:
            print(f"Successfully created: {output}\n")
        else:
            optimizer.compile_pdf(output_path, resume_name)

    logger.info("Resume optimization completed!")


if __name__ == "__main__":
    main()


    """
    python resume_optimizer.py ai_resume.tex keywords.txt --strict

    python resume_optimizer.py ai_resume.tex keywords.txt --strict

    """

