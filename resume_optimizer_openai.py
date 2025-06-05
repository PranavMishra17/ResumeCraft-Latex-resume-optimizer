#!/usr/bin/env python3
"""
Simplified LaTeX Resume Optimizer
Optimizes resume by section blocks, not individual bullets.
Modified to use OpenAI API instead of Azure OpenAI.
"""

import argparse
import os
import sys
import re
import subprocess
from pathlib import Path
from langchain_openai import ChatOpenAI  # Changed from AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
import logging
from dataclasses import dataclass
from typing import List, Dict, Set
import config

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


class SimplifiedResumeOptimizer:
    def __init__(self):
        self.config = config.Config()

        logger.info("Initializing OpenAI client...")

        # Check for OpenAI API key instead of Azure credentials
        if not self.config.OPENAI_API_KEY:
            print("Error: OPENAI_API_KEY must be set.")
            sys.exit(1)

        try:
            # Initialize regular OpenAI client
            self.client = ChatOpenAI(
                api_key=self.config.OPENAI_API_KEY,
                model=getattr(self.config, 'OPENAI_MODEL',
                              'gpt-4'),  # Default to gpt-4
                temperature=self.config.TEMPERATURE
            )
            logger.info("OpenAI client initialized successfully!")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            sys.exit(1)

    def read_file(self, filepath):
        """Read file content"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            sys.exit(1)

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
        return keywords

    def extract_existing_keywords(self, latex_content: str, target_keywords: List[str]) -> Set[str]:
        """Extract keywords that already exist in the resume"""
        existing = set()
        content_lower = latex_content.lower()

        for keyword in target_keywords:
            if keyword.lower() in content_lower:
                existing.add(keyword)

        logger.info(
            f"Found {len(existing)} existing keywords: {list(existing)}")
        return existing

    def detect_changeable_components(self, latex_content: str) -> List[ResumeSection]:
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
                bullets.append(ResumeSection(
                    name=f"{current_section}_bullet_{len(bullets)}",
                    content=bullet_content,
                    start_line=i,
                    end_line=j-1,
                    original_content=bullet_content
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
                logger.info(f"  [{i}] {bullet.name}: {bullet.content[:60]}...")

            return changeable_bullets

        except Exception as e:
            logger.error(f"Error detecting changeable components: {e}")
            return bullets  # Fallback to all bullets

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

    def optimize_component_strict(self, component: ResumeSection, keywords_to_add: List[str]) -> str:
        """Optimize component with strict ±4 word limit"""
        if not keywords_to_add:
            return component.original_content

        original_word_count = self.count_words_strict(
            component.original_content)
        min_words = original_word_count - 4
        max_words = original_word_count + 4

        # Limit to 2 keywords max
        keywords = keywords_to_add[:2]

        system_prompt = f"""Rewrite this resume bullet to include these keywords: {', '.join(keywords)}

STRICT RULES:
1. Word count: {min_words}-{max_words} words (original: {original_word_count})
2. Keep ALL LaTeX commands exactly (\\textbf, \\href, \\item, etc.)
3. Replace generic terms with keywords
4. Maintain technical accuracy

EXAMPLES:
- "developed system" → "developed distributed system"
- "implemented framework" → "implemented robotics framework" 
- "optimized performance" → "optimized CPU performance"

Word count is CRITICAL. Count carefully."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Original bullet ({original_word_count} words):
{component.original_content}

Keywords to integrate: {', '.join(keywords)}

Target word count: {min_words}-{max_words} words
Return ONLY the rewritten bullet.""")
        ]

        try:
            response = self.client.invoke(messages)
            optimized = response.content.strip()

            # Clean markdown
            optimized = re.sub(r'```latex\s*', '', optimized)
            optimized = re.sub(r'```\s*', '', optimized)

            # Strict word count validation
            new_word_count = self.count_words_strict(optimized)

            if new_word_count < min_words or new_word_count > max_words:
                logger.warning(
                    f"Word count violation: {new_word_count} not in range [{min_words}, {max_words}]")
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
        """Main optimization method - focused on changeable components only"""

        # Extract keywords
        if is_jd:
            keywords = self.extract_keywords_from_jd(keywords_or_jd)
            target_keywords = self.parse_keywords(keywords)
        else:
            target_keywords = self.parse_keywords(keywords_or_jd)

        logger.info(f"Target keywords: {target_keywords}")

        # Find existing keywords and track their usage
        existing_keywords = self.extract_existing_keywords(
            latex_content, target_keywords)
        keyword_usage_count = {}

        # Count existing usage
        for keyword in target_keywords:
            count = latex_content.lower().count(keyword.lower())
            keyword_usage_count[keyword] = count
            if count > 0:
                logger.info(f"Keyword '{keyword}' already used {count} times")

        # Only add keywords that are used less than 2 times
        keywords_to_add = [
            kw for kw in target_keywords if keyword_usage_count.get(kw, 0) < 2]

        logger.info(f"Keywords to add: {keywords_to_add}")

        if not keywords_to_add:
            logger.info("All keywords already at max usage (2 times)")
            return latex_content

        # Detect changeable components using LLM
        changeable_components = self.detect_changeable_components(
            latex_content)

        if not changeable_components:
            logger.warning("No changeable components detected")
            return latex_content

        # Distribute keywords while enforcing max 2 usage per keyword
        optimized_components = {}
        changes_made = 0

        for i, component in enumerate(changeable_components):
            # Find keywords that still need placement (usage < 2)
            available_keywords = [
                kw for kw in keywords_to_add if keyword_usage_count.get(kw, 0) < 2]

            if not available_keywords:
                break  # All keywords used enough times

            # Assign 1-2 keywords, prioritizing least used
            component_keywords = available_keywords[:2]

            optimized_content = self.optimize_component_strict(
                component, component_keywords)

            if optimized_content != component.original_content:
                # Update keyword usage count based on what was actually added
                for keyword in component_keywords:
                    if keyword.lower() in optimized_content.lower() and keyword.lower() not in component.original_content.lower():
                        keyword_usage_count[keyword] = keyword_usage_count.get(
                            keyword, 0) + 1
                        logger.info(
                            f"Keyword '{keyword}' now used {keyword_usage_count[keyword]} times")

                optimized_components[component.name] = {
                    'content': optimized_content,
                    'start_line': component.start_line,
                    'end_line': component.end_line,
                    'keywords': component_keywords
                }
                changes_made += 1

                logger.info(f"\n[CHANGE {changes_made}]")
                logger.info(f"Component: {component.name}")
                logger.info(f"Keywords: {component_keywords}")
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
            count = optimized_latex.lower().count(keyword.lower())
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

    def compile_pdf(self, latex_file):
        """Compile LaTeX to PDF"""
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

            pdf_file = latex_file.replace('.tex', '.pdf')
            if os.path.exists(pdf_file):
                logger.info(f"PDF compiled successfully: {pdf_file}")
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

    # Set output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.resume))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"optimized_resume_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_optimized.tex")

    # Initialize optimizer
    optimizer = SimplifiedResumeOptimizer()

    # Read files
    logger.info(f"Reading resume: {args.resume}")
    latex_content = optimizer.read_file(args.resume)

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
        optimizer.compile_pdf(output_path)

    logger.info("Resume optimization completed!")


if __name__ == "__main__":
    main()
