#!/usr/bin/env python3
"""
LaTeX Resume Optimizer
Automatically customizes LaTeX resumes based on job descriptions or keywords.
"""

import argparse
import os
import sys
import re
import subprocess
from pathlib import Path
import shutil
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime

class Config:
    """Configuration for Azure OpenAI"""
    AZURE_DEPLOYMENT = "VARELab-GPT4o"
    AZURE_API_KEY = ""
    AZURE_API_VERSION = "2024-08-01-preview"
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")  # Fallback URL
    TEMPERATURE = 0.25

class ResumeOptimizer:
    def __init__(self):
        self.config = Config()
        
        # Debug configuration
        print("Initializing Azure OpenAI client...")
        print(f" AZURE_DEPLOYMENT = {self.config.AZURE_DEPLOYMENT}")
        print(f" AZURE_ENDPOINT = {self.config.AZURE_ENDPOINT}")
        print(f" AZURE_API_VERSION = {self.config.AZURE_API_VERSION}")
        
        # Validate configuration
        if not self.config.AZURE_ENDPOINT:
            print("Error: AZURE_ENDPOINT not set. Please set environment variable or update config.")
            sys.exit(1)
            
        if not self.config.AZURE_API_KEY:
            print("Error: AZURE_API_KEY not set.")
            sys.exit(1)
        
        try:
            self.client = AzureChatOpenAI(
                azure_deployment=self.config.AZURE_DEPLOYMENT,
                api_key=self.config.AZURE_API_KEY,
                api_version=self.config.AZURE_API_VERSION,
                azure_endpoint=self.config.AZURE_ENDPOINT,
                temperature=self.config.TEMPERATURE
            )
            print("Azure OpenAI client initialized successfully!")
        except Exception as e:
            print(f"Error initializing Azure OpenAI client: {e}")
            sys.exit(1)

    def read_file(self, filepath):
        """Read file content"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            sys.exit(1)

    def parse_subfiles(self, latex_content, base_dir):
        """Parse \\subfile commands and read component files"""
        subfile_pattern = r'\\subfile\{([^}]+)\}'
        subfiles = re.findall(subfile_pattern, latex_content)
        
        if not subfiles:
            return latex_content, {}
        
        print(f"Found {len(subfiles)} subfiles")
        
        # Read all subfiles
        subfile_contents = {}
        combined_content = latex_content
        
        for subfile_path in subfiles:
            full_path = os.path.join(base_dir, subfile_path)
            if not full_path.endswith('.tex'):
                full_path += '.tex'
            
            if os.path.exists(full_path):
                content = self.read_file(full_path)
                subfile_contents[subfile_path] = content
                # Replace \\subfile command with actual content for LLM processing
                combined_content = combined_content.replace(
                    f'\\subfile{{{subfile_path}}}', 
                    content
                )
                print(f"  - {subfile_path}")
            else:
                print(f"Warning: Subfile not found: {full_path}")
        
        return combined_content, subfile_contents

    def split_optimized_content(self, optimized_content, original_latex, subfile_contents):
        """Split optimized content back into main file and subfiles"""
        if not subfile_contents:
            return optimized_content, {}
        
        # Find section boundaries in optimized content
        sections = {}
        current_pos = 0
        
        # Use section headers to identify boundaries
        section_patterns = [
            r'\\section\{([^}]+)\}',
            r'\\begin\{center\}.*?\\end\{center\}',
            r'\\subfile\{([^}]+)\}'
        ]
        
        # For modular files, we need to intelligently split
        # This is a simplified approach - extract recognizable sections
        for subfile_path, original_content in subfile_contents.items():
            # Find the section in optimized content that corresponds to this subfile
            # Look for distinctive markers or section headers
            if 'education' in subfile_path.lower():
                edu_match = re.search(r'\\section\{education\}.*?(?=\\section|\Z)', optimized_content, re.DOTALL | re.IGNORECASE)
                if edu_match:
                    sections[subfile_path] = edu_match.group(0)
            elif 'experience' in subfile_path.lower() or 'work' in subfile_path.lower():
                exp_match = re.search(r'\\section\{.*?experience.*?\}.*?(?=\\section|\Z)', optimized_content, re.DOTALL | re.IGNORECASE)
                if exp_match:
                    sections[subfile_path] = exp_match.group(0)
            elif 'skill' in subfile_path.lower():
                skill_match = re.search(r'\\section\{.*?skill.*?\}.*?(?=\\section|\Z)', optimized_content, re.DOTALL | re.IGNORECASE)
                if skill_match:
                    sections[subfile_path] = skill_match.group(0)
            elif 'award' in subfile_path.lower():
                award_match = re.search(r'\\section\{.*?award.*?\}.*?(?=\\section|\Z)', optimized_content, re.DOTALL | re.IGNORECASE)
                if award_match:
                    sections[subfile_path] = award_match.group(0)
            else:
                # Fallback: use original content
                sections[subfile_path] = original_content
        
        # Reconstruct main file with \subfile commands
        main_content = original_latex
        for subfile_path in subfile_contents.keys():
            if subfile_path in sections:
                # Keep the \subfile command in main file
                pass
            
        return main_content, sections

    def extract_company_name(self, job_description):
        """Extract company name from job description"""
        system_prompt = """Extract the company name from this job description. Return ONLY the company name in a format suitable for filenames (no spaces, special characters). If no company name is found, return 'Company'."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=job_description)
        ]
        
        response = self.client.invoke(messages)
        company = response.content.strip().replace(' ', '').replace('-', '').replace('.', '')
        return company if company else "Company"
    
    def extract_keywords_from_jd(self, job_description):
        """Extract relevant keywords from job description using LLM"""
        system_prompt = """Extract 10-15 most important technical keywords, skills, and technologies from this job description. 
        Focus on:
        - Programming languages and frameworks
        - Technical skills and tools
        - Industry-specific terms
        - Required qualifications
        
        Return only a comma-separated list of keywords, no explanations."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Job Description:\n{job_description}")
        ]
        
        response = self.client.invoke(messages)
        keywords = response.content.strip()
        print(f"Extracted keywords: {keywords}")
        return keywords

    def parse_keywords(self, keyword_content):
        """Parse keywords from various formats (comma-separated or line-separated)"""
        # Try comma-separated first
        if ',' in keyword_content:
            keywords = [kw.strip() for kw in keyword_content.split(',') if kw.strip()]
        else:
            # Split by lines and filter empty lines
            keywords = [kw.strip() for kw in keyword_content.split('\n') if kw.strip()]
        
        # Join back as comma-separated for LLM
        return ', '.join(keywords)

    def optimize_resume(self, latex_content, keywords_or_jd, is_jd=False, base_dir="."):
        """Optimize LaTeX resume with keywords"""
        
        # Extract keywords if job description provided
        if is_jd:
            keywords = self.extract_keywords_from_jd(keywords_or_jd)
        else:
            keywords = self.parse_keywords(keywords_or_jd)
            print(f"Parsed keywords: {keywords}")

        # Parse subfiles if present
        combined_content, subfile_contents = self.parse_subfiles(latex_content, base_dir)

        messages = [
            SystemMessage(content="""You are a LaTeX resume optimization expert. Your task is to modify a LaTeX resume to incorporate relevant keywords while STRICTLY maintaining or reducing length.

CRITICAL LENGTH CONSTRAINTS:
- Each \\item bullet point must have EQUAL OR FEWER words than the original
- Count words carefully - NEVER exceed original word count per item
- Remove filler words like "various", "multiple", "different" to make room for keywords
- Replace generic terms with specific keywords when possible
- If you can't fit a keyword, DON'T force it - maintain readability

LATEX STRUCTURE RULES:
- NEVER add, remove, or move \\item commands
- NEVER modify \\begin{itemize}, \\end{itemize}, or any list environments
- NEVER change document structure or formatting commands
- ONLY modify the text content after \\item commands
- Preserve ALL LaTeX commands exactly as they appear

KEYWORD PLACEMENT:
- Replace existing terms with relevant keywords
- Focus on job-specific technical terms
- Maintain grammatical correctness
- Don't repeat keywords excessively

WORD COUNT ENFORCEMENT:
- Original: "Developed web application using React" (5 words)
- Good: "Developed Python application using ML" (5 words)
- Bad: "Developed distributed Python application using machine learning" (8 words)

Return ONLY the modified LaTeX code - no explanations, no markdown."""),
            HumanMessage(content=f"""Original LaTeX Resume:
{combined_content}

Keywords to incorporate:
{keywords}

STRICT REQUIREMENTS:
1. Count words in each \\item and ensure the optimized version has EQUAL OR FEWER words
2. NEVER add new \\item commands or modify list structure
3. Replace generic words with keywords, don't add to existing content
4. If LaTeX has "\\item Developed application using tools" - you can change to "\\item Developed Python application using multithreading" but NOT "\\item Developed distributed Python application using multithreading for data processing"

Return ONLY the complete LaTeX code.""")
        ]
        
        print("Optimizing resume...")
        try:
            response = self.client.invoke(messages)
            optimized_content = response.content.strip()
            
            # Debug: Check if we got a response
            if not optimized_content:
                print("Warning: Empty response from LLM")
                return latex_content, subfile_contents, latex_content
            
            print(f"LLM response length: {len(optimized_content)} characters")
            
            # Clean up any markdown artifacts
            optimized_content = self.clean_latex_output(optimized_content)
            
            if not optimized_content:
                print("Error: Cleaned content is empty!")
                return latex_content, subfile_contents, latex_content
            
            # Validate the optimized content
            if not self.validate_optimized_content(combined_content, optimized_content):
                print("Warning: Validation failed! Using original content.")
                return latex_content, subfile_contents, latex_content
            
            return optimized_content, subfile_contents, latex_content
            
        except Exception as e:
            print(f"Error during optimization: {e}")
            return latex_content, subfile_contents, latex_content
    
    def validate_optimized_content(self, original, optimized):
        """Validate that optimized content maintains structure and word count"""
        # Check for LaTeX structure integrity
        original_items = len(re.findall(r'\\item', original))
        optimized_items = len(re.findall(r'\\item', optimized))
        
        if original_items != optimized_items:
            print(f"Warning: \\item count mismatch! Original: {original_items}, Optimized: {optimized_items}")
            return False
            
        # Check for balanced braces
        if original.count('{') != optimized.count('{') or original.count('}') != optimized.count('}'):
            print("Warning: Brace mismatch detected!")
            return False
            
        # Check for balanced environments
        for env in ['itemize', 'enumerate', 'description']:
            orig_begin = len(re.findall(rf'\\begin\{{{env}\}}', original))
            orig_end = len(re.findall(rf'\\end\{{{env}\}}', original))
            opt_begin = len(re.findall(rf'\\begin\{{{env}\}}', optimized))
            opt_end = len(re.findall(rf'\\end\{{{env}\}}', optimized))
            
            if orig_begin != opt_begin or orig_end != opt_end:
                print(f"Warning: {env} environment mismatch!")
                return False
                
        return True

    def clean_latex_output(self, content):
        """Remove markdown artifacts and ensure LaTeX validity"""
        if not content:
            return content
            
        # Remove markdown code blocks
        content = re.sub(r'```latex\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        content = re.sub(r"'''latex\s*", '', content)
        content = re.sub(r"'''\s*", '', content)
        
        # Remove any stray markdown
        content = re.sub(r'^```.*$', '', content, flags=re.MULTILINE)
        
        # IMPORTANT: Return the cleaned content!
        return content

    def copy_all_assets(self, source_dir, output_dir):
        """Copy all non-tex files maintaining folder structure"""
        print("Copying assets...")
        copied_count = 0
        
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.tex'):
                    continue
                
                source_file = os.path.join(root, file)
                rel_path = os.path.relpath(root, source_dir)
                
                if rel_path == '.':
                    dest_file = os.path.join(output_dir, file)
                else:
                    dest_dir = os.path.join(output_dir, rel_path)
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_file = os.path.join(dest_dir, file)
                
                shutil.copy2(source_file, dest_file)
                copied_count += 1
                if file.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
                    print(f"  - {file}")
        
        print(f"Copied {copied_count} asset files")
        return copied_count > 0

    def save_optimized_resume(self, optimized_content, subfile_contents, original_latex, output_path, base_dir="."):
        """Save optimized LaTeX to file(s)"""
        try:
            if not optimized_content:
                print("Error: No optimized content to save!")
                return
                
            if subfile_contents:
                # Handle modular resume - split content back into components
                print("Saving modular resume...")
                
                output_dir = os.path.dirname(output_path)
                
                # Copy ALL assets before optimization
                print("Copying all assets from source directory...")
                asset_success = self.copy_all_assets(base_dir, output_dir)
                
                if not asset_success:
                    print("Warning: No assets found to copy")
                
                # For modular resumes, extract sections more carefully
                optimized_sections = self.extract_sections_smartly(optimized_content, subfile_contents)
                
                # Save main file (keep original structure with \subfile commands)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(original_latex)
                
                # Save optimized subfiles
                subfiles_dir = os.path.join(output_dir, "subsections")
                os.makedirs(subfiles_dir, exist_ok=True)
                
                for subfile_path, original_content in subfile_contents.items():
                    section_name = self.get_section_name_from_file(subfile_path)
                    optimized_section = optimized_sections.get(section_name, original_content)
                    
                    # Save to new location
                    output_subfile = os.path.join(subfiles_dir, os.path.basename(subfile_path))
                    if not output_subfile.endswith('.tex'):
                        output_subfile += '.tex'
                    
                    with open(output_subfile, 'w', encoding='utf-8') as f:
                        f.write(optimized_section)
                    print(f"  - {output_subfile}")
                
            else:
                # Handle single-file resume
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(optimized_content)
                    
            print(f"Optimized resume saved to: {output_path}")
            
        except Exception as e:
            print(f"Error saving optimized resume: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def extract_sections_smartly(self, optimized_content, subfile_contents):
        """Extract sections from optimized content without duplicating section headers"""
        sections = {}
        
        # For each subfile, extract only the content (not section headers)
        for subfile_path, original_content in subfile_contents.items():
            section_name = self.get_section_name_from_file(subfile_path)
            
            # Try to find the section in optimized content
            if section_name == 'education':
                pattern = r'\\section\{education\}(.*?)(?=\\section|\Z)'
            elif section_name == 'academic_experience':
                pattern = r'\\section\{academic.*?experience\}(.*?)(?=\\section|\Z)'
            elif section_name == 'experience' or section_name == 'work_ex':
                pattern = r'\\section\{.*?(work|experience|professional).*?\}(.*?)(?=\\section|\Z)'
            elif section_name == 'skills':
                pattern = r'\\section\{.*?skill.*?\}(.*?)(?=\\section|\Z)'
            elif section_name == 'awards':
                pattern = r'\\section\{.*?award.*?\}(.*?)(?=\\section|\Z)'
            else:
                # Fallback to original content
                sections[section_name] = original_content
                continue
            
            match = re.search(pattern, optimized_content, re.DOTALL | re.IGNORECASE)
            if match:
                # Extract only the content part (group 1), not the section header
                content_only = match.group(1).strip()
                sections[section_name] = content_only
            else:
                # Fallback to original content
                sections[section_name] = original_content
        
        return sections
    
    def extract_sections_from_optimized(self, optimized_content):
        """Extract sections from optimized content"""
        sections = {}
        
        # Define section mappings
        section_mapping = {
            'education': r'\\section\{education\}.*?(?=\\section|\Z)',
            'experience': r'\\section\{.*?experience.*?\}.*?(?=\\section|\Z)',
            'academic_experience': r'\\section\{academic.*?experience.*?\}.*?(?=\\section|\Z)',
            'work_ex': r'\\section\{.*?(work|professional).*?\}.*?(?=\\section|\Z)',
            'skills': r'\\section\{.*?skill.*?\}.*?(?=\\section|\Z)',
            'awards': r'\\section\{.*?award.*?\}.*?(?=\\section|\Z)',
        }
        
        for section_key, pattern in section_mapping.items():
            match = re.search(pattern, optimized_content, re.DOTALL | re.IGNORECASE)
            if match:
                sections[section_key] = match.group(0)
        
        return sections
    
    def get_section_name_from_file(self, filepath):
        """Get section name from filename"""
        filename = os.path.basename(filepath).replace('.tex', '')
        
        # Map common filename patterns to section names
        if 'education' in filename:
            return 'education'
        elif 'academic' in filename and 'experience' in filename:
            return 'academic_experience'
        elif 'work' in filename or 'experience' in filename:
            return 'experience'
        elif 'skill' in filename:
            return 'skills'
        elif 'award' in filename:
            return 'awards'
        else:
            return filename

    def compile_pdf(self, latex_file):
        """Compile LaTeX to PDF using pdflatex with better error handling"""
        try:
            print("Compiling PDF...")
            # Change to the directory containing the tex file
            work_dir = os.path.dirname(latex_file) or '.'
            tex_filename = os.path.basename(latex_file)
            
            # Run pdflatex twice to resolve references
            for run in range(2):
                print(f"  Running pdflatex (pass {run + 1}/2)...")
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', '-file-line-error', tex_filename],
                    capture_output=True,
                    text=True,
                    cwd=work_dir
                )
                
                if result.returncode != 0:
                    print(f"\nPDF compilation failed on pass {run + 1}!")
                    print("\n--- LaTeX Errors ---")
                    
                    # Parse error messages from stdout
                    lines = result.stdout.split('\n')
                    error_lines = []
                    for i, line in enumerate(lines):
                        if line.startswith('!') or 'Error:' in line or 'error' in line.lower():
                            # Print error and context
                            start = max(0, i - 2)
                            end = min(len(lines), i + 3)
                            error_lines.extend(lines[start:end])
                    
                    if error_lines:
                        print('\n'.join(error_lines[:50]))  # Limit output
                    else:
                        # Fallback to showing stderr
                        print(result.stderr[:1000])
                    
                    # Also check the log file
                    log_file = os.path.join(work_dir, tex_filename.replace('.tex', '.log'))
                    if os.path.exists(log_file):
                        print(f"\nCheck {log_file} for detailed error information")
                    
                    return False
            
            pdf_file = latex_file.replace('.tex', '.pdf')
            if os.path.exists(pdf_file):
                print(f"PDF compiled successfully: {pdf_file}")
                return True
            else:
                print("PDF compilation completed but PDF file not found!")
                return False
                
        except FileNotFoundError:
            print("Error: pdflatex not found!")
            print("Please install a LaTeX distribution:")
            print("  - Windows: MiKTeX (https://miktex.org/)")
            print("  - Mac: MacTeX (https://www.tug.org/mactex/)")
            print("  - Linux: TeX Live (sudo apt-get install texlive-full)")
            return False
        except Exception as e:
            print(f"Error compiling PDF: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description="Optimize LaTeX resume with keywords or job description")
    parser.add_argument("resume", help="Path to LaTeX resume file (.tex)")
    parser.add_argument("input_file", help="Path to keywords or job description file (.txt)")
    parser.add_argument("-o", "--output", help="Output LaTeX file path (default: resume_optimized.tex)")
    parser.add_argument("--jd", action="store_true", help="Input file contains job description (extract keywords)")
    parser.add_argument("--pdf", action="store_true", default=True, help="Compile to PDF after optimization (default: True)")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF compilation")
    parser.add_argument("--keep-tex", action="store_true", help="Keep intermediate .tex files")
    
    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.resume):
        print(f"Error: Resume file not found: {args.resume}")
        sys.exit(1)
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # Set output path - create new folder to preserve originals
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.resume))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"optimized_resume_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}.tex")

    # Initialize optimizer
    optimizer = ResumeOptimizer()

    # Read files
    print(f"Reading resume: {args.resume}")
    latex_content = optimizer.read_file(args.resume)
    
    print(f"Reading input: {args.input_file}")
    input_content = optimizer.read_file(args.input_file)

    # Get base directory for subfiles
    base_dir = os.path.dirname(args.resume) or "."

    # Optimize resume
    optimized_latex, subfile_contents, original_latex = optimizer.optimize_resume(
        latex_content, 
        input_content, 
        is_jd=args.jd,
        base_dir=base_dir
    )

    # Save optimized resume
    optimizer.save_optimized_resume(
        optimized_latex, 
        subfile_contents, 
        original_latex, 
        output_path,
        base_dir
    )

    # Compile PDF if requested (default behavior)
    if args.pdf and not args.no_pdf:
        success = optimizer.compile_pdf(output_path)
        if success and not args.keep_tex:
            # Clean up .tex files, keep only PDF
            try:
                pdf_path = output_path.replace('.tex', '.pdf')
                # Remove .tex files but keep PDF
                for file in os.listdir(os.path.dirname(output_path)):
                    if file.endswith(('.aux', '.log', '.out')):
                        os.remove(os.path.join(os.path.dirname(output_path), file))
                print(f"Final output: {pdf_path}")
            except Exception as e:
                print(f"Warning: Could not clean up intermediate files: {e}")

    print("Resume optimization completed!")

if __name__ == "__main__":
    main()