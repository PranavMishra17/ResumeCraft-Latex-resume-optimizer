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
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class Config:
    """Configuration for Azure OpenAI"""
    AZURE_DEPLOYMENT = "VARELab-GPT4o"  # Replace with your deployment
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_VARE_KEY")
    AZURE_API_VERSION = "2024-08-01-preview"
    AZURE_ENDPOINT =  os.getenv("AZURE_ENDPOINT")
    TEMPERATURE = 0.25

class ResumeOptimizer:
    def __init__(self):
        self.config = Config()
        print("Initializing Azure OpenAI client...", f"\n AZURE_DEPLOYMENT = {self.config.AZURE_DEPLOYMENT}, \n AZURE_API_KEY = {self.config.AZURE_API_KEY} ")
        self.client = AzureChatOpenAI(
            azure_deployment=self.config.AZURE_DEPLOYMENT,
            api_key=self.config.AZURE_API_KEY,
            api_version=self.config.AZURE_API_VERSION,
            azure_endpoint=self.config.AZURE_ENDPOINT,
            temperature=self.config.TEMPERATURE
        )

    def read_file(self, filepath):
        """Read file content"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            sys.exit(1)

    def parse_subfiles(self, latex_content, base_dir):
        """Parse \subfile commands and read component files"""
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
                # Replace \subfile command with actual content for LLM processing
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

    def optimize_resume(self, latex_content, keywords_or_jd, is_jd=False, base_dir="."):
        """Optimize LaTeX resume with keywords"""
        
        # Extract keywords if job description provided
        if is_jd:
            keywords = self.extract_keywords_from_jd(keywords_or_jd)
        else:
            keywords = keywords_or_jd

        # Parse subfiles if present
        combined_content, subfile_contents = self.parse_subfiles(latex_content, base_dir)

        system_prompt = """You are a LaTeX resume optimization expert. Your task is to modify a LaTeX resume to naturally incorporate relevant keywords while maintaining:

1. EXACT LaTeX structure and formatting
2. One-page length constraint
3. Meaningful project descriptions
4. Strategic keyword placement based on relevance:
   - Game design keywords → game development projects
   - ML/AI keywords → machine learning projects  
   - Web dev keywords → web development projects
   - General tech keywords → most relevant sections

RULES:
- Preserve ALL LaTeX commands, packages, and formatting
- Only modify content within sections, never structure
- Replace existing content strategically, don't just append
- Maintain natural language flow
- Ensure technical accuracy
- Keep bullet points concise
- Prioritize high-impact keywords

Return ONLY the modified LaTeX code, no explanations."""

        user_prompt = f"""Original LaTeX Resume:
{combined_content}

Keywords to incorporate:
{keywords}

Optimize this resume by strategically incorporating these keywords into relevant sections while maintaining the exact LaTeX structure and one-page constraint."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        print("Optimizing resume...")
        response = self.client.invoke(messages)
        optimized_content = response.content.strip()
        
        return optimized_content, subfile_contents, latex_content

    def save_optimized_resume(self, optimized_content, subfile_contents, original_latex, output_path, base_dir="."):
        """Save optimized LaTeX to file(s)"""
        try:
            if subfile_contents:
                # Handle modular resume - split content back into components
                print("Saving modular resume...")
                
                # For modular resumes, we use a smarter approach:
                # Extract sections from optimized content and map to subfiles
                optimized_sections = self.extract_sections_from_optimized(optimized_content)
                
                # Save main file (keep original structure with \subfile commands)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(original_latex)
                
                # Save optimized subfiles
                output_dir = os.path.dirname(output_path)
                subfiles_dir = os.path.join(output_dir, "subsections")
                os.makedirs(subfiles_dir, exist_ok=True)
                
                for subfile_path, original_content in subfile_contents.items():
                    # Map content based on filename
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
            sys.exit(1)
    
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
        """Compile LaTeX to PDF using pdflatex"""
        try:
            print("Compiling PDF...")
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', latex_file],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(latex_file) or '.'
            )
            
            if result.returncode == 0:
                pdf_file = latex_file.replace('.tex', '.pdf')
                print(f"PDF compiled successfully: {pdf_file}")
                return True
            else:
                print(f"PDF compilation failed:\n{result.stderr}")
                return False
                
        except FileNotFoundError:
            print("pdflatex not found. Install LaTeX distribution (e.g., TeX Live, MiKTeX)")
            return False
        except Exception as e:
            print(f"Error compiling PDF: {e}")
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
        output_path = os.path.join(output_dir, f"{base_name}_optimized.tex")

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
                    if file.endswith(('.tex', '.aux', '.log', '.out')):
                        os.remove(os.path.join(os.path.dirname(output_path), file))
                print(f"Final output: {pdf_path}")
            except Exception as e:
                print(f"Warning: Could not clean up intermediate files: {e}")

    print("Resume optimization completed!")

if __name__ == "__main__":
    main()


"""
# With keywords file
python resume_optimizer.py ai resume.tex keywords.txt

# With job description (auto-extracts keywords)
python resume_optimizer.py resume.tex job_description.txt --jd

# Keep LaTeX files for inspection
python resume_optimizer.py resume.tex keywords.txt --keep-tex

"""