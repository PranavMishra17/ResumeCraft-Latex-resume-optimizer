#!/usr/bin/env python3
"""
ResumeCraft Main Entry Point
Orchestrates reading the LaTeX resume, calling the Optimizer, writing to a single output file,
and compiling the final PDF into timestamped folders.
"""

import argparse
import os
import sys
import shutil
from datetime import datetime
import logging

from optimizer import ResumeOptimizer
from pdf_compiler import PDFCompiler
from latex_utils import extract_name_from_resume

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def read_file(filepath: str) -> str:
    """Read file content safely."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        sys.exit(1)

def write_file(content: str, filepath: str):
    """Write optimized LaTeX to file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Optimized resume saved to: {os.path.abspath(filepath)}")
    except Exception as e:
        logger.error(f"Error saving optimized resume: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="ResumeCraft: Automated LaTeX Resume Optimizer"
    )
    parser.add_argument("resume", help="Path to original LaTeX resume file (.tex)")
    parser.add_argument("input_file", help="Path to keywords or job description file (.txt)")
    parser.add_argument("--jd", action="store_true", help="Flag indicating input file is a Job Description, not just keywords")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode for word/char limits")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF compilation entirely")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.resume):
        logger.error(f"Resume file not found: {args.resume}")
        sys.exit(1)
        
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    logger.info(f"Starting ResumeCraft workflow...")
    logger.info(f"Original Resume: {args.resume}")
    logger.info(f"Keywords/JD Input: {args.input_file}")

    # 1. Read input files
    latex_content = read_file(args.resume)
    input_content = read_file(args.input_file)
    
    # 2. Extract name for potential PDF logic (backward compatibility)
    resume_name = extract_name_from_resume(latex_content)
    logger.info(f"Detected resume name: {resume_name}")

    # 3. Optimize the Resume
    optimizer = ResumeOptimizer()
    optimized_latex = optimizer.optimize_resume(
        latex_content=latex_content,
        keywords_or_jd=input_content,
        is_jd=args.jd,
        strict=args.strict
    )

    # 4. Save to the single designated output file in root
    output_tex = "optimized_resume.tex"
    write_file(optimized_latex, output_tex)

    # 5. Compile PDF if not skipped
    if not args.no_pdf:
        # Create a timestamped directory for the PDFs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_output_dir = os.path.join("pdfs", timestamp)
        
        # To compile cleanly, we should copy the `optimized_resume.tex` into the temporary folder,
        # compile it there, and keep the PDF there!
        os.makedirs(pdf_output_dir, exist_ok=True)
        pdf_temp_tex = os.path.join(pdf_output_dir, "Resume.tex")
        shutil.copy2(output_tex, pdf_temp_tex)
        
        compiler = PDFCompiler()
        # The PDFCompiler expects the latex file path and will write Resume.pdf in `pdf_output_dir`
        success = compiler.compile_pdf(
            latex_file=pdf_temp_tex,
            output_dir=pdf_output_dir,
            compiler="pdflatex", 
            use_fallback=True
        )
        
        if success:
            logger.info(f"Workflow completed! Check {pdf_output_dir}/Resume.pdf")
        else:
            logger.error("Workflow completed with PDF generation errors.")
    else:
        logger.info("PDF generation skipped. Workflow completed!")

if __name__ == "__main__":
    main()
