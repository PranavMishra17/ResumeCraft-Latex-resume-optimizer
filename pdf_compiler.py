import os
import requests
import json
import base64
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PDFCompiler:
    def __init__(self):
        self.api_url = "https://latex.ytotech.com/builds/sync"

    def compile_pdf(self, latex_file: str, output_dir: str, compiler: str = "pdflatex", use_fallback: bool = True) -> bool:
        """
        Attempt to compile PDF locally. If it fails and use_fallback is True, try the API.
        PDF is saved inside `output_dir` as `Resume.pdf`.
        """
        logger.info("Compiling PDF...")
        work_dir = os.path.dirname(latex_file) or '.'
        tex_filename = os.path.basename(latex_file)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        final_pdf_path = os.path.join(output_dir, "Resume.pdf")
        
        # 1. Try local compilation
        local_success = self._compile_local(tex_filename, work_dir, compiler)
        
        if local_success:
            original_pdf = os.path.join(work_dir, tex_filename.replace('.tex', '.pdf'))
            if os.path.exists(original_pdf):
                import shutil
                shutil.move(original_pdf, final_pdf_path)
                logger.info(f"✅ PDF compiled locally and saved to: {os.path.abspath(final_pdf_path)}")
                return True
            else:
                logger.error("Local compilation succeeded but PDF file not found.")
                
        # 2. Try API fallback if requested
        if use_fallback:
            logger.info("Falling back to API compilation...")
            api_success = self._compile_api(latex_file, final_pdf_path, compiler)
            if api_success:
                logger.info(f"✅ PDF compiled via API and saved to: {os.path.abspath(final_pdf_path)}")
                return True
                
        logger.error("❌ Failed to compile PDF locally and via API.")
        return False
        
    def _compile_local(self, tex_filename: str, work_dir: str, compiler: str) -> bool:
        """Helper to run local LaTeX compiler"""
        try:
            for run in range(2): # Double run for references etc.
                result = subprocess.run(
                    [compiler, '-interaction=nonstopmode', '-file-line-error', tex_filename],
                    capture_output=True,
                    text=True,
                    cwd=work_dir
                )

                if result.returncode != 0:
                    logger.error(f"Local {compiler} compilation failed on run {run+1}!")
                    # Show first error found
                    for line in result.stdout.split('\n'):
                        if 'Error:' in line or line.startswith('!'):
                            logger.error(line)
                            break
                    return False
            return True
        except FileNotFoundError:
            logger.warning(f"{compiler} not found! Install LaTeX distribution or ensure it is in PATH.")
            return False
        except Exception as e:
            logger.error(f"Error compiling PDF locally: {e}")
            return False

    def _compile_api(self, tex_file_path: str, output_pdf_path: str, compiler: str = "pdflatex") -> bool:
        """Helper to run API fallback using LaTeX-on-HTTP"""
        tex_path = Path(tex_file_path)

        if not tex_path.exists():
            logger.error(f"❌ File not found: {tex_file_path}")
            return False

        # Read the tex file
        try:
            with open(tex_path, 'r', encoding='utf-8') as f:
                latex_content = f.read()
        except UnicodeDecodeError:
            try:
                with open(tex_path, 'r', encoding='latin-1') as f:
                    latex_content = f.read()
            except Exception as e:
                logger.error(f"❌ Error reading file: {e}")
                return False

        payload = {
            "compiler": compiler,
            "resources": [
                {
                    "main": True,
                    "content": latex_content
                }
            ]
        }
        headers = {'Content-Type': 'application/json'}

        try:
            response = requests.post(self.api_url, data=json.dumps(payload), headers=headers, timeout=60)

            if response.status_code in [200, 201]:
                content_type = response.headers.get('content-type', '')

                if 'application/pdf' in content_type:
                    # Direct PDF response
                    with open(output_pdf_path, 'wb') as f:
                        f.write(response.content)
                    return True
                else:
                    # JSON response - might contain base64 PDF or error
                    try:
                        result = response.json()
                        if 'pdf' in result:
                            pdf_data = base64.b64decode(result['pdf'])
                            with open(output_pdf_path, 'wb') as f:
                                f.write(pdf_data)
                            return True
                        else:
                            logger.error("❌ No PDF in response API")
                            return False
                    except json.JSONDecodeError:
                        logger.error("❌ Invalid JSON response from API")
                        return False
            else:
                logger.error(f"❌ API request failed with status {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ API Request error: {e}")
            return False
