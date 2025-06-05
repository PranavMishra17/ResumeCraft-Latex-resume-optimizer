import os
import sys
import requests
import json
import base64
from pathlib import Path
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class LaTeXConverter:
    def __init__(self):
        self.api_url = "https://latex.ytotech.com/builds/sync"

    def tex_file_to_pdf(self, tex_file_path, output_pdf_path=None, compiler="pdflatex"):
        """
        Convert a .tex file to PDF using LaTeX-on-HTTP API

        Args:
            tex_file_path: Path to .tex file
            output_pdf_path: Output PDF path (optional)
            compiler: LaTeX compiler to use (pdflatex, xelatex, lualatex, etc.)

        Returns:
            Path to generated PDF or None if failed
        """
        tex_path = Path(tex_file_path)

        if not tex_path.exists():
            print(f"❌ File not found: {tex_file_path}")
            return None

        # Read the tex file
        try:
            with open(tex_path, 'r', encoding='utf-8') as f:
                latex_content = f.read()
        except UnicodeDecodeError:
            try:
                with open(tex_path, 'r', encoding='latin-1') as f:
                    latex_content = f.read()
            except Exception as e:
                print(f"❌ Error reading file: {e}")
                return None

        # Prepare payload according to the API documentation
        payload = {
            "compiler": compiler,
            "resources": [
                {
                    "main": True,
                    "content": latex_content
                }
            ]
        }

        headers = {
            'Content-Type': 'application/json'
        }

        try:
            print(f"🔄 Compiling {tex_file_path} with {compiler}...")

            response = requests.post(
                self.api_url,
                data=json.dumps(payload),
                headers=headers,
                timeout=60
            )

            print(f"📡 Response status: {response.status_code}")

            if response.status_code in [200, 201]:
                # Check if response is PDF
                content_type = response.headers.get('content-type', '')

                if 'application/pdf' in content_type:
                    # Direct PDF response
                    if output_pdf_path is None:
                        output_pdf_path = tex_path.with_suffix('.pdf')

                    with open(output_pdf_path, 'wb') as f:
                        f.write(response.content)

                    print(f"✅ PDF generated successfully: {output_pdf_path}")
                    return str(output_pdf_path)
                else:
                    # JSON response - might contain base64 PDF or error
                    try:
                        result = response.json()
                        if 'pdf' in result:
                            # Base64 encoded PDF
                            pdf_data = base64.b64decode(result['pdf'])
                            if output_pdf_path is None:
                                output_pdf_path = tex_path.with_suffix('.pdf')

                            with open(output_pdf_path, 'wb') as f:
                                f.write(pdf_data)

                            logger.info(
                                f"✅ PDF compiled successfully: {os.path.abspath(output_pdf_path)}")
                            logger.info(f"\n📄 RESUME READY!")
                            logger.info(f"PDF Location: {os.path.abspath(output_pdf_path)}")

                            return str(output_pdf_path)
                        else:
                            print("❌ No PDF in response")
                            print("Response:", result)
                            return None
                    except json.JSONDecodeError:
                        print("❌ Invalid JSON response")
                        print("Response text:", response.text[:500])
                        return None
            else:
                print(
                    f"❌ API request failed with status {response.status_code}")
                try:
                    error_info = response.json()
                    print("Error details:", error_info)
                except:
                    print("Response text:", response.text[:500])
                return None

        except requests.exceptions.Timeout:
            print("❌ Request timed out")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None


# Usage Examples
if __name__ == "__main__":
    converter = LaTeXConverter()

    # Convert your specific file
    # tex_file = r"D:/FullStack/ResumeCraft-Latex-resume-optimizer/optimized_resume_20250604_225205/ai_resume_optimized.tex"
    # output_pdf = r"D:/FullStack/ResumeCraft-Latex-resume-optimizer/optimized_resume_20250604_225205/ai_resume_optimized.pdf"

    tex_file_dir = r"./optimized_resume_20250604_225205"
    folder = Path(tex_file_dir)
    compiler = 'xelatex'

    # pdf_path = converter.tex_file_to_pdf(
    #     tex_file, output_pdf, compiler='xelatex')

    for tex_path in folder.glob("*.tex"):
        pdf_path = tex_path.with_suffix(".pdf")
        if pdf_path.exists():
            print(
                f"Skipping '{tex_path.name}' because '{pdf_path.name}' already exists.")
            continue

        print(
            f"Found TEX: {tex_path.name}  → no PDF found, converting now...")
        # Call your existing method; it will print status internally
        output = converter.tex_file_to_pdf(
            tex_file_path=str(tex_path),
            output_pdf_path=str(pdf_path),
            compiler=compiler
        )
        if output:
            print(f"Successfully created: {pdf_path.name}\n")
        else:
            print(f"Failed to create PDF for {tex_path.name}\n")
