# ResumeCraft 🎯

*AI-powered LaTeX resume optimization with intelligent keyword integration*

## Overview

ResumeCraft uses LLM-powered analysis to intelligently optimize LaTeX resumes for ATS systems. Unlike simple keyword stuffing, it identifies changeable resume components, enforces strict word limits, and tracks keyword usage to maintain natural content flow.

![Screenshot](https://github.com/PranavMishra17/ResumeCraft-Latex-resume-optimizer/blob/c5eb36a9bb31ac2a28450fd9faf8770bad4118ed/resume.png)

> **For Recruiters:** This tool demonstrates semantic analysis, LaTeX AST manipulation, constraint solving, and real-time validation—probably more engineering skill than any keyword could convey. 😉

## Technical Architecture

### Core Innovation: Component-Level Intelligence
- **Component Detection**: Identifies modifiable bullets (skips headers, titles).
- **Semantic Keyword Mapping**: Inserts keywords only where contextually relevant.
- **Frequency Control**: Max 2 uses per keyword, no spamming.
- **Surgical Reconstruction**: Targeted line replacement preserves document structure

### Optimization Flow
1. **Keyword Extraction** – LLM extracts key terms from job description.
2. **Resume Analysis** – Detects keyword presence, filters static content.
3. **Component Detection** – Identifies eligible lines for rewriting.
4. **Keyword Assignment** – Distributes keywords to best-fit sections.
5. **Constraint Rewriting** – Rewrites with equal/fewer words, excludes LaTeX in word count.
6. **Usage Tracking** – Limits keyword frequency during generation.
7. **Validation** – Verifies LaTeX syntax, rebuilds final resume as pdf.

### Key Algorithms
- **Semantic Component Analysis**: LLM distinguishes between descriptive content vs. static information
- **Keyword Frequency Control**: Real-time tracking prevents overuse
- **Word Count Validation**: Excludes LaTeX commands from count for accuracy
- **Structure Preservation**: Maintains all `\item`, `\textbf`, `\href` commands exactly

## Installation & Configuration

### 1. Python Dependencies
```bash
git clone https://github.com/PranavMishra17/ResumeCraft-Latex-resume-optimizer
pip install -r requirements.txt
```

### 2. API Configuration

#### Azure OpenAI (Default)
```bash
export AZURE_API_KEY="your-api-key"
export AZURE_ENDPOINT="https://your-resource.openai.azure.com/"
```

#### OpenAI API Alternative
To use OpenAI instead of Azure, modify `resume_optimizer.py`:

```python
# Replace the Config class with:
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL = "gpt-4o"
    TEMPERATURE = 0.2

# Replace client initialization with:
from langchain_openai import ChatOpenAI

self.client = ChatOpenAI(
    api_key=self.config.OPENAI_API_KEY,
    model=self.config.MODEL,
    temperature=self.config.TEMPERATURE
)
```

#### Other LLM Providers
```python
# Anthropic Claude
from langchain_anthropic import ChatAnthropic
self.client = ChatAnthropic(api_key="your-key", model="claude-3-sonnet-20240229")

# Local Models (Ollama)
from langchain_community.chat_models import ChatOllama
self.client = ChatOllama(model="llama2", base_url="http://localhost:11434")
```

### 3. LaTeX Distribution (Required for PDF compilation)

LaTeX is required to compile `.tex` files to PDF. Choose based on your platform and storage availability:

#### Windows - MiKTeX
**Recommended for Windows users**

```bash
# Download installer from https://miktex.org/download
# Two installation options:
```

**Basic Installation** (~200MB):
- Core LaTeX packages only
- Downloads missing packages automatically during compilation
- Suitable for most resume templates

**Complete Installation** (~2.5GB):
- All LaTeX packages included
- No internet required during compilation
- Recommended for complex documents

**Installation Steps:**
1. Download MiKTeX installer
2. Run as administrator
3. Choose "Install for all users" (recommended)
4. Enable "Always install missing packages on-the-fly"
5. Verify: Open Command Prompt → `pdflatex --version`

#### macOS - MacTeX
**Full-featured LaTeX for Mac**

```bash
# Download from https://www.tug.org/mactex/
```

**MacTeX Full** (~4GB):
- Complete TeX Live distribution
- GUI applications (TeXShop, LaTeXiT)
- All packages included
- Most reliable option

**BasicTeX** (~80MB):
- Minimal command-line only
- Requires manual package installation
- For experienced users only

**Installation:**
1. Download `.pkg` file (allow 30-60 minutes)
2. Double-click to install 
3. Add to PATH: Add `/usr/local/texlive/2023/bin/universal-darwin` to `~/.zshrc`
4. Verify: `which pdflatex`

#### Linux - TeX Live
**Native LaTeX for Linux distributions**

**Ubuntu/Debian:**
```bash
# Full installation (~3GB)
sudo apt-get update
sudo apt-get install texlive-full

# Minimal installation (~500MB) - may need additional packages
sudo apt-get install texlive-latex-base texlive-latex-recommended

# Essential packages for resumes
sudo apt-get install texlive-fonts-recommended texlive-latex-extra
```

**Fedora/RHEL:**
```bash
# Complete installation (~3GB)
sudo dnf install texlive-scheme-full

# Basic installation (~800MB)
sudo dnf install texlive-scheme-basic texlive-latex
```

**Arch Linux:**
```bash
# Full installation (~3GB)
sudo pacman -S texlive-most texlive-langextra

# Basic installation (~600MB)
sudo pacman -S texlive-core texlive-latexextra
```

#### Storage Requirements Summary
| Distribution | Basic | Full | Download Time |
|--------------|-------|------|---------------|
| MiKTeX (Windows) | 200MB | 2.5GB | 5-45 min |
| MacTeX (macOS) | 80MB | 4GB | 10-60 min |
| TeX Live (Linux) | 500MB | 3GB | 10-30 min |

#### Installation Verification
Test your installation works:
```bash
# Check LaTeX is installed
pdflatex --version

# Test compilation (creates test.pdf)
echo '\documentclass{article}\begin{document}Hello World\end{document}' > test.tex
pdflatex test.tex
```

#### Troubleshooting Installation
**Windows PATH issues:**
```bash
# Add to PATH manually:
# C:\Program Files\MiKTeX\miktex\bin\x64\
```

**macOS permission issues:**
```bash
# Fix permissions
sudo chown -R $(whoami) /usr/local/texlive
```

**Linux missing packages:**
```bash
# Install additional packages as needed
sudo apt-get install texlive-science texlive-pictures
```

## Usage

### Basic Optimization
```bash
# From job description
python resume_optimizer.py resume.tex job_description.txt --jd

# From keyword list
python resume_optimizer.py resume.tex keywords.txt

# Custom output
python resume_optimizer.py resume.tex job_description.txt --jd -o custom_output.tex
```

### Advanced Examples

#### Batch Processing Multiple Jobs
```bash
for jd in job_descriptions/*.txt; do
    python resume_optimizer.py resume.tex "$jd" --jd -o "output/$(basename $jd .txt)_resume.tex"
done
```

#### CI/CD Integration
```yaml
# GitHub Actions
- name: Optimize Resume
  run: |
    python resume_optimizer.py resume.tex job_description.txt --jd
    mv optimized_resume_*/resume_optimized.pdf artifacts/
```

### Input Formats

**Job Description (Recommended)**
Simply copy-paste any job posting text into a `.txt` file. The tool handles messy formatting automatically:

```text
Software Engineer @ TechCorp
$120K-180K | Remote | Full-time

We're looking for someone with:
• Python/JavaScript experience  
• Knowledge of AWS, Docker, Kubernetes
• Machine learning background preferred
• 3+ years building scalable systems

Apply now! Email: jobs@techcorp.com
```

**Keywords List**
Flexible format support - use whatever's convenient:

```text
# Comma-separated (single line)
Python, JavaScript, AWS, Docker, Kubernetes, machine learning

# Line-separated (multi-line)  
Python
JavaScript
AWS
Docker
Kubernetes
machine learning

# Mixed format (also works)
Python, JavaScript
AWS
Docker, Kubernetes
machine learning
```

## Troubleshooting

### PDF Compilation Issues

**If pdflatex fails:**
1. Check LaTeX installation: `pdflatex --version`
2. Install missing packages via MiKTeX Console (Windows) or `tlmgr` (Linux/Mac)

**Overleaf Fallback (Always Works)**
If local compilation fails:
1. Copy the generated `.tex` file content
2. Create new project at [overleaf.com](https://overleaf.com)
3. Paste content and compile online
4. Download PDF directly from Overleaf

This bypasses all local LaTeX installation issues.

### Common Errors

**"No changeable components detected"**
- Resume may have only static content (education, titles)
- Add more descriptive bullet points about technical work

**"Word count violation"**
- LLM exceeded ±4 word limit
- Try with fewer or shorter keywords

**"Keyword usage at maximum"**
- All keywords already used 2+ times
- Use different keywords or accept current optimization

## Performance & Optimization

### Component Detection Examples
**Changeable (Will be optimized):**
- `\item Developed web application using React and Node.js`
- `\item Implemented machine learning pipeline for data processing`
- `\item Built distributed system handling 10K requests/second`

**Non-changeable (Will be skipped):**
- `\item University of Illinois at Chicago, B.S. Computer Science, GPA: 3.8`
- `\item Software Engineer Intern, Google (June 2023 - August 2023)`
- `\item Dean's List, Outstanding Student Award`

### Keyword Distribution Logic
- Maximum 2 keywords per component
- Maximum 2 occurrences per keyword across entire resume
- Prioritizes least-used keywords for natural distribution

## Command Reference

```bash
python resume_optimizer.py [-h] [-o OUTPUT] [--jd] [--pdf] [--no-pdf] resume input_file

positional arguments:
  resume                LaTeX resume file (.tex)
  input_file           Keywords or job description file (.txt)

options:
  -h, --help           Show help message
  -o OUTPUT            Output file path (default: timestamped folder)
  --jd                 Input contains job description (extract keywords)
  --pdf                Compile to PDF (default: True)
  --no-pdf            Skip PDF compilation
```

## VS Code Integration

```json
// .vscode/tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Optimize Resume for Job",
            "type": "shell",
            "command": "python",
            "args": ["resume_optimizer.py", "${file}", "${input:jobFile}", "--jd"],
            "group": "build"
        }
    ],
    "inputs": [
        {
            "id": "jobFile",
            "type": "promptString", 
            "description": "Job description file path"
        }
    ]
}
```

## Ethical Guidelines

**Enhance, Don't Fabricate:**
- ✅ Improve keyword alignment for existing skills
- ✅ Optimize ATS compatibility
- ❌ Add skills you don't possess
- ❌ Misrepresent experience levels

## Contributing

```bash
git checkout -b feature/enhancement
pytest tests/ -v
black resume_optimizer.py --check
git commit -m 'Add enhancement'
```

## License

MIT License - see [LICENSE](./LICENSE)

## Support

- 📧 **Email**: pmishr23@uic.edu
- 🔗 **Portfolio**: [portfolio-pranav-mishra.vercel.app](https://portfolio-pranav-mishra.vercel.app)
- 💼 **LinkedIn**: [linkedin.com/in/pranavgamedev](https://www.linkedin.com/in/pranavgamedev/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/resumecraft/issues)

---

*Happy job hunting!*
