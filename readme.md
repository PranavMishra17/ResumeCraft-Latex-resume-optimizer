# ResumeCraft 🎯

*AI-powered LaTeX resume optimization with intelligent keyword integration*

## Overview

ResumeCraft leverages Azure OpenAI's GPT-4 to intelligently optimize LaTeX resumes for ATS systems while preserving document structure and semantic meaning. The tool performs contextual keyword placement, handles complex modular LaTeX structures, and maintains strict formatting constraints.

> **For Recruiters:** Yes, this technically "games" ATS systems - but building an AI tool that parses LaTeX AST, maintains document invariants, handles distributed file structures, and performs semantic keyword mapping while preserving compilation targets probably demonstrates more engineering competence than any keyword ever could. 😉

## Technical Architecture

### Core Components
- **LaTeX Parser**: Handles `\subfile{}` resolution and document tree construction
- **LLM Integration**: Azure OpenAI GPT-4 for semantic understanding and rewriting
- **Constraint Solver**: Maintains word count limits and structural integrity
- **PDF Compiler**: Automated `pdflatex` invocation with error handling

### Key Algorithms
- **Keyword Extraction**: TF-IDF-like importance scoring from job descriptions
- **Section Mapping**: Semantic similarity matching (ML keywords → ML sections)
- **Word Count Preservation**: Per-bullet-point token counting and replacement
- **Structure Validation**: AST comparison pre/post optimization

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/resumecraft.git
cd resumecraft
```

### 2. Python Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `langchain-openai>=0.1.0`
- `langchain-core>=0.2.0`
- `azure-identity` (optional for managed identity)

### 3. LaTeX Distribution (Required for PDF)

**Windows - MiKTeX**
- Download: https://miktex.org/download
- Install with automatic package installation enabled
- Verify: `pdflatex --version`

**macOS - MacTeX**
- Download: https://www.tug.org/mactex/
- Full distribution (~4GB) or BasicTeX (~80MB)
- Verify: `which pdflatex`

**Linux - TeX Live**
```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# Fedora
sudo dnf install texlive-scheme-full

# Arch
sudo pacman -S texlive-core texlive-latexextra
```

### 4. Azure OpenAI Configuration
```bash
# Environment variables
export AZURE_API_KEY="your-api-key"
export AZURE_ENDPOINT="https://your-resource.openai.azure.com/"

# Or modify in resume_optimizer.py:
AZURE_DEPLOYMENT = "your-deployment-name"
AZURE_API_VERSION = "2024-08-01-preview"
```

## Usage

### Basic Commands
```bash
# Optimize with keyword list
python resume_optimizer.py resume.tex keywords.txt

# Extract keywords from job description
python resume_optimizer.py resume.tex job_description.txt --jd

# Custom output location
python resume_optimizer.py resume.tex keywords.txt -o output/resume_v2.tex

# Debug mode (keep all intermediate files)
python resume_optimizer.py resume.tex keywords.txt --keep-tex --no-pdf
```

### Advanced Usage

#### Modular Resume Support
```
project/
├── main_resume.tex         # Contains \subfile{} commands
├── subsections/
│   ├── education.tex
│   ├── experience.tex
│   ├── projects.tex
│   └── skills.tex
└── job_description.txt
```

```bash
# Automatically detects and processes subfiles
python resume_optimizer.py main_resume.tex job_description.txt --jd
```

#### Batch Processing
```bash
# Process multiple job descriptions
for jd in job_descriptions/*.txt; do
    python resume_optimizer.py resume.tex "$jd" --jd -o "output/$(basename $jd .txt)/"
done
```

#### Integration with CI/CD
```yaml
# GitHub Actions example
- name: Optimize Resume
  run: |
    python resume_optimizer.py resume.tex job_description.txt --jd
    mv optimized_resume_*/resume_optimized.pdf artifacts/
```

### Input Formats

**keywords.txt** (multiple formats supported):
```
# Comma-separated
Python, C++, Machine Learning, Distributed Systems, CUDA, PyTorch

# Line-separated
Python
C++
Machine Learning
Distributed Systems
```

**job_description.txt**:
```
Senior Software Engineer - ML Infrastructure

We're seeking an engineer with experience in:
- Python and C++ for high-performance computing
- Distributed systems and data pipelines
- ML frameworks (PyTorch, TensorFlow)
- GPU programming (CUDA/ROCm)
```

## Technical Details

### LaTeX Preservation Rules
- Maintains all `\begin{}` and `\end{}` environments
- Preserves `\item` count and list structures
- Respects custom commands and packages
- Handles unicode and special characters
- Maintains bibliography and citations

### Optimization Constraints
```python
# Per-bullet-point word count enforcement
original: "\item Developed web application using React"  # 6 words
valid:    "\item Developed Python application using FastAPI"  # 6 words
invalid:  "\item Developed distributed Python ML application"  # 7 words
```

### Error Handling
- **LaTeX Compilation Errors**: Parses `pdflatex` output and highlights issues
- **Structure Corruption**: Validates AST and falls back to original
- **API Failures**: Retries with exponential backoff
- **File System**: Creates timestamped backups

## Command Reference

```
usage: resume_optimizer.py [-h] [-o OUTPUT] [--jd] [--pdf] [--no-pdf] [--keep-tex] resume input_file

positional arguments:
  resume                Path to LaTeX resume file (.tex)
  input_file            Path to keywords or job description file (.txt)

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output LaTeX file path (default: timestamped folder)
  --jd                  Input file contains job description (extract keywords)
  --pdf                 Compile to PDF after optimization (default: True)
  --no-pdf              Skip PDF compilation
  --keep-tex            Keep intermediate .tex files
```

## VS Code Integration

### Task Configuration (`.vscode/tasks.json`)
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Optimize Resume",
            "type": "shell",
            "command": "python",
            "args": [
                "resume_optimizer.py",
                "${file}",
                "${input:jobDescription}",
                "--jd"
            ],
            "group": "build",
            "presentation": {
                "reveal": "always",
                "panel": "new"
            }
        }
    ],
    "inputs": [
        {
            "id": "jobDescription",
            "type": "promptString",
            "description": "Path to job description file"
        }
    ]
}
```

### Launch Configuration (`.vscode/launch.json`)
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Resume Optimizer",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/resume_optimizer.py",
            "args": ["resume.tex", "job_description.txt", "--jd", "--keep-tex"],
            "console": "integratedTerminal"
        }
    ]
}
```

## Troubleshooting

### Common Issues

**"pdflatex not found"**
- Ensure LaTeX is in PATH: `echo $PATH`
- Windows: Run MiKTeX Console → Settings → Update PATH
- macOS/Linux: Add to `.bashrc`: `export PATH="/usr/local/texlive/2023/bin/x86_64-linux:$PATH"`

**"LaTeX Error: Lonely \item"**
- LLM corrupted list structure
- Check logs: `optimized_resume_*/ai_resume_optimized.log`
- Use `--keep-tex` to inspect output

**"Azure OpenAI Error"**
- Verify API key and endpoint
- Check deployment name matches
- Ensure quota isn't exceeded

### Debug Output
```bash
# Enable verbose logging
export AZURE_OPENAI_LOG_LEVEL=debug
python resume_optimizer.py resume.tex keywords.txt
```

## Performance Optimization

- **Caching**: Reuses keyword extraction across runs
- **Parallel Processing**: Multi-file support planned
- **Token Optimization**: Minimizes API calls through prompt engineering
- **LaTeX Compilation**: Two-pass compilation for references

## Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `pytest tests/`
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push branch: `git push origin feature/amazing-feature`
6. Open Pull Request

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Check formatting
black resume_optimizer.py --check
flake8 resume_optimizer.py
```

## Ethical Considerations

This tool optimizes presentation, not fabrication:
- ✅ Highlight existing skills with relevant keywords
- ✅ Improve ATS compatibility for qualified candidates
- ❌ Don't add skills you don't possess
- ❌ Don't misrepresent experience levels

## License

This project is licensed under the [MIT License](./LICENSE).


## Support

- 📧 Email: pmishr23@uic.edu
- 🔗 Portfolio: [portfolio-pranav-mishra.vercel.app](https://portfolio-pranav-mishra.vercel.app)
- 💼 LinkedIn: [linkedin.com/in/pranavgamedev](https://www.linkedin.com/in/pranavgamedev/)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/resumecraft/issues)

---

*Built with ❤️ and respect for both candidates and recruiters*