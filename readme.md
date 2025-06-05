# ResumeCraft 🎯

*AI-powered LaTeX resume optimization with intelligent keyword integration*

## Overview

ResumeCraft uses LLM-powered analysis to intelligently optimize LaTeX resumes for ATS systems. Unlike simple keyword stuffing, it identifies changeable resume components, enforces strict character limits, and tracks keyword usage to maintain natural content flow.

> **For Recruiters:** This tool demonstrates semantic analysis, LaTeX AST manipulation, constraint solving, and real-time validation—probably more engineering skill than any keyword could convey. 😉

![Screenshot](https://github.com/PranavMishra17/ResumeCraft-Latex-resume-optimizer/blob/c5eb36a9bb31ac2a28450fd9faf8770bad4118ed/resume.png)

## Key Features

- **Smart Component Detection**: LLM identifies which resume bullets can be meaningfully rewritten
- **Keyword Usage Tracking**: Maximum 2 occurrences per keyword prevents spam
- **Cloud PDF Compilation**: No local LaTeX installation required - uses web API
- **Strict Character Limits**: ±10 character constraint maintains resume formatting
- **Real-time Validation**: Preserves LaTeX structure and commands

## Quick Startup

```bash
# Clone repository
git clone https://github.com/PranavMishra17/ResumeCraft-Latex-resume-optimizer
cd ResumeCraft-Latex-resume-optimizer

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run optimization
python resume_optimizer.py your_resume.tex keywords.txt
python resume_optimizer.py your_resume.tex job_description.txt --jd --strict
```

## Technical Architecture

### Core Innovation: Component-Level Intelligence
- **LLM Component Detection**: Automatically identifies which resume bullets can be meaningfully rewritten
- **Keyword Usage Tracking**: Enforces maximum 2 occurrences per keyword to prevent spam
- **Strict Character Limits**: ±10 character constraint maintains resume formatting
- **Cloud PDF Generation**: Web-based LaTeX compilation eliminates local setup

### Optimization Flow
1. **Keyword Extraction**: Extract relevant keywords from job descriptions using LLM analysis
2. **Existing Usage Analysis**: Count current keyword occurrences in resume
3. **Component Detection**: LLM identifies changeable components (project descriptions, technical work)
4. **Smart Distribution**: Assign 1-2 keywords per component, prioritizing least-used keywords
5. **Constraint Optimization**: Rewrite with strict character count enforcement (±10 chars)
6. **Cloud Compilation**: Generate PDF via web API without local LaTeX installation

## API Configuration

### Azure OpenAI (Default)
```bash
export AZURE_API_KEY="your-api-key"
export AZURE_ENDPOINT="https://your-resource.openai.azure.com/"
```

### OpenAI API Alternative
```python
# In config.py
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL = "gpt-4o"
    TEMPERATURE = 0.2

# Update client initialization
from langchain_openai import ChatOpenAI
self.client = ChatOpenAI(
    api_key=self.config.OPENAI_API_KEY,
    model=self.config.MODEL,
    temperature=self.config.TEMPERATURE
)
```

## Usage

### Basic Optimization
```bash
# From job description
python resume_optimizer.py resume.tex job_description.txt --jd

# From keyword list
python resume_optimizer.py resume.tex keywords.txt

# Strict mode with retry logic
python resume_optimizer.py resume.tex job_description.txt --jd --strict
```

### Input Formats

**Job Description (Recommended)**
Simply copy-paste any job posting text into a `.txt` file:

```text
Software Engineer @ TechCorp
$120K-180K | Remote | Full-time

We're looking for someone with:
• Python/JavaScript experience  
• Knowledge of AWS, Docker, Kubernetes
• Machine learning background preferred
```

**Keywords List**
```text
# Comma-separated (single line)
Python, JavaScript, AWS, Docker, Kubernetes, machine learning

# Line-separated (multi-line)  
Python
JavaScript
AWS
Docker
```

## PDF Compilation

ResumeCraft automatically compiles your optimized LaTeX to PDF using cloud-based compilation:

- **No Local Setup**: No LaTeX installation required
- **Multiple Compilers**: Supports pdflatex, xelatex, lualatex
- **Automatic Fallback**: Falls back to local compilation if available
- **Standard Naming**: Outputs as `Resume.pdf`

### Overleaf Fallback
If compilation fails:
1. Copy generated `.tex` content
2. Create project at [overleaf.com](https://overleaf.com)
3. Paste content and compile online
4. Download PDF directly

## Performance & Optimization

### Component Detection Examples
**Changeable (Will be optimized):**
- `\item Developed web application using React and Node.js`
- `\item Implemented machine learning pipeline for data processing`
- `\item Built distributed system handling 10K requests/second`

**Non-changeable (Will be skipped):**
- `\item University of Illinois, B.S. Computer Science, GPA: 3.8`
- `\item Software Engineer Intern, Google (June 2023 - August 2023)`
- `\item Dean's List, Outstanding Student Award`

### Keyword Distribution Logic
- Maximum 2 keywords per component
- Maximum 2 occurrences per keyword across entire resume
- Prioritizes least-used keywords for natural distribution

## Command Reference

```bash
python resume_optimizer.py [-h] [-o OUTPUT] [--jd] [--pdf] [--no-pdf] [--strict] resume input_file

positional arguments:
  resume                LaTeX resume file (.tex)
  input_file           Keywords or job description file (.txt)

options:
  -h, --help           Show help message
  -o OUTPUT            Output file path (default: timestamped folder)
  --jd                 Input contains job description (extract keywords)
  --pdf                Compile to PDF (default: True)
  --no-pdf            Skip PDF compilation
  --strict             Enable retry logic for character limits
```

## Troubleshooting

### Common Errors

**"No changeable components detected"**
- Resume may have only static content (education, titles)
- Add more descriptive bullet points about technical work

**"Character count violation"**
- LLM exceeded ±10 character limit
- Use `--strict` for retry logic

**"PDF compilation failed"**
- Uses cloud compilation automatically
- Fallback: copy `.tex` to Overleaf for manual compilation

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


## 🤝 Contributors

| Name             | GitHub                                           | LinkedIn                                        |
|------------------|--------------------------------------------------|-------------------------------------------------|
| Pranav Mishra    | [@PranavMishra17](https://github.com/PranavMishra17) | [Pranav Mishra](https://www.linkedin.com/in/pranavgamedev/) |
| Pranav Vasist    | [@Pranav2701](https://github.com/pranav2701)         | [Pranav Vasist](https://www.linkedin.com/in/pranav-vasist/) |
| Kranti Yeole     | [@KrantiYeole20](https://github.com/krantiyeole20)   | [Kranti Yeole](https://www.linkedin.com/in/krantiyeole/)    |



## License

MIT License - see [LICENSE](./LICENSE)

## Support

- 📧 **Email**: pmishr23@uic.edu
- 🔗 **Portfolio**: [portfolio-pranav-mishra.vercel.app](https://portfolio-pranav-mishra.vercel.app)
- 💼 **LinkedIn**: [linkedin.com/in/pranavgamedev](https://www.linkedin.com/in/pranavgamedev/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/resumecraft/issues)

---

*Happy job hunting!*
