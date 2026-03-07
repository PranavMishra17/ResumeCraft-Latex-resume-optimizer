<div align="center">
  <img src="https://github.com/PranavMishra17/ResumeCraft-Latex-resume-optimizer/blob/c5eb36a9bb31ac2a28450fd9faf8770bad4118ed/resume.png" alt="ResumeCraft Banner" width="800"/>

  <h1>ResumeCraft 🎯</h1>

  <p>
    <em>AI-powered LaTeX resume optimization with intelligent keyword integration and PDF compilation</em>
  </p>
</div>

---

## Overview

ResumeCraft uses LLM-powered analysis to intelligently optimize LaTeX resumes for ATS systems. Unlike simple keyword stuffing, it identifies changeable resume components, enforces strict character limits, and tracks keyword usage to maintain natural content flow.

> **For Recruiters:** This tool demonstrates semantic analysis, LaTeX AST manipulation, constraint solving, and real-time validation—probably more engineering skill than any keyword could convey. 😉

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
python main.py resume.tex keywords.txt
python main.py resume.tex job_description.txt --jd --strict

# Recompile optimized resume
python pdf_compiler.py optimized_resume.tex
```

## Technical Architecture

### Core Innovation: Component-Level Intelligence
- **LLM Component Detection**: Automatically identifies which resume bullets can be meaningfully rewritten
- **Keyword Usage Tracking**: Enforces maximum 2 occurrences per keyword to prevent spam
- **Strict Character Limits**: ±10 character constraint maintains resume formatting
- **Cloud PDF Generation**: Web-based LaTeX compilation eliminates local setup
- **Modular & Multi-Provider**: Support for Azure, OpenAI, Gemini, and Claude LLMs out of the box with a cleanly separated architecture.

### Optimization Flow
1. **Keyword Extraction**: Extract relevant keywords from job descriptions using LLM analysis
2. **Existing Usage Analysis**: Count current keyword occurrences in resume
3. **Component Detection**: LLM identifies changeable components (project descriptions, technical work)
4. **Smart Distribution**: Assign 1-2 keywords per component, prioritizing least-used keywords
5. **Constraint Optimization**: Rewrite with strict character count enforcement (±10 chars)
6. **Result Overwrite**: Generates a single `optimized_resume.tex` file in the root for easy manual edits.
7. **Cloud Compilation**: Generate PDF via local or web API isolation in `pdfs/<timestamp>/Resume.pdf`.

## 🔧 API Configuration

### Environment Setup

Create a `.env` file in your project root and set your preferred provider.

```env
# Choose your active provider: azure, openai, gemini, or anthropic
LLM_PROVIDER=azure
TEMPERATURE=0.8

# Azure OpenAI (Default)
AZURE_DEPLOYMENT=gpt-4o-Krantiji
AZURE_API_KEY=your-azure-api-key
AZURE_API_VERSION=2024-12-01-preview
AZURE_ENDPOINT=https://your-resource.openai.azure.com/

# OpenAI Alternative  
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o

# Google Gemini Alternative
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-1.5-pro

# Anthropic Claude Alternative
ANTHROPIC_API_KEY=your-claude-api-key
ANTHROPIC_MODEL=claude-3-5-sonnet-20240620
```

ResumeCraft dynamically imports the correct LangChain library based on `LLM_PROVIDER`. Ensure you have installed the respective package (`langchain-openai`, `langchain-google-genai`, or `langchain-anthropic`).

## Usage

### Basic Optimization
```bash
# From job description
python main.py resume.tex job_description.txt --jd

# From keyword list
python main.py resume.tex keywords.txt

# Strict mode with retry logic
python main.py resume.tex job_description.txt --jd --strict
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

## Workflow & PDF Compilation

ResumeCraft adopts a standardized output workflow:
1. **Single TeX Output**: The optimized resume is always saved to `optimized_resume.tex` in the root folder. You can safely open and manually edit this single file knowing it represents your latest optimization. The original TeX file remains untouched.
2. **Isolated PDF Folders**: On each successful run, a timestamped folder is created inside the `pdfs/` directory (e.g. `pdfs/YYYYMMDD_HHMMSS/Resume.pdf`). This keeps your root folder clean while letting you track PDF history over time.

- **No Local Setup**: No LaTeX installation required. Falls back to LaTeX-on-HTTP API if local compilation fails.
- **Multiple Compilers**: Supports pdflatex out of the box.

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
python main.py [-h] [--jd] [--no-pdf] [--strict] resume input_file

positional arguments:
  resume                Path to original LaTeX resume file (.tex)
  input_file           Path to keywords or job description file (.txt)

options:
  -h, --help           Show help message
  --jd                 Input contains job description (extract keywords)
  --no-pdf            Skip PDF compilation entirely
  --strict             Enable strict mode for word/char limits
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

---

## Connect with me

<table align="center">
<tr>
<td width="200px">
  <img src="me.jpg" alt="Pranav Mishra" width="180" style="border: 5px solid; border-image: linear-gradient(45deg, #9d4edd, #ff006e) 1;">
</td>
<td>
  
[![Portfolio](https://img.shields.io/badge/-Portfolio-000?style=for-the-badge&logo=vercel&logoColor=white)](https://portfolio-pranav-mishra-paranoid.vercel.app)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pranavgamedev/)
[![Resume](https://img.shields.io/badge/-Resume-4B0082?style=for-the-badge&logo=read-the-docs&logoColor=white)](https://portfolio-pranav-mishra-paranoid.vercel.app/resume)
[![YouTube](https://img.shields.io/badge/-YouTube-8B0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@parano1dgames/featured)
[![Hugging Face](https://img.shields.io/badge/-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Paranoiid)

</td>
</tr>
</table>

<div align="center">
---

*Happy job hunting!*
