# ResumeCraft 🎯

*Intelligently optimize LaTeX resumes with AI-powered keyword integration*

## Description

ResumeCraft automates LaTeX resume customization using Azure OpenAI to strategically incorporate job-relevant keywords while maintaining formatting and structure. 

> **For Recruiters:** Yes, this is technically "gaming the system" - but building an AI-powered LaTeX automation tool that preserves document structure, handles modular files, integrates with cloud APIs, and maintains semantic meaning probably tells you more about my engineering skills than any keyword-stuffed bullet point ever could. 😉

## Features

- **Smart Keyword Integration**: Maps keywords to relevant resume sections (ML → ML projects, Web Dev → Web projects)
- **Modular LaTeX Support**: Handles both single-file and `\subfile`-based resume structures
- **Job Description Parsing**: Automatically extracts keywords from job descriptions using LLM
- **Format Preservation**: Maintains exact LaTeX formatting, spacing, and one-page constraint
- **PDF Generation**: Compiles optimized resumes directly to PDF
- **Original File Safety**: Never modifies source files - creates timestamped output folders

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/resumecraft.git
cd resumecraft
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Azure OpenAI**
```bash
export AZURE_API_KEY="your-api-key"
```

Update `AZURE_DEPLOYMENT` and `AZURE_ENDPOINT` in `resume_optimizer.py`.

4. **Install LaTeX** (for PDF compilation)
- **Windows**: MiKTeX
- **macOS**: MacTeX  
- **Linux**: `sudo apt-get install texlive-full`

## Usage

### Basic Usage
```bash
# With keywords file
python resume_optimizer.py resume.tex keywords.txt

# With job description (auto-extracts keywords)
python resume_optimizer.py resume.tex job_description.txt --jd

# Keep LaTeX files for inspection
python resume_optimizer.py resume.tex keywords.txt --keep-tex
```

### File Structure
```
resumecraft/
├── resume_optimizer.py      # Main script
├── requirements.txt
├── resume.tex              # Your original resume
├── subsections/            # Subfile components (if modular)
│   ├── education.tex
│   ├── experience.tex
│   └── skills.tex
├── keywords.txt            # Keywords or job description
└── optimized_resume_*/     # Generated output folders
    └── resume_optimized.pdf
```

### Input Files

**keywords.txt** (comma-separated):
```
React, Node.js, Python, Machine Learning, TensorFlow, Docker, Kubernetes, AWS
```

**job_description.txt** (full job posting):
```
We're looking for a Full Stack Developer with experience in React, Node.js, 
and cloud platforms like AWS. Knowledge of Docker and CI/CD pipelines preferred...
```

## Command Line Options

```bash
python resume_optimizer.py [resume.tex] [input.txt] [options]

Options:
  --jd              Input file contains job description (extract keywords)
  -o, --output      Custom output path
  --keep-tex        Keep intermediate LaTeX files
  --no-pdf          Skip PDF compilation
```

## VS Code Integration

Run directly in VS Code terminal:
```bash
python resume_optimizer.py resume.tex keywords.txt
```

Or add to `.vscode/tasks.json`:
```json
{
    "label": "Optimize Resume",
    "type": "shell",
    "command": "python",
    "args": ["resume_optimizer.py", "resume.tex", "${input:keywordFile}"]
}
```

## How It Works

1. **Parse Structure**: Detects modular LaTeX files and combines content
2. **Extract Keywords**: Processes job descriptions or keyword lists
3. **AI Optimization**: Uses Azure OpenAI to strategically place keywords
4. **Smart Mapping**: Routes keywords to relevant sections based on content
5. **Preserve Format**: Maintains LaTeX structure and one-page constraint
6. **Generate Output**: Creates PDF in timestamped folder

## Requirements

- Python 3.8+
- Azure OpenAI API access
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Dependencies: `langchain-openai`, `langchain-core`

## Example Output

```
Reading resume: resume.tex
Found 5 subfiles
  - subsections/education.tex
  - subsections/experience.tex
  - subsections/skills.tex
Extracted keywords: React, Node.js, Python, AWS, Docker
Optimizing resume...
Optimized resume saved to: optimized_resume_20250601_143022/
Compiling PDF...
PDF compiled successfully
Final output: optimized_resume_20250601_143022/resume_optimized.pdf
Resume optimization completed!
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Ethical Use

This tool is designed to help candidates highlight relevant skills they actually possess. Use responsibly:
- Only include keywords for technologies you have genuine experience with
- Maintain truthful representations of your background
- Use as a formatting and relevance tool, not for misrepresentation

## License

MIT License

Copyright (c) 2025 Pranav Pushkar Mishra

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Support

- 📧 Contact: pmishr23@uic.edu
- 🔗 Portfolio: [portfolio-pranav-mishra](https://portfolio-pranav-mishra-paranoid.vercel.appp)
- 💼 LinkedIn: [pranavmishrabarca](https://www.linkedin.com/in/pranavgamedev//)

---

*Built with ❤️ and a healthy dose of automation*