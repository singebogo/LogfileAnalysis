# LogfileAnalysis 🚀

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A brief one-liner describing your project.

**Key Features**:
- 📦 Installation
- 🛠 Usage
- ✅ screenshots/GIFs
- 🤝 Contributing
- 📄 License

---

## 📦 Installation
        
### Clone the repo
    git clone https://github.com/yourusername/project-name.git
    cd project-name
### Create a .env file
    virtualenv .env
### Install dependencies
    pip install -r requirements.txt

### For development:
    pip install -r dev-requirements.txt
    
### Migrate Model
    set Flask_db = main.py
    flask db init
    flask db migrate 
    flask db upgrade 
### run 
    python main.py    

---
    
# 🛠 Usage

## Basic Example
    Or use the API:

        POST /api/analyze
        Content-Type: multipart/form-data

        Send log files as 'files' parameter (multiple files allowed)
    

# 📂 Project Structure

    project/
    |   .gitignore
    |   .python-version
    |   LICENSE
    |   main.py                              --主程序
    |   pyproject.toml
    |   README.md
    |   uv.lock            
    +---imges
    +---log_analyzer
    |   |   extensions.py
    |   |   handle_request.py
    |   |   log_analyzer_template.py        --日志文件解析
    |   |   log_parser.py
    |   |   models.py                       -- 模型
    |   |   requirements.txt
    |   |   servers.py                      -- 数据库操作
    |   |   __init__.py
    |   |   
    |   +---blueprint
    |   |   |   historys.py
    |   |   |   __init__.py
    |   |   |   
    |   +---config                          -- 基础配置包
    |   |   |   base.py
    |   |   |   development.py
    |   |   |   production.py
    |   |   |   testing.py
    |   |   |   __init__.py
    |   |           
    |   +---reports
    |   +---static
    |   |   +---css     
    |   |   \---js       
    |   +---templates
    |   |       base.html
    |   |       cleanup.html
    |   |       error.html
    |   |       history.html
    |   |       index.html
    |   |       report_template.html
    |   |       results.html
    |   |       settings.html
    |   |       
    |   +---uploads
    |   +---utils                        -- 工具包
    |   |   |   cleans.py
    |   |   |   combines.py
    |   |   |   errors.py
    |   |   |   files.py
    |   |   |   __init__.py
    |           
    +---migrations
    +---reports     
    +---uploads

# 🎨screenshots/GIFs
![2025-08-07_175733](vx_images/189085917250848.png)
        
 ![2025-08-07_175942](vx_images/13870018269274.png)

![2025-08-07_180042](vx_images/215760118257141.png)

![2025-08-07_180138](vx_images/560840118277307.png)

![2025-08-07_180418](vx_images/540110418269976.png)
![2025-08-07_180701](vx_images/432820718266531.png)
![2025-08-07_180801](vx_images/252120818262285.png)
![2025-08-07_180935](vx_images/587091018280165.png)
![配置](vx_images/463161118277769.png)
# 🤝 Contributing

    Fork the repository.

    Create a branch (git checkout -b feature/awesome-feature).

    Commit changes (git commit -m 'Add awesome feature').

    Push to the branch (git push origin feature/awesome-feature).

    Open a Pull Request.

# 📄 License

    Distributed under the MIT License. See LICENSE for details.    