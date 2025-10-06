# 🤖 Data Analysis Agent

<div align="center">

**An interactive, agentic data analysis application powered by NVIDIA's advanced LLM reasoning models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![NVIDIA API](https://img.shields.io/badge/NVIDIA-API-76B900?logo=nvidia&logoColor=white)](https://build.nvidia.com/)

</div>

---

## 📋 Overview

This repository contains a powerful Streamlit application that demonstrates a complete agentic workflow for data analysis:

- 📊 **Data Upload**: Upload CSV files for analysis
- 💬 **Natural Language Queries**: Ask questions about your data in plain English
- 📈 **Automated Visualization**: Generate relevant plots and charts
- 🧠 **Transparent Reasoning**: Get detailed explanations of the analysis process

The implementation leverages the powerful **Llama-3.1-Nemotron-Ultra-253B-v1** and **Llama-3.3-Nemotron-Super-49B-v1.5** models through NVIDIA's API, enabling sophisticated data analysis and reasoning.

> 💡 Learn more about the models [here](https://developer.nvidia.com/blog/build-enterprise-ai-agents-with-advanced-open-nvidia-llama-nemotron-reasoning-models/)

---

## ✨ Features

- 🔧 **Agentic Architecture**: Modular agents for data insight, code generation, execution, and reasoning
- 🗣️ **Natural Language Queries**: Ask questions about your data—no coding required
- 📊 **Automated Visualization**: Instantly generate and display relevant plots
- 🔍 **Transparent Reasoning**: Get clear, LLM-generated explanations for every result
- ⚡ **Powered by NVIDIA**: State-of-the-art reasoning and interpretability

![Workflow](./assets/workflow.png)

---

## 🚀 Quick Start

### Prerequisites

- 🐍 Python 3.10 or higher
- 🔑 NVIDIA API Key ([Get one here](https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1?integrate_nim=true&hosted_api=true&modal=integrate-nim))

---

## 📦 Installation

### Step 1️⃣: Clone the Repository

```bash
git clone https://github.com/NVIDIA/GenerativeAIExamples.git
cd GenerativeAIExamples/community/data-analysis-agent
```

### Step 2️⃣: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `streamlit` - Web interface
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `requests` - API calls

---

## 🔐 Environment Setup

### Option A: Use `.env` File (Recommended - Easiest!)

1. **Copy the template file:**
   ```bash
   copy .env.template .env
   ```
   Or on Linux/Mac:
   ```bash
   cp .env.template .env
   ```

2. **Edit `.env` file and add your API key:**
   ```
   NVIDIA_API_KEY=your_actual_api_key_here
   ```

3. **That's it!** The app will automatically load your API key.

### Option B: Export Environment Variable

**On Linux/Mac:**
```bash
export NVIDIA_API_KEY=your_nvidia_api_key_here
```

**On Windows (PowerShell):**
```powershell
$env:NVIDIA_API_KEY="your_nvidia_api_key_here"
```

**On Windows (CMD):**
```cmd
set NVIDIA_API_KEY=your_nvidia_api_key_here
```

### 🔑 Getting Your API Key

1. Sign up or log in at [NVIDIA Build](https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1?integrate_nim=true&hosted_api=true&modal=integrate-nim)
2. Navigate to the API section
3. Generate your API key
4. Copy and save it securely

---

## ▶️ Running the Application

### Start the Streamlit App

```bash
streamlit run data_analysis_agent.py
```

The app will automatically open in your default browser at `http://localhost:8501`

### 📥 Optional: Download Example Dataset

```bash
# Download the Titanic dataset for testing
wget https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

Or manually download from the URL above and save as `titanic.csv`

---

## 🎯 Usage Guide

### Step 1: Select a Model
Choose from the dropdown menu:
- **NVIDIA Llama 3.1 Nemotron Ultra 253B v1** (253B parameters)
- **NVIDIA Llama 3.3 Nemotron Super 49B v1.5** (49B parameters)

### Step 2: Upload Your Data
- Click "Browse files" to upload a CSV file
- Example: Use the Titanic dataset or your own data

### Step 3: Ask Questions
Type natural language questions like:
- "What is the average age of passengers?"
- "Show me a plot of survival rates by class"
- "Which features correlate most with survival?"

### Step 4: View Results
- 📊 See visualizations rendered inline
- 📝 Read the reasoning behind each answer
- 💻 Expand the code accordion to see generated Python code

---

## 🎬 Example

![App Demo](./assets/data_analysis_agent_demo.png)

**Example queries you can try:**
- "Show me the distribution of ages"
- "What is the survival rate by passenger class?"
- "Create a bar chart comparing male and female survival rates"
- "Calculate the correlation between fare and survival"

---

## 🤖 Model Details

### Llama-3.1-Nemotron-Ultra-253B-v1
- **Parameters**: 253B
- **Features**: Advanced reasoning capabilities
- **Use Cases**: Complex data analysis, multi-agent systems
- **Enterprise Ready**: ✅ Optimized for production deployment

### Llama-3.3-Nemotron-Super-49B-v1.5
- **Parameters**: 49B
- **Features**: Efficient reasoning and chat model
- **Use Cases**: AI Agent systems, chatbots, RAG systems, typical instruction-following tasks
- **Enterprise Ready**: ✅ Optimized for production deployment

---

## 🙏 Acknowledgments

- [NVIDIA Llama-3.1-Nemotron-Ultra-253B-v1](https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1)
- [NVIDIA Llama-3.3-Nemotron-Super-49B-v1.5](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5)
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- 🐛 Report bugs
- 💡 Suggest new features
- 🔧 Submit pull requests

**Please open an issue or submit a pull request to get started!**

---

## 📄 License

This project follows the licensing terms of the NVIDIA GenerativeAIExamples repository.

---

<div align="center">

**Made with ❤️ using NVIDIA AI**

[⭐ Star this repo](https://github.com/NVIDIA/GenerativeAIExamples) | [📖 Documentation](https://developer.nvidia.com/) | [💬 Discussions](https://github.com/NVIDIA/GenerativeAIExamples/discussions)

</div>
