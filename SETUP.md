# 🚀 ANEETA Setup Guide

**ANEETA - Agents for National Eligibility cum Entrance Test Assistance**

This guide will help you set up the ANEETA AI tutoring system on your local machine for NEET preparation.

---

## 📋 System Requirements

- **RAM**: At least 4 GB available (8 GB+ recommended)
- **Storage**: 10 GB free space
- **OS**: Windows 10/11, macOS, or Linux
- **Internet**: Required for initial setup only

### 💾 Check Available Disk Space

Before downloading models, check your available disk space:

#### Windows:

```cmd
# Check disk space for C: drive (where Ollama stores models)
dir C:\ /-c

# Or use PowerShell for detailed info
Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='C:'" | Select-Object Size,FreeSpace
```

#### macOS:

```bash
# Check available space
df -h

# Check Ollama model directory specifically
du -sh ~/.ollama
```

#### Linux:

```bash
# Check available space
df -h

# Check Ollama model directory specifically
du -sh ~/.ollama
```

### 📦 Model Size Reference

| Model              | Size   | Memory Usage | Performance         |
| ------------------ | ------ | ------------ | ------------------- |
| `phi4-mini`        | 2.5 GB | ~3-4 GB RAM  | Good for most tasks |
| `gemma2:2b`        | 1.7 GB | ~2-3 GB RAM  | Lighter, faster     |
| `gemma2`           | 5.4 GB | ~7-8 GB RAM  | Better quality      |
| `nomic-embed-text` | 274 MB | ~500 MB RAM  | Required for search |

**💡 Tip:** Start with `phi4-mini` + `nomic-embed-text` (total ~2.8 GB) for a good balance of performance and resource usage.

---

## 🛠️ Step 1: Install Ollama

### Windows:

1. Download Ollama from [https://ollama.ai](https://ollama.ai)
2. Run the installer and follow the setup wizard
3. Open Command Prompt or PowerShell to verify installation:
   ```cmd
   ollama --version
   ```

### macOS:

```bash
# Using Homebrew
brew install ollama

# Or download from https://ollama.ai
```

### Linux:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

---

## 🤖 Step 2: Install Required AI Models

Open your terminal/command prompt and run:

```bash
# Install the main language model (2.5 GB - memory efficient)
ollama pull phi4-mini

# Install the embedding model for vector search (274 MB)
ollama pull nomic-embed-text
```

**💡 Alternative Models (if phi4-mini doesn't work):**

- For systems with more memory (6+ GB): `ollama pull gemma2:2b`
- For very limited memory (2-3 GB): Check `ollama list` for smaller models

---

## 🐍 Step 3: Install Python Dependencies

### Option A: Using pip (Recommended)

```bash
pip install --user pandas "numpy==1.26.4" langchain python-dotenv ipykernel langchain_community pypdf langchain-openai "chromadb==0.4.24" langchain-ollama sentence-transformers langgraph streamlit
```

### Option B: If you get permission errors

```bash
pip install --user -r requirements.txt
```

**📝 Note:** The `--user` flag installs packages in your user directory to avoid permission issues.

---

## ⚙️ Step 4: Configure Environment Variables

Create or update the `.env` file in the project root with the following configuration:

```env
# --- LLM and Embedding Model Configuration ---
# Specifies the model name for the primary LLM used for routing, RAG, and general tasks.
LLM_MODEL="phi4-mini"

# Specifies the model name for the creative LLM, used specifically for generating quiz questions.
# Note: In the original code, this is the same model as LLM_MODEL but with a higher temperature.
CREATIVE_LLM_MODEL="phi4-mini"

# Specifies the embedding model for vectorizing text.
EMBEDDING_MODEL="nomic-embed-text"

# --- Vector Database Configuration ---
# Base path where the Chroma vector database folders are located.
VECTORDB_BASE_PATH="src/aneeta/vectordb"
```

---

## 🚀 Step 5: Run ANEETA

### Windows:

```cmd
# Navigate to the ANEETA directory
cd "path\to\your\ANEETA\folder"

# Run the application
streamlit run app.py
```

**If `streamlit` command is not recognized:**

```cmd
# Use the full path (replace 'YourUsername' with your actual username)
C:\Users\YourUsername\AppData\Roaming\Python\Python312\Scripts\streamlit.exe run app.py
```

### macOS/Linux:

```bash
# Navigate to the ANEETA directory
cd /path/to/your/ANEETA/folder

# Run the application
streamlit run app.py
```

---

## 🌐 Step 6: Access ANEETA

1. Open your web browser
2. Go to: **http://localhost:8501**
3. Start using ANEETA for NEET preparation!

---

## 🎭 Meet the ANEETA Agents

### 👩🏽‍⚕️ Mentor Agent

- Provides personalized study plans
- Offers motivational coaching
- Time management guidance
- Based on NEET toppers' strategies

### 👩‍🏫 Teacher Agent

- Explains complex concepts in Physics, Chemistry, Biology
- NCERT syllabus aligned
- Multi-language support (Tamil, Hindi, Bengali, Telugu, Marathi)

### 📝 Trainer Agent

- Generates custom NEET-format quizzes
- Based on last 3 years' official papers
- Adaptive difficulty levels

### ✍️ Doubt Solver Agent

- Step-by-step problem solutions
- Quick and precise answers
- MCQ solving strategies

---

## 🛠️ Troubleshooting

### Issue: "Model requires more system memory"

**Solution:** Switch to a smaller model in `.env`:

```env
LLM_MODEL="phi3:mini"
# or try other available models with: ollama list
```

### Issue: "Could not connect to Ollama LLM"

**Solutions:**

1. Ensure Ollama is running: `ollama serve`
2. Check if models are installed: `ollama list`
3. Verify model names match `.env` configuration

### Issue: "streamlit command not found"

**Solutions:**

1. **Windows:** Use full path as shown in Step 5
2. **macOS/Linux:** Add to PATH or use: `python -m streamlit run app.py`

### Issue: Permission errors during installation

**Solution:** Use `--user` flag: `pip install --user -r requirements.txt`

### Issue: Import errors

**Solution:** Ensure all dependencies are installed:

```bash
pip install --user --upgrade -r requirements.txt
```

---

## 🔄 Daily Usage Commands

### Start ANEETA:

```bash
# Navigate to project folder
cd /path/to/ANEETA

# Run the app
streamlit run app.py
```

### Stop ANEETA:

- Press `Ctrl+C` in the terminal
- Or close the terminal window

### Check Ollama models:

```bash
ollama list
```

### Update models (if needed):

```bash
ollama pull phi4-mini
ollama pull nomic-embed-text
```

### Check model storage usage:

```bash
# List all models with sizes
ollama list

# Check total Ollama storage usage
# Windows: C:\Users\YourUsername\.ollama
# macOS/Linux: ~/.ollama
```

### Clean up unused models (if space is needed):

```bash
# Remove a specific model
ollama rm model-name

# Example: Remove large model to free space
ollama rm gemma2
```

---

## 🎯 Key Features

✅ **Offline Operation** - Works without internet once set up  
✅ **Multi-language Support** - Indian regional languages  
✅ **NCERT-aligned Content** - Based on official curriculum  
✅ **Multi-agent Architecture** - Specialized AI assistants  
✅ **NEET Focused** - Tailored for medical entrance prep  
✅ **Memory Efficient** - Optimized for regular laptops

---

## 📞 Getting Help

If you encounter issues:

1. **Check the troubleshooting section above**
2. **Verify system requirements**
3. **Ensure all steps were followed in order**
4. **Contact the team** for additional support

---

## 🎓 Ready to Learn!

Once set up, ANEETA will help democratize NEET preparation by providing:

- **Personalized study guidance**
- **Interactive learning sessions**
- **Practice tests and quizzes**
- **Doubt resolution support**

**Happy studying with ANEETA! 🩺✨**

---

_Last updated: October 2025_
_Version: 1.0_
