# 🤖 DCGAN AI Forge – Face Generator API

Welcome to **DCGAN AI Forge** – a powerful and easy-to-integrate FastAPI-based REST API that uses a trained **Deep Convolutional GAN (DCGAN)** to generate synthetic human faces.

> You can also **use it as an npm package** to directly access the face generation feature in your frontend projects.

### 🔗 Live Demo  
🚀 https://dark-face-forge.vercel.app/

### 💻 GitHub  
📁 [https://github.com/awais7012/ai-r1](https://github.com/awais7012/ai-r1)

---

## 🎯 Features

- ✅ Pre-trained **Generator** and **Discriminator** models included
- ✅ RESTful FastAPI backend
- ✅ Image output via static URL
- ✅ Easy integration into frontend via `npm`
- ✅ CORS support for dev usage

---

## 🛠️ Installation & Setup

### 🔌 Backend (FastAPI)

```bash
git clone https://github.com/awais7012/ai-r1.git
cd ai-r1
pip install -r requirements.txt
uvicorn main:app --reload
