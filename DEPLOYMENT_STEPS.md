# SAT Vocabulary System Deployment Guide

## Step-by-Step Deployment to Railway

### 1. Create GitHub Repository
1. Go to [GitHub](https://github.com)
2. Click "New Repository"
3. Name: `sat-vocab-ai-system`
4. Description: `Clean SAT vocabulary generator with feedback-driven learning`
5. Make it Public
6. Click "Create Repository"

### 2. Push Code to GitHub
```bash
# In your terminal, run these commands:
cd "i:\SAT Vocab\sat-vocab-rag"
git remote add origin https://github.com/YOUR_USERNAME/sat-vocab-ai-system.git
git branch -M main
git push -u origin main
```

### 3. Deploy to Railway
1. Go to [Railway](https://railway.app)
2. Sign up/login with GitHub
3. Click "New Project"
4. Click "Deploy from GitHub repo"
5. Select your `sat-vocab-ai-system` repository
6. Railway will automatically detect the Dockerfile

### 4. Set Environment Variables
1. In Railway dashboard, click on your project
2. Go to "Variables" tab
3. Add variable:
   - Key: `HACKCLUB_API_KEY`
   - Value: `your_api_key_here`

### 5. Get API Key (Free)
1. Go to [HackClub AI](https://ai.hackclub.com/)
2. Sign up for free account
3. Get your API key
4. Add it to Railway environment variables

### 6. Deploy!
- Railway will automatically build and deploy
- Your app will be available at: `https://your-app-name.railway.app`

## Alternative: Render Deployment

### 1. Go to Render
1. Visit [Render](https://render.com)
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repo

### 2. Configure
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn src.app:app --host 0.0.0.0 --port $PORT`
- Environment Variables: Add `HACKCLUB_API_KEY`

## Testing Deployment

Once deployed, test these features:
1. Generate a vocabulary word
2. Rate it and provide feedback
3. Try regenerating to see learning in action
4. Check that feedback is stored and used

Your app is now live and learning from user feedback!
