#!/usr/bin/env python3
"""
GitHub Setup Script for Peak Finder ML Project

This script helps you set up the Git repository and provides 
instructions for uploading to GitHub.
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ {description} - Failed")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {str(e)}")
        return False
    return True

def check_git_installed():
    """Check if Git is installed"""
    try:
        result = subprocess.run("git --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Git is installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Git is not installed or not accessible")
            return False
    except:
        print("❌ Git is not installed or not accessible")
        return False

def setup_git_repository():
    """Initialize Git repository and add files"""
    print("\n" + "="*70)
    print("🚀 SETTING UP GIT REPOSITORY")
    print("="*70)
    
    if not check_git_installed():
        print("\n❌ Please install Git first:")
        print("   https://git-scm.com/downloads")
        return False
    
    # Initialize git repository
    if not run_command("git init", "Initializing Git repository"):
        return False
    
    # Add all files
    if not run_command("git add .", "Adding all files to Git"):
        return False
    
    # Make initial commit
    commit_message = "Initial commit: Advanced Peak Finder ML solution with ensemble models and multi-strategy optimization"
    if not run_command(f'git commit -m "{commit_message}"', "Making initial commit"):
        return False
    
    print("\n✅ Git repository setup complete!")
    return True

def print_github_instructions():
    """Print instructions for creating GitHub repository"""
    print("\n" + "="*70)
    print("📚 GITHUB UPLOAD INSTRUCTIONS")
    print("="*70)
    
    print("\n🎯 STEP 1: Create Repository on GitHub")
    print("-" * 50)
    print("1. Go to: https://github.com/new")
    print("2. Repository name: peak-finder-ml")
    print("3. Description: Advanced ML solution for peak finding in unknown terrains")
    print("4. Keep it PUBLIC (to showcase your work)")
    print("5. DON'T initialize with README (we already have one)")
    print("6. Click 'Create repository'")
    
    print("\n🔗 STEP 2: Link Local Repository to GitHub")
    print("-" * 50)
    print("After creating the GitHub repo, run these commands:")
    print()
    print("git remote add origin https://github.com/YOURUSERNAME/peak-finder-ml.git")
    print("git branch -M main")
    print("git push -u origin main")
    print()
    print("Replace 'YOURUSERNAME' with your actual GitHub username!")
    
    print("\n⚡ ALTERNATIVE: Use GitHub CLI (if you have it installed)")
    print("-" * 50)
    print("gh repo create peak-finder-ml --public --description \"Advanced ML solution for peak finding in unknown terrains\"")
    print("git remote add origin https://github.com/YOURUSERNAME/peak-finder-ml.git")
    print("git push -u origin main")
    
    print("\n🎨 STEP 3: Customize Your Repository")
    print("-" * 50)
    print("1. Add topics/tags: machine-learning, optimization, ensemble, python, scikit-learn")
    print("2. Enable Issues (for feedback)")
    print("3. Add a nice repository description")
    print("4. Consider adding a repository avatar/image")
    
    print("\n📊 REPOSITORY FEATURES")
    print("-" * 50)
    features = [
        "✅ Professional README with badges",
        "✅ Comprehensive documentation",
        "✅ MIT License included", 
        "✅ Proper .gitignore for Python",
        "✅ Complete test suite",
        "✅ Mathematical foundation explained",
        "✅ Visualization examples",
        "✅ Production-ready code"
    ]
    
    for feature in features:
        print(f"   {feature}")

def print_repository_suggestions():
    """Print suggestions for repository names"""
    print("\n" + "="*70)
    print("📝 REPOSITORY NAME SUGGESTIONS")
    print("="*70)
    
    suggestions = [
        ("peak-finder-ml", "⭐ RECOMMENDED - Clear, professional, indicates ML approach"),
        ("terrain-peak-optimizer", "Emphasizes optimization aspect"),
        ("ensemble-peak-detection", "Highlights ensemble ML approach"),
        ("intelligent-summit-finder", "Creative but professional"),
        ("bluestone-peak-challenge", "References the specific challenge"),
        ("summit-seeker-ai", "AI-focused branding"),
        ("peak-pursuit-ensemble", "Alliterative and technical"),
        ("elevation-optimizer-ml", "Descriptive of the task"),
        ("terrain-navigator-ai", "Broader appeal"),
        ("gradient-peak-finder", "Technical but accessible")
    ]
    
    print("\n🏆 Top Recommendations:")
    for i, (name, desc) in enumerate(suggestions[:3], 1):
        print(f"{i}. {name:<25} - {desc}")
    
    print("\n💡 Alternative Options:")
    for i, (name, desc) in enumerate(suggestions[3:], 4):
        print(f"{i}. {name:<25} - {desc}")

def main():
    print("🏔️  PEAK FINDER ML - GITHUB SETUP ASSISTANT")
    print("="*70)
    
    print("\n📁 Current directory:", os.getcwd())
    print("📊 Files in project:")
    
    # List important files
    important_files = [
        "submission.py",
        "README.md", 
        "requirements.txt",
        "LICENSE",
        ".gitignore"
    ]
    
    for file in important_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - Missing!")
    
    print("\n" + "="*70)
    
    # Show repository name suggestions
    print_repository_suggestions()
    
    # Ask if user wants to proceed with Git setup
    print("\n" + "="*70)
    response = input("\n🤔 Do you want to initialize Git repository now? (y/n): ").strip().lower()
    
    if response in ['y', 'yes']:
        if setup_git_repository():
            print_github_instructions()
        else:
            print("\n❌ Git setup failed. Please check the errors above.")
    else:
        print("\n📋 Skipping Git setup. You can run this script again later.")
        print_github_instructions()
    
    print("\n" + "="*70)
    print("🎉 READY TO SHOWCASE YOUR WORK!")
    print("="*70)
    print("\nYour Peak Finder ML project is ready for GitHub!")
    print("This advanced solution demonstrates:")
    print("  ✅ Ensemble Machine Learning")
    print("  ✅ Multi-Strategy Optimization") 
    print("  ✅ Production-Quality Code")
    print("  ✅ Comprehensive Documentation")
    print("\nTime to show the world your ML engineering skills! 🚀")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()