"""
============================================================
  Project : Engineering College Recommendation System
  Name    : Nishal KV
  Roll No : 67
============================================================

package_project.py
Automates the creation of the submission tarball.
Name format: <Roll-No_Name_Project_No.tar>
"""

import os
import tarfile
import shutil

# --- Configuration ---
ROLL_NO = "67"
NAME = "Nishal_KV"
PROJECT_NO = "College_Recommendation"
TAR_NAME = f"{ROLL_NO}_{NAME}_{PROJECT_NO}.tar"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = "submission_payload"

FILES_TO_INCLUDE = [
    "main.py",
    "recommendation_engine.py",
    "deep_learning_model.py",
    "generate_dataset.py",
    "visualizations.py",
    "requirements.txt",
    "college_dataset.csv",
    "README.md",
    ".gitignore"
]

DIRS_TO_INCLUDE = [
    "docs",
    "output"
]

def main():
    print(f"📦 Starting packaging for {TAR_NAME}...")
    
    # 1. Create temporary submission folder
    if os.path.exists(SUBMISSION_DIR):
        shutil.rmtree(SUBMISSION_DIR)
    os.makedirs(os.path.join(SUBMISSION_DIR, "code"))
    
    # 2. Copy files into 'code' folder
    for f in FILES_TO_INCLUDE:
        if os.path.exists(os.path.join(PROJECT_ROOT, f)):
            shutil.copy(os.path.join(PROJECT_ROOT, f), os.path.join(SUBMISSION_DIR, "code", f))
            
    for d in DIRS_TO_INCLUDE:
        if os.path.exists(os.path.join(PROJECT_ROOT, d)):
            shutil.copytree(os.path.join(PROJECT_ROOT, d), os.path.join(SUBMISSION_DIR, d))
            
    # 3. Handle 'screenshots' folder (ensure it exists)
    if not os.path.exists(os.path.join(SUBMISSION_DIR, "screenshots")):
        os.makedirs(os.path.join(SUBMISSION_DIR, "screenshots"))
        print("   ⚠️  Note: 'screenshots' folder created but is empty. Please add your screenshots there.")

    # 4. Create the tar ball
    with tarfile.open(TAR_NAME, "w") as tar:
        tar.add(SUBMISSION_DIR, arcname=os.path.sep)
        
    # 5. Cleanup
    shutil.rmtree(SUBMISSION_DIR)
    
    print(f"\n✅ Success! Submission file created: {TAR_NAME}")
    print(f"   Path: {os.path.join(PROJECT_ROOT, TAR_NAME)}")

if __name__ == "__main__":
    main()
