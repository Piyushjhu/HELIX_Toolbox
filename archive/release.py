#!/usr/bin/env python3
"""
Release script for HELIX Toolbox
Helps prepare and tag releases for GitHub
"""

import os
import sys
import subprocess
import re
from datetime import datetime

def run_command(command, check=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"Error running command: {command}")
            print(f"Error: {result.stderr}")
            sys.exit(1)
        return result
    except Exception as e:
        print(f"Exception running command: {command}")
        print(f"Exception: {e}")
        sys.exit(1)

def check_git_status():
    """Check if git repository is clean"""
    result = run_command("git status --porcelain", check=False)
    if result.stdout.strip():
        print("Warning: Git repository has uncommitted changes:")
        print(result.stdout)
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(0)
    else:
        print("✓ Git repository is clean")

def get_current_version():
    """Get current version from setup.py"""
    with open("setup.py", "r") as f:
        content = f.read()
    match = re.search(r'version="([^"]+)"', content)
    if match:
        return match.group(1)
    else:
        print("Error: Could not find version in setup.py")
        sys.exit(1)

def update_version(new_version):
    """Update version in setup.py"""
    with open("setup.py", "r") as f:
        content = f.read()
    
    content = re.sub(r'version="[^"]+"', f'version="{new_version}"', content)
    
    with open("setup.py", "w") as f:
        f.write(content)
    
    print(f"✓ Updated version to {new_version}")

def create_changelog():
    """Create a changelog entry"""
    print("\nCreating changelog entry...")
    
    # Get recent commits
    result = run_command("git log --oneline -10")
    commits = result.stdout.strip().split('\n')
    
    print("Recent commits:")
    for i, commit in enumerate(commits, 1):
        print(f"{i:2d}. {commit}")
    
    print("\nEnter the changelog entry (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    
    changelog = '\n'.join(lines[:-1])  # Remove the last empty line
    
    # Write to CHANGELOG.md
    with open("CHANGELOG.md", "a") as f:
        f.write(f"\n## [{new_version}] - {datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write(changelog)
        f.write("\n")
    
    print("✓ Added changelog entry")

def create_release():
    """Create a new release"""
    print("HELIX Toolbox Release Script")
    print("=" * 40)
    
    # Check git status
    check_git_status()
    
    # Get current version
    current_version = get_current_version()
    print(f"Current version: {current_version}")
    
    # Get new version
    new_version = input(f"Enter new version (current: {current_version}): ").strip()
    if not new_version:
        new_version = current_version
    
    # Update version
    update_version(new_version)
    
    # Create changelog
    create_changelog()
    
    # Commit changes
    print("\nCommitting changes...")
    run_command("git add setup.py CHANGELOG.md")
    run_command(f'git commit -m "Release version {new_version}"')
    
    # Create tag
    print(f"\nCreating tag v{new_version}...")
    run_command(f'git tag -a v{new_version} -m "Release version {new_version}"')
    
    # Push changes
    print("\nPushing to GitHub...")
    run_command("git push origin main")
    run_command(f"git push origin v{new_version}")
    
    print(f"\n✓ Release {new_version} created successfully!")
    print(f"GitHub release URL: https://github.com/Piyushjhu/HELIX_Toolbox/releases/tag/v{new_version}")

if __name__ == "__main__":
    create_release() 