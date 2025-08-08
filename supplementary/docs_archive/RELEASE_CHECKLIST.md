# Release Checklist

## Pre-Release Tasks

### ✅ Code Quality
- [ ] All tests pass (`python test_helix.py`)
- [ ] No linter errors
- [ ] Code is properly documented
- [ ] GUI functionality verified

### ✅ Documentation
- [ ] README.md is complete and accurate
- [ ] Installation guide is up to date
- [ ] User guide covers all features
- [ ] CHANGELOG.md is updated
- [ ] Credits and acknowledgments are correct

### ✅ Files and Structure
- [ ] All required files exist:
  - [ ] `alpss_spade_gui.py` (main GUI)
  - [ ] `README.md` (project description)
  - [ ] `requirements.txt` (dependencies)
  - [ ] `LICENSE` (MIT license)
  - [ ] `setup.py` (package configuration)
  - [ ] `CHANGELOG.md` (version history)
  - [ ] `.gitignore` (version control exclusions)
- [ ] Directory structure is correct:
  - [ ] `ALPSS/` (ALPSS package)
  - [ ] `SPADE/` (SPADE package)
  - [ ] `docs/` (documentation)
  - [ ] `.github/workflows/` (CI/CD)

### ✅ Credits and Attribution
- [ ] Author information: Piyush Wanchoo (@Piyushjhu)
- [ ] ALPSS credits: Jake Diamond (@Jake-Diamond-9)
- [ ] SPADE credits: Piyush Wanchoo (@Piyushjhu)
- [ ] Institution: Johns Hopkins University
- [ ] Year: 2025

## Release Process

### 1. Update Version
```bash
# Edit setup.py to update version number
# Update CHANGELOG.md with release date
```

### 2. Test Everything
```bash
# Run comprehensive tests
python test_helix.py

# Test GUI launch
python alpss_spade_gui.py
```

### 3. Commit and Tag
```bash
# Add all changes
git add .

# Commit with descriptive message
git commit -m "Release version X.X.X"

# Create annotated tag
git tag -a vX.X.X -m "Release version X.X.X"

# Push to GitHub
git push origin main
git push origin vX.X.X
```

### 4. Create GitHub Release
1. Go to GitHub repository
2. Click "Releases"
3. Click "Create a new release"
4. Select the tag you just created
5. Add release title: "HELIX Toolbox vX.X.X"
6. Add release description from CHANGELOG.md
7. Mark as latest release
8. Publish release

## Post-Release Tasks

### ✅ Verification
- [ ] Release is visible on GitHub
- [ ] Download links work
- [ ] Installation instructions are correct
- [ ] Documentation is accessible

### ✅ Communication
- [ ] Update any related documentation
- [ ] Notify users if applicable
- [ ] Update any external references

## Version History

### v1.0.0 (Initial Release)
- Complete ALPSS and SPADE integration
- Comprehensive GUI with all analysis modes
- Optional Gaussian notch filter
- Complete uncertainty propagation
- Batch processing capabilities
- Cross-platform compatibility
- Comprehensive documentation

## Notes

- Always test on multiple platforms if possible
- Ensure all dependencies are properly specified
- Keep documentation synchronized with code changes
- Maintain proper attribution and credits
- Follow semantic versioning guidelines 