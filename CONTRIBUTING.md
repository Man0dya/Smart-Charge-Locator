# Contributing to Smart Charge Locator

Thanks for your interest in contributing! This document explains how to get started and our basic workflows.

## Getting started

1. Fork the repo and clone your fork.
2. Create a virtual environment and install dependencies:
   - Runtime: `pip install -r requirements.txt`
   - Notebooks/dev: `pip install -r requirements-dev.txt` (optional)
3. Ensure the runtime assets exist:
   - `data/processed/city_features_engineered.csv`
   - `data/processed/scaler.pkl`
   - `data/processed/feature_columns.pkl`
   - `models/xgboost.pkl`

## Development

- Create a feature branch: `git checkout -b feature/short-description`
- Run the app locally: `streamlit run streamlit_app.py`
- Add tests or notebook updates if you change data or model logic.

## Coding style

- Python 3.11+ recommended locally
- Follow PEP 8 style guidelines
- Keep `requirements.txt` minimal for runtime; use `requirements-dev.txt` for notebook tooling

## Commit & PR process

1. Commit with clear messages: `feat: add city filter`
2. Push your branch and open a Pull Request (PR)
3. Fill out the PR template (see .github/pull_request_template.md)
4. Ensure CI (if configured) passes
5. A maintainer will review and merge

## Reporting issues

- Use GitHub Issues. Please include:
  - Steps to reproduce
  - Expected behavior
  - Actual behavior and logs/screenshots
  - Environment (OS, Python version)

## Security

See SECURITY.md for reporting vulnerabilities. Please do not disclose security issues publicly before they are resolved.