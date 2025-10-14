# Contributing to YMERA Enterprise Platform

Thank you for your interest in contributing to YMERA! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please:
1. Search existing issues to avoid duplicates
2. Use the issue templates when available
3. Provide clear, detailed information
4. Include steps to reproduce (for bugs)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ymera.git
   cd ymera
   ```

2. **Set Up Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set Up Node.js Environment**
   ```bash
   npm install
   ```

4. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Set Up Database**
   ```bash
   # Start PostgreSQL and Redis (using Docker)
   docker-compose up -d postgres redis
   
   # Run migrations
   alembic upgrade head
   ```

### Code Style

#### Python
- Follow PEP 8 style guide
- Use type hints
- Maximum line length: 100 characters
- Use `black` for formatting:
  ```bash
  black .
  ```
- Use `flake8` for linting:
  ```bash
  flake8 .
  ```
- Use `mypy` for type checking:
  ```bash
  mypy .
  ```

#### TypeScript/JavaScript
- Follow ESLint configuration
- Use TypeScript for all new code
- Use meaningful variable names
- Format with Prettier:
  ```bash
  npm run format
  ```

### Testing

#### Python Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test
pytest tests/test_specific.py
```

#### TypeScript Tests
```bash
# Type checking
npm run type-check

# Linting
npm run lint
```

### Commit Messages

Use clear, descriptive commit messages:

- **Format**: `type(scope): description`
- **Types**:
  - `feat`: New feature
  - `fix`: Bug fix
  - `docs`: Documentation changes
  - `style`: Code style changes (formatting)
  - `refactor`: Code refactoring
  - `test`: Adding or updating tests
  - `chore`: Maintenance tasks

**Examples**:
```
feat(auth): add OAuth2 authentication
fix(api): resolve timeout in file upload
docs(readme): update installation instructions
```

### Pull Request Process

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clean, tested code
   - Follow code style guidelines
   - Update documentation as needed
   - Add tests for new features

3. **Test Your Changes**
   ```bash
   # Run all tests
   pytest
   npm run type-check
   
   # Check code style
   black . --check
   npm run lint
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat(scope): your message"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Use a clear, descriptive title
   - Reference related issues
   - Describe what changed and why
   - Include screenshots for UI changes

### Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New code has tests
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No sensitive data in commits
- [ ] Branch is up to date with main

### Code Review

- Be respectful and constructive
- Respond to feedback promptly
- Make requested changes
- Ask questions if unclear

### What to Contribute

Good areas for contribution:
- 🐛 Bug fixes
- 📝 Documentation improvements
- ✨ New features (discuss first in an issue)
- 🧪 Additional tests
- 🎨 UI/UX improvements
- ♿ Accessibility enhancements
- 🌐 Internationalization

### Questions?

- Open a discussion on GitHub
- Check existing documentation
- Review closed issues for similar questions

## 📜 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the project

## 🎉 Recognition

Contributors will be acknowledged in:
- Release notes
- Contributors page
- Project documentation

Thank you for contributing to YMERA! 🚀
