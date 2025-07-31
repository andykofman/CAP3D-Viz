# Security Policy

## Supported Versions

We are committed to providing security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in cap3d_view, please follow these steps:

### 1. **DO NOT** create a public GitHub issue

Security vulnerabilities should be reported privately to prevent potential exploitation.

### 2. Report the vulnerability

Send an email to : **ali.a@aucegypt.edu**

**Include the following information:**

- **Description**: A clear description of the vulnerability
- **Steps to reproduce**: Detailed steps to reproduce the issue
- **Impact**: Potential impact of the vulnerability
- **Affected versions**: Which versions are affected
- **Suggested fix**: If you have a suggested fix (optional)
- **Proof of concept**: If applicable, include a proof of concept
- **Your contact information**: So we can reach you for additional details

### 3. What happens next?

1. **Acknowledgment**: You will receive an acknowledgment within 48 hours
2. **Investigation**: Our security team will investigate the report
3. **Updates**: We will keep you updated on the progress
4. **Resolution**: Once fixed, we will:
   - Credit you in the security advisory (if you wish)
   - Release a patch
   - Update the documentation

## Security Best Practices

### For Users

- Always use the latest stable version
- Keep your dependencies updated
- Run security scans on your code
- Follow secure coding practices
- Use virtual environments to isolate dependencies

### For Contributors

- Follow secure coding guidelines
- Never commit sensitive information (API keys, passwords, etc.)
- Use environment variables for configuration
- Validate all user inputs
- Implement proper error handling
- Use HTTPS for all external communications

## Security Features

cap3d_view includes several security features:

- **Input validation**: All user inputs are validated
- **Safe file handling**: Secure file operations
- **Memory management**: Proper memory handling for large datasets
- **Error handling**: Comprehensive error handling without information leakage

## Known Security Issues

### None Currently Known

If you find a security issue, please report it using the process above.


## Responsible Disclosure

We follow responsible disclosure practices:

- **Timeline**: We aim to fix critical vulnerabilities within 30 days
- **Communication**: We will communicate progress and timelines
- **Credit**: Contributors will be credited in security advisories (if they wish)
- **Coordination**: We will coordinate with other projects if needed

## Security Contacts

- **Team**: ali.a@aucegypt.edu
- **Maintainers**: See [CONTRIBUTING.md](CONTRIBUTING.md) for maintainer contacts
- **GitHub Security**: Use GitHub's security features for repository-specific issues

## Security Checklist

Before submitting code:

- [ ] No hardcoded secrets or credentials
- [ ] Input validation implemented
- [ ] Error handling without information leakage
- [ ] Dependencies are up to date
- [ ] Security tests included (if applicable)
- [ ] Documentation updated for security-related changes

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python-security.readthedocs.io/)
- [GitHub Security Features](https://docs.github.com/en/github/managing-security-vulnerabilities)

---

Thank you for helping keep cap3d_view secure! 🔒 