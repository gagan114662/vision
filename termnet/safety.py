import re
from typing import Tuple
from urllib.parse import urlparse

class SafetyChecker:
    DANGEROUS_PATTERNS = [
        r'rm\s+-rf\s+/?(\s|$)',
        r'dd\s+if=.*\s+of=/dev/',
        r'mkfs\.',
        r'\bshutdown\b',
        r'\breboot\b',
        r'\bhalt\b',
        r'\bpoweroff\b',
        r'\binit\s+[06]\b',
        r':\(\)\{.*\}',
        r':\(\)\{.*\|\s*:\s*&\s*\}',  # fork bomb
        r'>.*\s*/dev/',  # redirect to device
        r'\s+;\s*rm\s+',  # chained rm commands
        r'`.*`',  # backticks
        r'\$\(',  # command substitution
        r'sudo\s+'  # sudo commands
    ]
    RISKY_COMMANDS = {
        'rm','rmdir','chmod','chown','mv','dd','fdisk',
        'mkfs','format','del','rd','attrib','cacls'
    }

    def __init__(self):
        """Initialize SafetyChecker with compiled dangerous patterns"""
        self.dangerous_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.DANGEROUS_PATTERNS]

    @classmethod
    def is_safe(cls, command: str) -> Tuple[bool, str]:
        """Legacy class method for backward compatibility"""
        if not command or not command.strip():
            return True, ""
        low = command.lower().strip()
        for pat in cls.DANGEROUS_PATTERNS:
            if re.search(pat, low):
                return False, "Dangerous pattern detected."
        head = low.split()[0] if low.split() else ""
        if head in cls.RISKY_COMMANDS:
            return True, "⚠️ Caution: risky command."
        return True, ""

    def is_safe_command(self, command: str) -> Tuple[bool, str]:
        """Check if a command is safe to execute"""
        if not command or not command.strip():
            return True, ""

        command = command.strip()
        command_lower = command.lower()

        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if pattern.search(command_lower):
                return False, f"Dangerous pattern detected: {pattern.pattern}"

        # Check for sudo
        if command_lower.startswith('sudo '):
            return False, "Sudo commands are not allowed"

        # Check for pipes and redirects that might be dangerous
        if '>|' in command or '2>&1' in command:
            if '/dev/' in command:
                return False, "Dangerous redirect to device file"

        # Check command word
        parts = command_lower.split()
        if parts:
            cmd = parts[0]
            if cmd in self.RISKY_COMMANDS:
                return True, f"Caution: {cmd} is a risky command"

        return True, ""

    def check_file_path(self, path: str) -> Tuple[bool, str]:
        """Check if a file path is safe to access"""
        if not path:
            return True, ""

        path = path.strip()
        path_lower = path.lower()

        # Check for dangerous paths
        dangerous_paths = [
            '/etc/passwd', '/etc/shadow', '/boot/', '/sys/',
            '/proc/sys/', '/dev/', '/.ssh/', '/root/', '/var/log/',
            'c:\\windows\\system32', 'c:\\windows\\boot'
        ]

        for dangerous in dangerous_paths:
            if dangerous in path_lower:
                return False, f"Access to {dangerous} is not allowed"

        # Check for path traversal
        if '..' in path or path.startswith('/'):
            return False, "Path traversal or absolute paths not allowed"

        return True, ""

    def is_safe_url(self, url: str) -> Tuple[bool, str]:
        """Check if a URL is safe to access"""
        if not url:
            return True, ""

        try:
            parsed = urlparse(url)

            # Must have a scheme
            if not parsed.scheme:
                return False, "URL must have a scheme (http/https)"

            # Only allow http/https
            if parsed.scheme not in ('http', 'https'):
                return False, "Only HTTP/HTTPS URLs are allowed"

            # Must have a hostname
            if not parsed.netloc:
                return False, "URL must have a hostname"

            # Check for localhost/private IPs (basic check)
            hostname = parsed.netloc.split(':')[0].lower()
            if hostname in ('localhost', '127.0.0.1', '0.0.0.0'):
                return False, "Localhost URLs are not allowed"

            return True, ""

        except Exception as e:
            return False, f"Invalid URL format: {e}"

    def sanitize_output(self, text: str) -> str:
        """Sanitize command output by removing sensitive information"""
        if not text:
            return text

        # Patterns for sensitive data
        sensitive_patterns = [
            (r'password\s*[:=]\s*\S+', 'password: [REDACTED]'),
            (r'token\s*[:=]\s*\S+', 'token: [REDACTED]'),
            (r'key\s*[:=]\s*\S+', 'key: [REDACTED]'),
            (r'secret\s*[:=]\s*\S+', 'secret: [REDACTED]'),
            (r'\b[A-Za-z0-9+/]{20,}\b', '[REDACTED_TOKEN]'),  # Base64-like tokens
            (r'ssh-rsa\s+\S+', 'ssh-rsa [REDACTED]'),  # SSH keys
            (r'BEGIN\s+(RSA\s+)?PRIVATE\s+KEY', 'BEGIN PRIVATE KEY [REDACTED]')
        ]

        sanitized = text
        for pattern, replacement in sensitive_patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        return sanitized
