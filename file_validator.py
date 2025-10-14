"""
YMERA Enterprise - File Validator Utility
Production-ready file validation with security checks
"""

import magic
import mimetypes
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import hashlib

@dataclass
class ValidationResult:
    """File validation result"""
    is_valid: bool
    mime_type: Optional[str] = None
    file_extension: Optional[str] = None
    file_size: int = 0
    error: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class FileValidator:
    """Comprehensive file validation"""
    
    # Dangerous file extensions
    DANGEROUS_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
        '.jar', '.msi', '.app', '.deb', '.rpm', '.dmg', '.pkg', '.sh',
        '.ps1', '.psm1', '.psd1', '.ps1xml', '.psc1', '.bat', '.cmd'
    }
    
    # Image bomb limits
    MAX_IMAGE_PIXELS = 89478485  # ~8K resolution
    
    def __init__(
        self,
        max_size: int = 100 * 1024 * 1024,  # 100MB
        allowed_types: Optional[List[str]] = None,
        blocked_types: Optional[List[str]] = None
    ):
        self.max_size = max_size
        self.allowed_types = set(allowed_types) if allowed_types else None
        self.blocked_types = set(blocked_types) if blocked_types else set()
        self.magic_detector = magic.Magic(mime=True)
    
    async def validate(
        self,
        content: bytes,
        filename: str,
        mime_type: Optional[str] = None
    ) -> ValidationResult:
        """
        Comprehensive file validation.
        
        Args:
            content: File content bytes
            filename: Original filename
            mime_type: Declared MIME type (optional)
            
        Returns:
            ValidationResult with validation details
        """
        warnings = []
        
        # Size validation
        file_size = len(content)
        if file_size > self.max_size:
            return ValidationResult(
                is_valid=False,
                file_size=file_size,
                error=f"File too large: {file_size} bytes (max: {self.max_size})"
            )
        
        if file_size == 0:
            return ValidationResult(
                is_valid=False,
                file_size=0,
                error="Empty file"
            )
        
        # Extension validation
        file_extension = Path(filename).suffix.lower()
        if file_extension in self.DANGEROUS_EXTENSIONS:
            return ValidationResult(
                is_valid=False,
                file_extension=file_extension,
                file_size=file_size,
                error=f"Dangerous file extension: {file_extension}"
            )
        
        # MIME type detection
        detected_mime = self.magic_detector.from_buffer(content)
        
        # MIME type validation
        if self.blocked_types and detected_mime in self.blocked_types:
            return ValidationResult(
                is_valid=False,
                mime_type=detected_mime,
                file_extension=file_extension,
                file_size=file_size,
                error=f"Blocked MIME type: {detected_mime}"
            )
        
        if self.allowed_types and detected_mime not in self.allowed_types:
            return ValidationResult(
                is_valid=False,
                mime_type=detected_mime,
                file_extension=file_extension,
                file_size=file_size,
                error=f"MIME type not allowed: {detected_mime}"
            )
        
        # MIME type mismatch warning
        if mime_type and mime_type != detected_mime:
            warnings.append(
                f"MIME type mismatch: declared '{mime_type}', detected '{detected_mime}'"
            )
        
        # Image-specific validation
        if detected_mime.startswith('image/'):
            image_validation = self._validate_image(content)
            if not image_validation['is_valid']:
                return ValidationResult(
                    is_valid=False,
                    mime_type=detected_mime,
                    file_extension=file_extension,
                    file_size=file_size,
                    error=image_validation['error']
                )
            warnings.extend(image_validation.get('warnings', []))
        
        # Archive-specific validation
        if detected_mime in ['application/zip', 'application/x-rar-compressed', 'application/x-tar']:
            archive_validation = self._validate_archive(content, detected_mime)
            if not archive_validation['is_valid']:
                return ValidationResult(
                    is_valid=False,
                    mime_type=detected_mime,
                    file_extension=file_extension,
                    file_size=file_size,
                    error=archive_validation['error']
                )
            warnings.extend(archive_validation.get('warnings', []))
        
        return ValidationResult(
            is_valid=True,
            mime_type=detected_mime,
            file_extension=file_extension,
            file_size=file_size,
            warnings=warnings
        )
    
    def _validate_image(self, content: bytes) -> Dict[str, Any]:
        """Validate image files for bombs and malicious content"""
        try:
            from PIL import Image
            import io
            
            image = Image.open(io.BytesIO(content))
            width, height = image.size
            pixels = width * height
            
            if pixels > self.MAX_IMAGE_PIXELS:
                return {
                    'is_valid': False,
                    'error': f'Image too large: {pixels} pixels (max: {self.MAX_IMAGE_PIXELS})'
                }
            
            warnings = []
            
            # Check for excessive dimensions
            if width > 10000 or height > 10000:
                warnings.append(f'Large image dimensions: {width}x{height}')
            
            # Check for decompression bombs
            if pixels > 25000000:  # 25MP
                warnings.append('Large image file - potential decompression bomb')
            
            return {
                'is_valid': True,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': f'Image validation failed: {str(e)}'
            }
    
    def _validate_archive(self, content: bytes, mime_type: str) -> Dict[str, Any]:
        """Validate archive files for zip bombs and dangerous content"""
        try:
            import zipfile
            import io
            
            warnings = []
            
            if mime_type == 'application/zip':
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    # Check for zip bombs
                    total_uncompressed = sum(info.file_size for info in zf.filelist)
                    compression_ratio = total_uncompressed / len(content) if len(content) > 0 else 0
                    
                    if compression_ratio > 100:
                        return {
                            'is_valid': False,
                            'error': f'Potential zip bomb detected: {compression_ratio:.1f}x compression ratio'
                        }
                    
                    if compression_ratio > 50:
                        warnings.append(f'High compression ratio: {compression_ratio:.1f}x')
                    
                    # Check for dangerous files in archive
                    for info in zf.filelist:
                        file_ext = Path(info.filename).suffix.lower()
                        if file_ext in self.DANGEROUS_EXTENSIONS:
                            return {
                                'is_valid': False,
                                'error': f'Dangerous file in archive: {info.filename}'
                            }
            
            return {
                'is_valid': True,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': f'Archive validation failed: {str(e)}'
            }
    
    def calculate_checksum(self, content: bytes, algorithm: str = 'sha256') -> str:
        """Calculate file checksum"""
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(content)
        return hash_obj.hexdigest()

__all__ = ['FileValidator', 'ValidationResult']
