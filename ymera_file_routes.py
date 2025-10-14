"""
YMERA Enterprise - File Management Routes
Production-Ready File Upload/Download System - v4.0
Complete implementation with security, validation, and monitoring
"""

# ===============================================================================
# STANDARD IMPORTS
# ===============================================================================

import os
import uuid
import hashlib
import mimetypes
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, BinaryIO
import aiofiles
import magic
from PIL import Image

# Third-party imports
import structlog
from fastapi import (
    APIRouter, 
    HTTPException, 
    Depends, 
    UploadFile, 
    File, 
    Form,
    Query,
    status,
    BackgroundTasks
)
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func

# Local imports
from app.CORE_CONFIGURATION.config_settings import get_settings
from app.DATABASE_CORE.database_connection import get_db_session
from app.SECURITY.jwt_handler import get_current_user
from app.models.user import User
from app.models.file import FileMetadata, FileVersion, FileShare
from app.utils.file_validator import FileValidator
from app.utils.virus_scanner import VirusScanner

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================

logger = structlog.get_logger("ymera.file_routes")

# ===============================================================================
# CONFIGURATION
# ===============================================================================

settings = get_settings()

# File storage configuration
FILE_STORAGE_PATH = Path(settings.FILE_STORAGE_PATH)
TEMP_STORAGE_PATH = Path(settings.TEMP_STORAGE_PATH)
MAX_FILE_SIZE = settings.MAX_FILE_SIZE  # 100MB default
MAX_FILES_PER_USER = settings.MAX_FILES_PER_USER  # 1000 default
ALLOWED_MIME_TYPES = settings.ALLOWED_MIME_TYPES or [
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/plain",
    "text/csv",
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "video/mp4",
    "video/webm",
    "audio/mpeg",
    "audio/wav",
    "application/zip",
    "application/x-rar-compressed"
]

# Create storage directories
FILE_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
TEMP_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

# ===============================================================================
# DATA MODELS & SCHEMAS
# ===============================================================================

class FileUploadRequest(BaseModel):
    """File upload request schema"""
    project_id: Optional[str] = Field(None, description="Associated project ID")
    folder_path: Optional[str] = Field(None, description="Virtual folder path")
    tags: List[str] = Field(default_factory=list, description="File tags")
    description: Optional[str] = Field(None, max_length=1000)
    is_public: bool = Field(False, description="Public access flag")
    
    @validator('folder_path')
    def validate_folder_path(cls, v):
        if v and ('..' in v or v.startswith('/')):
            raise ValueError("Invalid folder path")
        return v
    
    @validator('tags')
    def validate_tags(cls, v):
        if len(v) > 10:
            raise ValueError("Maximum 10 tags allowed")
        return [tag.strip().lower() for tag in v if tag.strip()]

class FileUploadResponse(BaseModel):
    """File upload response schema"""
    file_id: str
    filename: str
    original_filename: str
    file_size: int
    mime_type: str
    checksum: str
    upload_timestamp: datetime
    storage_path: str
    url: str
    thumbnail_url: Optional[str] = None
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

class FileMetadataResponse(BaseModel):
    """File metadata response schema"""
    file_id: str
    filename: str
    original_filename: str
    file_size: int
    mime_type: str
    checksum: str
    upload_timestamp: datetime
    updated_timestamp: Optional[datetime]
    owner_id: str
    project_id: Optional[str]
    folder_path: Optional[str]
    tags: List[str]
    description: Optional[str]
    is_public: bool
    download_count: int
    version: int
    status: str
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

class FileListResponse(BaseModel):
    """File list response schema"""
    files: List[FileMetadataResponse]
    total_count: int
    total_size: int
    page: int
    page_size: int

class FileShareRequest(BaseModel):
    """File share request schema"""
    user_ids: List[str] = Field(..., description="User IDs to share with")
    permission: str = Field("read", description="Permission level: read, write, admin")
    expires_at: Optional[datetime] = Field(None, description="Expiration datetime")
    
    @validator('permission')
    def validate_permission(cls, v):
        if v not in ['read', 'write', 'admin']:
            raise ValueError("Invalid permission level")
        return v

# ===============================================================================
# FILE SERVICE CLASS
# ===============================================================================

class FileService:
    """Core file management service"""
    
    def __init__(self):
        self.logger = logger.bind(component="file_service")
        self.validator = FileValidator(
            max_size=MAX_FILE_SIZE,
            allowed_types=ALLOWED_MIME_TYPES
        )
        self.virus_scanner = VirusScanner() if settings.VIRUS_SCAN_ENABLED else None
    
    async def upload_file(
        self,
        file: UploadFile,
        user: User,
        db: AsyncSession,
        request_data: FileUploadRequest,
        background_tasks: BackgroundTasks
    ) -> FileUploadResponse:
        """
        Upload file with comprehensive validation and processing.
        
        Args:
            file: Uploaded file
            user: Current user
            db: Database session
            request_data: Upload request data
            background_tasks: Background task manager
            
        Returns:
            File upload response
        """
        try:
            # Check user file quota
            await self._check_user_quota(user.id, db)
            
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            
            # Read file content
            content = await file.read()
            file_size = len(content)
            
            # Validate file
            validation_result = await self.validator.validate(
                content=content,
                filename=file.filename,
                mime_type=file.content_type
            )
            
            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File validation failed: {validation_result.error}"
                )
            
            # Virus scan if enabled
            if self.virus_scanner:
                scan_result = await self.virus_scanner.scan(content)
                if not scan_result.is_clean:
                    self.logger.warning(
                        "Virus detected in file",
                        file_id=file_id,
                        user_id=user.id,
                        threat=scan_result.threat_name
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="File contains malicious content"
                    )
            
            # Calculate checksum
            checksum = hashlib.sha256(content).hexdigest()
            
            # Check for duplicate files
            existing_file = await self._check_duplicate(checksum, user.id, db)
            if existing_file:
                self.logger.info("Duplicate file detected", file_id=existing_file.file_id)
                return self._create_upload_response(existing_file)
            
            # Determine storage path
            storage_subdir = self._get_storage_subdir(user.id, request_data.project_id)
            storage_dir = FILE_STORAGE_PATH / storage_subdir
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate safe filename
            file_extension = Path(file.filename).suffix.lower()
            safe_filename = f"{file_id}{file_extension}"
            file_path = storage_dir / safe_filename
            
            # Write file to disk
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            # Create file metadata
            file_metadata = FileMetadata(
                file_id=file_id,
                filename=safe_filename,
                original_filename=file.filename,
                file_size=file_size,
                mime_type=validation_result.mime_type,
                checksum=checksum,
                storage_path=str(file_path.relative_to(FILE_STORAGE_PATH)),
                owner_id=user.id,
                project_id=request_data.project_id,
                folder_path=request_data.folder_path,
                tags=request_data.tags,
                description=request_data.description,
                is_public=request_data.is_public,
                upload_timestamp=datetime.utcnow(),
                status="active"
            )
            
            db.add(file_metadata)
            await db.commit()
            await db.refresh(file_metadata)
            
            # Generate thumbnail for images (background task)
            if validation_result.mime_type.startswith('image/'):
                background_tasks.add_task(
                    self._generate_thumbnail,
                    file_path,
                    file_id,
                    db
                )
            
            self.logger.info(
                "File uploaded successfully",
                file_id=file_id,
                user_id=user.id,
                size=file_size,
                mime_type=validation_result.mime_type
            )
            
            return self._create_upload_response(file_metadata)
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error("File upload failed", error=str(e), user_id=user.id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="File upload failed"
            )
    
    async def download_file(
        self,
        file_id: str,
        user: User,
        db: AsyncSession
    ) -> FileResponse:
        """
        Download file with access control.
        
        Args:
            file_id: File identifier
            user: Current user
            db: Database session
            
        Returns:
            File response
        """
        try:
            # Get file metadata
            query = select(FileMetadata).where(FileMetadata.file_id == file_id)
            result = await db.execute(query)
            file_metadata = result.scalar_one_or_none()
            
            if not file_metadata:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File not found"
                )
            
            # Check access permissions
            has_access = await self._check_access(file_metadata, user, db)
            if not has_access:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
            
            # Get file path
            file_path = FILE_STORAGE_PATH / file_metadata.storage_path
            
            if not file_path.exists():
                self.logger.error("File not found on disk", file_id=file_id)
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File not found on storage"
                )
            
            # Increment download count
            file_metadata.download_count += 1
            await db.commit()
            
            self.logger.info(
                "File downloaded",
                file_id=file_id,
                user_id=user.id
            )
            
            return FileResponse(
                path=file_path,
                filename=file_metadata.original_filename,
                media_type=file_metadata.mime_type,
                headers={
                    "X-File-ID": file_id,
                    "X-Checksum": file_metadata.checksum
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error("File download failed", error=str(e), file_id=file_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="File download failed"
            )
    
    async def list_files(
        self,
        user: User,
        db: AsyncSession,
        project_id: Optional[str] = None,
        folder_path: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ) -> FileListResponse:
        """List files with filtering and pagination"""
        try:
            # Build query
            query = select(FileMetadata).where(
                or_(
                    FileMetadata.owner_id == user.id,
                    FileMetadata.is_public == True
                )
            )
            
            if project_id:
                query = query.where(FileMetadata.project_id == project_id)
            
            if folder_path:
                query = query.where(FileMetadata.folder_path == folder_path)
            
            if tags:
                query = query.where(FileMetadata.tags.contains(tags))
            
            if search:
                search_pattern = f"%{search}%"
                query = query.where(
                    or_(
                        FileMetadata.original_filename.ilike(search_pattern),
                        FileMetadata.description.ilike(search_pattern)
                    )
                )
            
            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total_count_result = await db.execute(count_query)
            total_count = total_count_result.scalar()
            
            # Apply pagination
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)
            query = query.order_by(FileMetadata.upload_timestamp.desc())
            
            # Execute query
            result = await db.execute(query)
            files = result.scalars().all()
            
            # Calculate total size
            total_size = sum(f.file_size for f in files)
            
            return FileListResponse(
                files=[self._create_metadata_response(f) for f in files],
                total_count=total_count,
                total_size=total_size,
                page=page,
                page_size=page_size
            )
            
        except Exception as e:
            self.logger.error("File listing failed", error=str(e), user_id=user.id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list files"
            )
    
    async def delete_file(
        self,
        file_id: str,
        user: User,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Delete file with access control"""
        try:
            # Get file metadata
            query = select(FileMetadata).where(FileMetadata.file_id == file_id)
            result = await db.execute(query)
            file_metadata = result.scalar_one_or_none()
            
            if not file_metadata:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File not found"
                )
            
            # Check ownership
            if file_metadata.owner_id != user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only file owner can delete"
                )
            
            # Delete physical file
            file_path = FILE_STORAGE_PATH / file_metadata.storage_path
            if file_path.exists():
                file_path.unlink()
            
            # Delete thumbnail if exists
            thumbnail_path = self._get_thumbnail_path(file_id)
            if thumbnail_path.exists():
                thumbnail_path.unlink()
            
            # Soft delete in database
            file_metadata.status = "deleted"
            file_metadata.updated_timestamp = datetime.utcnow()
            await db.commit()
            
            self.logger.info("File deleted", file_id=file_id, user_id=user.id)
            
            return {
                "status": "success",
                "message": "File deleted successfully",
                "file_id": file_id
            }
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error("File deletion failed", error=str(e), file_id=file_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="File deletion failed"
            )
    
    # ========== HELPER METHODS ==========
    
    async def _check_user_quota(self, user_id: str, db: AsyncSession):
        """Check if user has exceeded file quota"""
        query = select(func.count()).select_from(FileMetadata).where(
            and_(
                FileMetadata.owner_id == user_id,
                FileMetadata.status == "active"
            )
        )
        result = await db.execute(query)
        file_count = result.scalar()
        
        if file_count >= MAX_FILES_PER_USER:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File quota exceeded. Maximum {MAX_FILES_PER_USER} files allowed"
            )
    
    async def _check_duplicate(
        self,
        checksum: str,
        user_id: str,
        db: AsyncSession
    ) -> Optional[FileMetadata]:
        """Check for duplicate files"""
        query = select(FileMetadata).where(
            and_(
                FileMetadata.checksum == checksum,
                FileMetadata.owner_id == user_id,
                FileMetadata.status == "active"
            )
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def _check_access(
        self,
        file_metadata: FileMetadata,
        user: User,
        db: AsyncSession
    ) -> bool:
        """Check if user has access to file"""
        # Owner always has access
        if file_metadata.owner_id == user.id:
            return True
        
        # Public files are accessible
        if file_metadata.is_public:
            return True
        
        # Check file shares
        query = select(FileShare).where(
            and_(
                FileShare.file_id == file_metadata.file_id,
                FileShare.user_id == user.id,
                or_(
                    FileShare.expires_at.is_(None),
                    FileShare.expires_at > datetime.utcnow()
                )
            )
        )
        result = await db.execute(query)
        share = result.scalar_one_or_none()
        
        return share is not None
    
    def _get_storage_subdir(
        self,
        user_id: str,
        project_id: Optional[str]
    ) -> str:
        """Generate storage subdirectory path"""
        if project_id:
            return f"projects/{project_id[:2]}/{project_id}"
        return f"users/{user_id[:2]}/{user_id}"
    
    def _get_thumbnail_path(self, file_id: str) -> Path:
        """Get thumbnail file path"""
        return FILE_STORAGE_PATH / "thumbnails" / f"{file_id}_thumb.jpg"
    
    async def _generate_thumbnail(
        self,
        file_path: Path,
        file_id: str,
        db: AsyncSession
    ):
        """Generate thumbnail for image file"""
        try:
            thumbnail_dir = FILE_STORAGE_PATH / "thumbnails"
            thumbnail_dir.mkdir(parents=True, exist_ok=True)
            
            thumbnail_path = self._get_thumbnail_path(file_id)
            
            with Image.open(file_path) as img:
                img.thumbnail((300, 300))
                img.save(thumbnail_path, "JPEG", quality=85)
            
            self.logger.info("Thumbnail generated", file_id=file_id)
            
        except Exception as e:
            self.logger.error("Thumbnail generation failed", error=str(e), file_id=file_id)
    
    def _create_upload_response(self, file_metadata: FileMetadata) -> FileUploadResponse:
        """Create file upload response"""
        thumbnail_url = None
        if file_metadata.mime_type.startswith('image/'):
            thumbnail_url = f"/api/v1/files/{file_metadata.file_id}/thumbnail"
        
        return FileUploadResponse(
            file_id=file_metadata.file_id,
            filename=file_metadata.filename,
            original_filename=file_metadata.original_filename,
            file_size=file_metadata.file_size,
            mime_type=file_metadata.mime_type,
            checksum=file_metadata.checksum,
            upload_timestamp=file_metadata.upload_timestamp,
            storage_path=file_metadata.storage_path,
            url=f"/api/v1/files/{file_metadata.file_id}",
            thumbnail_url=thumbnail_url
        )
    
    def _create_metadata_response(self, file_metadata: FileMetadata) -> FileMetadataResponse:
        """Create file metadata response"""
        return FileMetadataResponse(
            file_id=file_metadata.file_id,
            filename=file_metadata.filename,
            original_filename=file_metadata.original_filename,
            file_size=file_metadata.file_size,
            mime_type=file_metadata.mime_type,
            checksum=file_metadata.checksum,
            upload_timestamp=file_metadata.upload_timestamp,
            updated_timestamp=file_metadata.updated_timestamp,
            owner_id=file_metadata.owner_id,
            project_id=file_metadata.project_id,
            folder_path=file_metadata.folder_path,
            tags=file_metadata.tags,
            description=file_metadata.description,
            is_public=file_metadata.is_public,
            download_count=file_metadata.download_count,
            version=file_metadata.version,
            status=file_metadata.status
        )

# ===============================================================================
# ROUTER SETUP
# ===============================================================================

router = APIRouter(prefix="/api/v1/files", tags=["files"])
file_service = FileService()

# ===============================================================================
# ROUTE HANDLERS
# ===============================================================================

@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None),
    folder_path: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    is_public: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Upload a file with metadata.
    
    Supports:
    - Multiple file types
    - Virus scanning
    - Duplicate detection
    - Thumbnail generation
    - Access control
    """
    request_data = FileUploadRequest(
        project_id=project_id,
        folder_path=folder_path,
        tags=[t.strip() for t in tags.split(",")] if tags else [],
        description=description,
        is_public=is_public
    )
    
    return await file_service.upload_file(
        file=file,
        user=current_user,
        db=db,
        request_data=request_data,
        background_tasks=background_tasks
    )

@router.get("/{file_id}", response_class=FileResponse)
async def download_file(
    file_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Download a file"""
    return await file_service.download_file(file_id, current_user, db)

@router.get("/{file_id}/metadata", response_model=FileMetadataResponse)
async def get_file_metadata(
    file_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Get file metadata"""
    query = select(FileMetadata).where(FileMetadata.file_id == file_id)
    result = await db.execute(query)
    file_metadata = result.scalar_one_or_none()
    
    if not file_metadata:
        raise HTTPException(status_code=404, detail="File not found")
    
    has_access = await file_service._check_access(file_metadata, current_user, db)
    if not has_access:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return file_service._create_metadata_response(file_metadata)

@router.get("/", response_model=FileListResponse)
async def list_files(
    project_id: Optional[str] = Query(None),
    folder_path: Optional[str] = Query(None),
    tags: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """List files with filtering and pagination"""
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    
    return await file_service.list_files(
        user=current_user,
        db=db,
        project_id=project_id,
        folder_path=folder_path,
        tags=tag_list,
        search=search,
        page=page,
        page_size=page_size
    )

@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Delete a file"""
    return await file_service.delete_file(file_id, current_user, db)

@router.get("/{file_id}/thumbnail", response_class=FileResponse)
async def get_thumbnail(
    file_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Get file thumbnail (images only)"""
    thumbnail_path = file_service._get_thumbnail_path(file_id)
    
    if not thumbnail_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    return FileResponse(
        path=thumbnail_path,
        media_type="image/jpeg"
    )

@router.post("/{file_id}/share")
async def share_file(
    file_id: str,
    share_request: FileShareRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Share file with other users"""
    # Get file metadata
    query = select(FileMetadata).where(FileMetadata.file_id == file_id)
    result = await db.execute(query)
    file_metadata = result.scalar_one_or_none()
    
    if not file_metadata:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Check ownership
    if file_metadata.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only owner can share files")
    
    # Create file shares
    shares_created = []
    for user_id in share_request.user_ids:
        file_share = FileShare(
            file_id=file_id,
            user_id=user_id,
            permission=share_request.permission,
            expires_at=share_request.expires_at,
            shared_by=current_user.id,
            shared_at=datetime.utcnow()
        )
        db.add(file_share)
        shares_created.append(user_id)
    
    await db.commit()
    
    logger.info(
        "File shared",
        file_id=file_id,
        shared_with=shares_created,
        owner_id=current_user.id
    )
    
    return {
        "status": "success",
        "message": f"File shared with {len(shares_created)} users",
        "shared_with": shares_created
    }

# ===============================================================================
# EXPORTS
# ===============================================================================

__all__ = [
    "router",
    "FileService",
    "FileUploadRequest",
    "FileUploadResponse",
    "FileMetadataResponse",
    "FileListResponse"
]
