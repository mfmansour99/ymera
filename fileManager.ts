import * as fs from 'fs/promises';
import * as path from 'path';
import type { InsertFile } from "@shared/schema";

export class FileManager {
  private uploadDir: string;

  constructor() {
    this.uploadDir = process.env.UPLOAD_DIR || path.join(process.cwd(), 'uploads');
    this.ensureUploadDirectory();
  }

  private async ensureUploadDirectory(): Promise<void> {
    try {
      await fs.access(this.uploadDir);
    } catch {
      await fs.mkdir(this.uploadDir, { recursive: true });
    }
  }

  async saveFile(buffer: Buffer, fileData: Omit<InsertFile, 'id' | 'createdAt' | 'updatedAt'>): Promise<InsertFile> {
    const filePath = path.join(this.uploadDir, fileData.filename);
    
    try {
      await fs.writeFile(filePath, buffer);
      
      return {
        ...fileData,
        version: fileData.version || 1,
        downloadCount: fileData.downloadCount || 0,
        isPublic: fileData.isPublic || false,
      };
    } catch (error) {
      console.error('Error saving file:', error);
      throw new Error(`Failed to save file: ${error.message}`);
    }
  }

  async getFile(filename: string): Promise<Buffer> {
    const filePath = path.join(this.uploadDir, filename);
    
    try {
      return await fs.readFile(filePath);
    } catch (error) {
      console.error('Error reading file:', error);
      throw new Error(`Failed to read file: ${error.message}`);
    }
  }

  async deleteFile(filename: string): Promise<void> {
    const filePath = path.join(this.uploadDir, filename);
    
    try {
      await fs.unlink(filePath);
    } catch (error) {
      console.error('Error deleting file:', error);
      throw new Error(`Failed to delete file: ${error.message}`);
    }
  }

  async fileExists(filename: string): Promise<boolean> {
    const filePath = path.join(this.uploadDir, filename);
    
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }

  async getFileStats(filename: string): Promise<{ size: number; mtime: Date }> {
    const filePath = path.join(this.uploadDir, filename);
    
    try {
      const stats = await fs.stat(filePath);
      return {
        size: stats.size,
        mtime: stats.mtime,
      };
    } catch (error) {
      console.error('Error getting file stats:', error);
      throw new Error(`Failed to get file stats: ${error.message}`);
    }
  }

  async createFileVersion(originalFilename: string, newBuffer: Buffer): Promise<string> {
    const ext = path.extname(originalFilename);
    const basename = path.basename(originalFilename, ext);
    const timestamp = Date.now();
    const versionedFilename = `${basename}_v${timestamp}${ext}`;
    
    const filePath = path.join(this.uploadDir, versionedFilename);
    await fs.writeFile(filePath, newBuffer);
    
    return versionedFilename;
  }

  getFileUrl(filename: string): string {
    return `/api/files/${filename}/download`;
  }
}
