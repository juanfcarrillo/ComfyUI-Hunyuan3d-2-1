"""
Cloudflare R2 Storage Uploader Utility
=====================================

This module provides functionality to upload 3D model files (GLB, OBJ, etc.)
to Cloudflare R2 storage using boto3.

Environment Variables Required:
- R2_ACCOUNT_ID: Your Cloudflare account ID
- R2_ACCESS_KEY_ID: R2 access key ID
- R2_SECRET_ACCESS_KEY: R2 secret access key
- R2_BUCKET_NAME: Name of your R2 bucket

Example usage:
    uploader = R2Uploader()
    url = uploader.upload_file("/path/to/model.glb", "models/my_model.glb")
    print(f"File uploaded: {url}")
"""

import os
import boto3
import mimetypes
import time
from pathlib import Path
from typing import Optional, Dict, Any
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
load_dotenv()


class R2Uploader:
    """Cloudflare R2 storage uploader using boto3"""

    def __init__(self):
        """Initialize R2 uploader with environment variables"""
        # Load configuration from environment variables
        self.account_id = os.getenv("R2_ACCOUNT_ID")
        self.access_key_id = os.getenv("R2_ACCESS_KEY_ID")
        self.secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("R2_BUCKET_NAME")
        self.public_url = os.getenv("R2_PUBLIC_URL")

        # Validate required environment variables
        required_vars = {
            "R2_ACCOUNT_ID": self.account_id,
            "R2_ACCESS_KEY_ID": self.access_key_id,
            "R2_SECRET_ACCESS_KEY": self.secret_access_key,
            "R2_BUCKET_NAME": self.bucket_name,
        }

        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        # Build endpoint URL
        self.endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"

        # Initialize boto3 resource
        self.s3 = None
        self.bucket = None
        self._initialize_client()

        print(f"✅ R2 Uploader initialized for bucket: {self.bucket_name}")

    def _initialize_client(self):
        """Initialize the boto3 S3 resource for R2"""
        try:
            self.s3 = boto3.resource(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
            )

            # Get bucket reference
            self.bucket = self.s3.Bucket(self.bucket_name)

            # Test connection by checking if bucket exists
            self.s3.meta.client.head_bucket(Bucket=self.bucket_name)
            print(f"🔗 Successfully connected to R2 bucket: {self.bucket_name}")

        except NoCredentialsError as e:
            raise ValueError("Invalid R2 credentials provided") from e
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                raise ValueError(f"R2 bucket '{self.bucket_name}' not found") from e
            elif error_code == "403":
                raise ValueError(
                    "Access denied to R2 bucket. Check your credentials and permissions"
                ) from e
            else:
                raise ValueError(f"Failed to connect to R2: {e}") from e

    def _get_content_type(self, file_path: str) -> str:
        """Get content type for the file"""
        content_type, _ = mimetypes.guess_type(file_path)

        # Map common 3D model formats
        extension_map = {
            ".glb": "model/gltf-binary",
            ".gltf": "model/gltf+json",
            ".obj": "model/obj",
            ".ply": "application/octet-stream",
            ".stl": "model/stl",
            ".fbx": "application/octet-stream",
        }

        file_ext = Path(file_path).suffix.lower()
        if file_ext in extension_map:
            return extension_map[file_ext]

        return content_type or "application/octet-stream"

    def upload_file(
        self,
        local_file_path: str,
        remote_key: str,
        make_public: bool = True,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload a file to R2 storage

        Args:
            local_file_path: Path to the local file
            remote_key: Key (path) for the file in R2
            make_public: Whether to make the file publicly accessible
            metadata: Optional metadata to attach to the file

        Returns:
            Public URL of the uploaded file
        """
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file not found: {local_file_path}")

        try:
            # Prepare upload parameters
            file_size = os.path.getsize(local_file_path)
            content_type = self._get_content_type(local_file_path)

            upload_kwargs = {
                "ContentType": content_type,
            }

            # Add metadata if provided
            if metadata:
                upload_kwargs["Metadata"] = metadata

            # Set ACL for public access if requested
            if make_public:
                upload_kwargs["ACL"] = "public-read"

            print(f"📤 Uploading {os.path.basename(local_file_path)} to R2...")
            print(f"   Local: {local_file_path}")
            print(f"   Remote: {remote_key}")
            print(f"   Size: {file_size:,} bytes")
            print(f"   Content-Type: {content_type}")

            # Upload the file
            self.bucket.upload_file(
                local_file_path, remote_key, ExtraArgs=upload_kwargs
            )

            # Generate public URL
            public_url = self._get_public_url(remote_key)

            print("✅ Upload successful!")
            print(f"🔗 Public URL: {public_url}")

            return public_url

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            raise RuntimeError(f"Failed to upload file to R2 (Error {error_code}): {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during upload: {e}") from e

    def _get_public_url(self, remote_key: str) -> str:
        """Generate public URL for the uploaded file"""
        # Use default R2 public URL format
        return f"{self.public_url}/{remote_key}"

    def upload_multiple_files(
        self,
        file_mappings: Dict[str, str],
        make_public: bool = True,
        base_metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Upload multiple files to R2

        Args:
            file_mappings: Dict mapping local_path -> remote_key
            make_public: Whether to make files publicly accessible
            base_metadata: Base metadata to apply to all files

        Returns:
            Dict mapping local_path -> public_url
        """
        results = {}

        for local_path, remote_key in file_mappings.items():
            try:
                # Add file-specific metadata
                metadata = base_metadata.copy() if base_metadata else {}
                metadata.update(
                    {
                        "original_filename": os.path.basename(local_path),
                        "upload_timestamp": str(int(time.time())),
                    }
                )

                public_url = self.upload_file(
                    local_path, remote_key, make_public=make_public, metadata=metadata
                )
                results[local_path] = public_url

            except Exception as e:
                print(f"❌ Failed to upload {local_path}: {e}")
                results[local_path] = None

        return results

    def delete_file(self, remote_key: str) -> bool:
        """
        Delete a file from R2

        Args:
            remote_key: Key of the file to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.bucket.Object(remote_key).delete()
            print(f"🗑️ Deleted file: {remote_key}")
            return True
        except ClientError as e:
            print(f"❌ Failed to delete {remote_key}: {e}")
            return False

    def list_files(self, prefix: str = "", limit: int = 100) -> list:
        """
        List files in the R2 bucket

        Args:
            prefix: Prefix to filter files
            limit: Maximum number of files to return

        Returns:
            List of file information
        """
        try:
            files = []
            count = 0

            for obj in self.bucket.objects.filter(Prefix=prefix):
                if count >= limit:
                    break

                files.append(
                    {
                        "key": obj.key,
                        "size": obj.size,
                        "last_modified": obj.last_modified,
                        "public_url": self._get_public_url(obj.key),
                    }
                )
                count += 1

            return files

        except ClientError as e:
            print(f"❌ Failed to list files: {e}")
            return []

    def get_file_info(self, remote_key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a file in R2

        Args:
            remote_key: Key of the file

        Returns:
            File information dict or None if not found
        """
        try:
            obj = self.bucket.Object(remote_key)
            obj.load()  # This will raise an exception if the object doesn't exist

            return {
                "key": remote_key,
                "size": obj.content_length,
                "content_type": obj.content_type,
                "last_modified": obj.last_modified,
                "metadata": obj.metadata,
                "public_url": self._get_public_url(remote_key),
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return None
            else:
                print(f"❌ Failed to get file info: {e}")
                return None


def create_model_uploader() -> R2Uploader:
    """Factory function to create an R2 uploader instance"""
    return R2Uploader()


def upload_3d_model(
    local_file_path: str,
    model_name: str,
    workflow_type: str = "unknown",
    additional_metadata: Optional[Dict[str, str]] = None,
) -> str:
    """
    Convenience function to upload a 3D model with standard naming and metadata

    Args:
        local_file_path: Path to the local 3D model file
        model_name: Name for the model (used in remote path)
        workflow_type: Type of workflow used (mesh, texture, enhanced)
        additional_metadata: Additional metadata to attach

    Returns:
        Public URL of the uploaded file
    """
    uploader = create_model_uploader()

    # Create a structured remote path
    file_extension = Path(local_file_path).suffix
    timestamp = int(time.time())
    remote_key = f"models/{workflow_type}/{model_name}_{timestamp}{file_extension}"

    # Prepare metadata
    metadata = {
        "workflow_type": workflow_type,
        "model_name": model_name,
        "generated_timestamp": str(timestamp),
        "file_type": "3d_model",
    }

    if additional_metadata:
        metadata.update(additional_metadata)

    return uploader.upload_file(
        local_file_path=local_file_path,
        remote_key=remote_key,
        make_public=True,
        metadata=metadata,
    )


# Test functions
def test_r2_connection():
    """Test R2 connection and list bucket contents"""
    try:
        uploader = R2Uploader()
        files = uploader.list_files(limit=10)

        print(f"✅ R2 connection successful!")
        print(f"📁 Found {len(files)} files in bucket")

        for file_info in files[:5]:  # Show first 5 files
            print(f"   - {file_info['key']} ({file_info['size']:,} bytes)")

        return True

    except Exception as e:
        print(f"❌ R2 connection failed: {e}")
        return False


def test_upload_sample():
    """Test uploading a sample file"""
    # Create a small test file
    test_file = "test_model.txt"
    test_content = "This is a test 3D model file for R2 upload testing."

    try:
        # Create test file
        with open(test_file, "w") as f:
            f.write(test_content)

        # Upload test file
        url = upload_3d_model(
            local_file_path=test_file,
            model_name="test_upload",
            workflow_type="test",
            additional_metadata={"test": "true", "purpose": "validation"},
        )

        print(f"✅ Test upload successful: {url}")

        # Clean up local test file
        os.remove(test_file)

        return url

    except Exception as e:
        print(f"❌ Test upload failed: {e}")
        # Clean up on failure
        if os.path.exists(test_file):
            os.remove(test_file)
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-connection":
            test_r2_connection()
        elif sys.argv[1] == "--test-upload":
            test_upload_sample()
        elif sys.argv[1] == "--list":
            uploader = R2Uploader()
            files = uploader.list_files()
            print(f"Files in bucket ({len(files)}):")
            for file_info in files:
                print(f"  {file_info['key']} - {file_info['public_url']}")
        else:
            print(
                "Usage: python r2_uploader.py [--test-connection|--test-upload|--list]"
            )
    else:
        print("R2 Uploader utility - use --test-connection to verify setup")
