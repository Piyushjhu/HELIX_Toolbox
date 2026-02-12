"""Girder client wrapper for HTMDEC data portal (data.htmdec.org)."""
from __future__ import annotations

import io
import logging
import os
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from girder_client import GirderClient

logger = logging.getLogger("helix.girder")

DEFAULT_API_URL = "https://data.htmdec.org/api/v1"


class HelixGirderClient:
    """Thin wrapper around GirderClient tailored for HELIX workflows.

    Usage::

        client = HelixGirderClient()          # reads env vars
        client = HelixGirderClient(api_url=..., api_key=...)  # explicit

        # Browse
        folders = client.list_folders(collection_name="ALPSS")
        items = client.list_items(folder_id)

        # Download
        local_paths = client.download_folder(folder_id, "/tmp/pdv_data")

        # Upload
        client.upload_results("/tmp/output", target_folder_id)
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.api_url = api_url or os.environ.get("GIRDER_API_URL", DEFAULT_API_URL)
        self.api_key = api_key or os.environ.get(
            "HTMDEC_GIRDER_API_KEY",
            os.environ.get("GIRDER_API_KEY", ""),
        )
        self._gc: Optional[GirderClient] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> GirderClient:
        """Authenticate and return the underlying GirderClient."""
        if self._gc is not None:
            return self._gc
        self._gc = GirderClient(apiUrl=self.api_url)
        if self.api_key:
            self._gc.authenticate(apiKey=self.api_key)
            logger.info("Authenticated to %s", self.api_url)
        else:
            logger.warning("No API key provided — connecting anonymously to %s", self.api_url)
        return self._gc

    @property
    def gc(self) -> GirderClient:
        return self.connect()

    @property
    def connected(self) -> bool:
        return self._gc is not None

    # ------------------------------------------------------------------
    # Browsing
    # ------------------------------------------------------------------

    def list_collections(self) -> List[Dict[str, Any]]:
        """Return all visible collections."""
        return list(self.gc.listCollection())

    def find_collection(self, name: str) -> Optional[Dict[str, Any]]:
        """Find a collection by name."""
        for c in self.gc.listCollection():
            if c["name"] == name:
                return c
        return None

    def list_folders(
        self,
        parent_id: Optional[str] = None,
        parent_type: str = "folder",
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List child folders. If *collection_name* is given, list its top-level folders."""
        if collection_name:
            col = self.find_collection(collection_name)
            if col is None:
                raise ValueError(f"Collection '{collection_name}' not found")
            parent_id = col["_id"]
            parent_type = "collection"
        return list(self.gc.listFolder(parent_id, parentFolderType=parent_type))

    def list_items(self, folder_id: str, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List items in a folder, optionally filtered by name."""
        kwargs: Dict[str, Any] = {}
        if name:
            kwargs["name"] = name
        return list(self.gc.listItem(folder_id, **kwargs))

    def list_files(self, item_id: str) -> List[Dict[str, Any]]:
        """List files attached to an item."""
        return list(self.gc.listFile(item_id))

    def get_folder(self, folder_id: str) -> Dict[str, Any]:
        """Get folder metadata."""
        return self.gc.getFolder(folder_id)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_file(self, file_id: str) -> io.BytesIO:
        """Download a single file into a BytesIO buffer."""
        buf = io.BytesIO()
        self.gc.downloadFile(file_id, buf)
        buf.seek(0)
        return buf

    def download_file_to_path(self, file_id: str, local_path: str) -> str:
        """Download a single file to a local path."""
        buf = self.download_file(file_id)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(buf.read())
        logger.info("Downloaded file %s → %s", file_id, local_path)
        return local_path

    def download_item(self, item_id: str, dest_dir: str) -> str:
        """Download all files in an item to *dest_dir*."""
        os.makedirs(dest_dir, exist_ok=True)
        self.gc.downloadItem(item_id, dest_dir)
        return dest_dir

    def download_folder(
        self,
        folder_id: str,
        dest_dir: str,
        pattern: str = "*.csv",
    ) -> List[str]:
        """Download all CSV items from a Girder folder to *dest_dir*.

        Returns list of local file paths.
        """
        os.makedirs(dest_dir, exist_ok=True)
        downloaded: List[str] = []

        for item in self.gc.listItem(folder_id):
            item_name = item["name"]
            if pattern != "*" and not Path(item_name).match(pattern):
                continue

            files = list(self.gc.listFile(item["_id"]))
            if not files:
                continue

            local_path = os.path.join(dest_dir, item_name)
            self.download_file_to_path(files[0]["_id"], local_path)
            downloaded.append(local_path)

        logger.info("Downloaded %d files from folder %s", len(downloaded), folder_id)
        return downloaded

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_file(
        self,
        local_path: str,
        folder_id: str,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a single file to a Girder folder."""
        fname = filename or os.path.basename(local_path)
        mime, _ = mimetypes.guess_type(fname)
        mime = mime or "application/octet-stream"
        result = self.gc.uploadFileToFolder(
            folder_id, local_path, mimeType=mime, filename=fname,
        )
        logger.info("Uploaded %s → folder %s", fname, folder_id)
        return result

    def upload_results(
        self,
        output_dir: str,
        target_folder_id: str,
        subfolder_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload all result files from *output_dir* to Girder.

        If *subfolder_name* is given, creates a child folder first.
        Returns the folder metadata dict.
        """
        if subfolder_name:
            folder = self.gc.createFolder(
                target_folder_id, subfolder_name,
                parentType="folder", reuseExisting=True,
            )
            target_folder_id = folder["_id"]
        else:
            folder = self.gc.getFolder(target_folder_id)

        for fname in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, fname)
            if os.path.isfile(fpath):
                self.upload_file(fpath, target_folder_id)

        logger.info("Uploaded results from %s → folder %s", output_dir, target_folder_id)
        return folder

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def add_item_metadata(self, item_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata to an item."""
        return self.gc.addMetadataToItem(item_id, metadata)

    def add_folder_metadata(self, folder_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata to a folder."""
        return self.gc.addMetadataToFolder(folder_id, metadata)
