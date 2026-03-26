"""
Email tools for sending SMTP emails with attachments.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import smtplib
import uuid
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from langchain_core.tools import tool

from config import Settings

logger = logging.getLogger(__name__)


def create_email_tools(settings: Settings) -> list:
    """Build email tools with SMTP settings injected."""

    @tool
    def send_email(
        to: list[str],
        subject: str,
        body: str,
        body_format: str = "html",
        attachments: list[str] | None = None,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
    ) -> dict[str, Any]:
        """Send an email via SMTP with optional file attachments.

        Args:
            to: List of recipient email addresses.
            subject: Email subject line.
            body: Email body content.
            body_format: 'html' or 'plain'.
            attachments: List of file paths to attach.
            cc: CC recipients.
            bcc: BCC recipients.

        Returns:
            Dict with success status and message_id.
        """
        if not settings.smtp_user or not settings.smtp_from:
            return {
                "success": False,
                "error": "SMTP not configured. Set AGENT_SMTP_* env vars.",
                "message_id": None,
            }

        msg = MIMEMultipart()
        msg["From"] = settings.smtp_from
        msg["To"] = ", ".join(to)
        msg["Subject"] = subject

        if cc:
            msg["Cc"] = ", ".join(cc)

        msg_id = f"<{uuid.uuid4()}@agent-platform>"
        msg["Message-ID"] = msg_id

        # Body
        msg.attach(MIMEText(body, body_format, "utf-8"))

        # Attachments
        for filepath in attachments or []:
            if not os.path.isfile(filepath):
                logger.warning("Attachment not found: %s", filepath)
                continue

            mime_type, _ = mimetypes.guess_type(filepath)
            if mime_type is None:
                mime_type = "application/octet-stream"

            maintype, subtype = mime_type.split("/", 1)
            with open(filepath, "rb") as f:
                part = MIMEBase(maintype, subtype)
                part.set_payload(f.read())

            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                "attachment",
                filename=os.path.basename(filepath),
            )
            msg.attach(part)

        # Send
        all_recipients = list(to) + (cc or []) + (bcc or [])
        try:
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
                if settings.smtp_use_tls:
                    server.starttls()
                server.login(
                    settings.smtp_user,
                    settings.smtp_password.get_secret_value(),
                )
                server.sendmail(settings.smtp_from, all_recipients, msg.as_string())

            logger.info(
                "Email sent: to=%s subject=%s attachments=%d",
                to,
                subject,
                len(attachments or []),
            )
            return {"success": True, "message_id": msg_id, "error": None}

        except Exception as e:
            logger.error("Email send failed: %s", e)
            return {"success": False, "message_id": None, "error": str(e)}

    return [send_email]
