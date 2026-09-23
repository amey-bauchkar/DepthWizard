"""Application error taxonomy (Phase 7 §28). Every error carries a stable code, a user-facing message
and a flag saying whether processing can continue in a degraded mode. Internals go to the developer log."""
from __future__ import annotations


class DepthWizardError(Exception):
    code: str = "INTERNAL_ERROR"
    user_message: str = "Unable to process image."
    recoverable: bool = False
    http_status: int = 500

    def __init__(self, detail: str = "", *, user_message: str | None = None):
        super().__init__(detail or self.user_message)
        self.detail = detail
        if user_message:
            self.user_message = user_message

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.user_message,
            "detail": self.detail,
            "recoverable": self.recoverable,
        }


class InvalidFileError(DepthWizardError):
    code = "INVALID_FILE"
    user_message = "Unsupported or corrupted image."
    http_status = 400


class UnsupportedFormatError(DepthWizardError):
    code = "UNSUPPORTED_FORMAT"
    user_message = "Unsupported file type. Supported: PNG, JPEG."
    http_status = 400


class ImageTooLargeError(DepthWizardError):
    code = "IMAGE_TOO_LARGE"
    user_message = "Image exceeds configured processing limit."
    recoverable = True
    http_status = 413


class EmptyInputError(DepthWizardError):
    code = "EMPTY_INPUT"
    user_message = "The uploaded file is empty."
    http_status = 400


class ModelUnavailableError(DepthWizardError):
    code = "MODEL_UNAVAILABLE"
    user_message = "Model weights are not installed."
    http_status = 503


class InferenceFailedError(DepthWizardError):
    code = "INFERENCE_FAILED"
    user_message = "Depth inference failed. No result was produced."
    http_status = 500


class JobNotFoundError(DepthWizardError):
    code = "JOB_NOT_FOUND"
    user_message = "Job not found."
    http_status = 404


class JobStateError(DepthWizardError):
    code = "JOB_STATE"
    user_message = "Job is not in a state that allows this action."
    http_status = 409


class VerticalTransformUnsafeError(DepthWizardError):
    """Phase 8 C-1: PROJ selected a 'ballpark' (no-op) vertical transformation because geoid grids are missing."""

    code = "VERTICAL_TRANSFORM_UNSAFE"
    user_message = "Vertical reference conversion is unavailable (geoid grids missing). Absolute elevation cannot be produced."
    recoverable = True
    http_status = 422


class GridsMissingError(VerticalTransformUnsafeError):
    code = "GEOID_GRIDS_MISSING"
